import os
# Monkeypatch OGB to bypass download confirmation
import ogb.utils.url
ogb.utils.url.decide_download = lambda url: True

# Disable all tqdm progress bars
os.environ["TQDM_DISABLE"] = "1"

os.environ['DGLBACKEND'] = 'pytorch'
try:
    if not os.access(os.path.expanduser("~"), os.W_OK):
        os.environ['DGL_HOME'] = os.path.join(os.getcwd(), ".dgl")
except Exception:
    os.environ['DGL_HOME'] = os.path.join(os.getcwd(), ".dgl")

import argparse, time, dgl, dgl.nn as dglnn, torch, torch.nn as nn, torch.nn.functional as F, torchmetrics.functional as MF, tqdm, sys
# Suppress download progress
os.environ['DGL_DOWNLOAD_PROGRESS'] = '0'
from dgl.data import AsNodePredDataset, RedditDataset, YelpDataset, PubmedGraphDataset, FlickrDataset, CoraGraphDataset, FraudYelpDataset as LegacyFraudYelpDataset
from dgl.dataloading import DataLoader, MultiLayerFullNeighborSampler, NeighborSampler
from ogb.nodeproppred import DglNodePropPredDataset

class GAT(nn.Module):
    def __init__(self, in_size, hid_size, out_size, num_layers, num_heads):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.GATConv(in_size, hid_size // num_heads, num_heads, activation=F.relu, allow_zero_in_degree=True))
        for _ in range(num_layers - 2):
            self.layers.append(dglnn.GATConv(hid_size, hid_size // num_heads, num_heads, activation=F.relu, allow_zero_in_degree=True))
        self.layers.append(dglnn.GATConv(hid_size, out_size, 1, allow_zero_in_degree=True))
        self.dropout = nn.Dropout(0.5); self.hid_size, self.out_size = hid_size, out_size

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1: h = h.flatten(1); h = self.dropout(h)
            else: h = h.mean(1)
        return h

    def inference(self, g, device, batch_size):
        feat = g.ndata["feat"]
        sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=["feat"])
        dataloader = DataLoader(g, torch.arange(g.num_nodes()).to(g.device), sampler, device=device, batch_size=batch_size, shuffle=False, drop_last=False)
        for l, layer in enumerate(self.layers):
            y = torch.empty(g.num_nodes(), self.hid_size if l != len(self.layers) - 1 else self.out_size, dtype=feat.dtype, device="cpu")
            feat = feat.to(device)
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader, disable=not sys.stdout.isatty()):
                x = feat[input_nodes]; h = layer(blocks[0], x)
                if l != len(self.layers) - 1: h = h.flatten(1); h = self.dropout(h)
                else: h = h.mean(1)
                y[output_nodes] = h.to("cpu")
            feat = y
        return y

def evaluate(model, g, idx, num_classes, is_multilabel, device):
    model.eval()
    with torch.no_grad():
        pred = model.inference(g, device, 4096)[idx]
        label = g.ndata["label"][idx].to(pred.device)
        return MF.accuracy(pred, label, task="multiclass", num_classes=num_classes).item()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--dataset", default="reddit")
    parser.add_argument("--mode", default="puregpu"); parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--num_layers", type=int, default=3); parser.add_argument("--hid_size", type=int, default=256)
    parser.add_argument("--fan_out", type=str, default="10,10,10"); parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_heads", type=int, default=4); parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument('--path', type=str, default='/data/dgl_lab'); args = parser.parse_args()
    device = torch.device("cuda" if args.mode != "cpu" else "cpu")
    run_id = f"{int(time.time())}_{os.getpid()}"
    data_path = args.path if os.path.exists(args.path) else os.path.expanduser("~/.dgl")
    if args.dataset.startswith("ogbn-"):
        dataset = AsNodePredDataset(DglNodePropPredDataset(args.dataset, root=data_path)); g, num_classes = dataset[0], dataset.num_classes
    elif args.dataset == "reddit":
        dataset = RedditDataset(raw_dir=data_path); g, num_classes = dataset[0], dataset.num_classes
    elif args.dataset == "flickr":
        dataset = FlickrDataset(raw_dir=data_path); g, num_classes = dataset[0], dataset.num_classes
    elif args.dataset == "yelp":
        dataset = LegacyFraudYelpDataset(raw_dir=data_path); g_raw = dataset[0]; g = dgl.to_homogeneous(g_raw)
        g.ndata['feat'], g.ndata['label'] = g_raw.ndata['feature'], g_raw.ndata['label'].long()
        g.ndata['train_mask'], g.ndata['val_mask'], g.ndata['test_mask'] = g_raw.ndata['train_mask'], g_raw.ndata['val_mask'], g_raw.ndata['test_mask']
        num_classes = 2
    else:
        dataset = PubmedGraphDataset(raw_dir=data_path); g, num_classes = dataset[0], dataset.num_classes
    is_multilabel = g.ndata['label'].ndim > 1 and g.ndata['label'].shape[1] > 1
    train_idx = g.ndata['train_mask'].nonzero().squeeze(); val_idx = g.ndata['val_mask'].nonzero().squeeze(); test_idx = g.ndata['test_mask'].nonzero().squeeze()
    sampler = NeighborSampler([int(f) for f in args.fan_out.split(",")[:args.num_layers]], prefetch_node_feats=["feat"], prefetch_labels=["label"])
    train_loader = DataLoader(g, train_idx, sampler, device=device, batch_size=args.batch_size, shuffle=True, use_uva=(args.mode=="mixed"))
    model = GAT(g.ndata["feat"].shape[1], args.hid_size, num_classes, args.num_layers, args.num_heads).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4); best_val_acc = 0
    train_start = time.time()
    for epoch in range(args.epoch):
        model.train(); total_loss = 0
        for it, (input_nodes, output_nodes, blocks) in enumerate(train_loader):
            x = blocks[0].srcdata["feat"]; y = blocks[-1].dstdata["label"]
            y_hat = model(blocks, x); loss = F.binary_cross_entropy_with_logits(y_hat, y.float()) if is_multilabel else F.cross_entropy(y_hat, y.squeeze().long())
            opt.zero_grad(); loss.backward(); opt.step(); total_loss += loss.item()
        val_acc = evaluate(model, g, val_idx, num_classes, is_multilabel, device)
        if val_acc > best_val_acc: best_val_acc = val_acc; torch.save(model.state_dict(), f"best_baseline_gat_{args.dataset}_{run_id}.pt")

        print(f"Epoch {epoch:03d} | Loss {total_loss/(it+1):.4f} | Val Acc: {val_acc:.4f}")
    train_time = time.time() - train_start
    ckpt_path = f"best_baseline_gat_{args.dataset}_{run_id}.pt"
    model.load_state_dict(torch.load(ckpt_path))
    test_start = time.time()
    test_acc = evaluate(model, g, test_idx, num_classes, is_multilabel, device)
    test_time = time.time() - test_start
    
    import json
    gpu_peak = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
    orig_nodes = g.num_nodes(); orig_edges = g.num_edges()
    total_mem_access_train = orig_edges * args.num_layers * args.epoch * 2
    
    result = {
        "dataset": args.dataset, "method": "BASELINE_GAT", "run_id": run_id, "ratio": 1.0,
        "orig_nodes": int(orig_nodes), "orig_edges": int(orig_edges),
        "red_nodes": int(orig_nodes), "red_edges": int(orig_edges),
        "train_time": float(train_time), "test_time": float(test_time),
        "test_acc": float(test_acc), "best_val_acc": float(best_val_acc),
        "gpu_peak_mem_mb": float(gpu_peak), "total_mem_access_train": int(total_mem_access_train)
    }
    print(f"\n--- BASELINE RESULTS (GAT) ---\nBEST_VAL_ACC: {best_val_acc:.4f}\nFINAL_TEST_ACC: {test_acc:.4f}")
    print(f"\nPRINT_RESULT: {json.dumps(result, indent=4)}")
    if os.path.exists(ckpt_path): os.remove(ckpt_path)
