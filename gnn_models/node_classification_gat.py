import os
# Monkeypatch OGB to bypass download confirmation
import ogb.utils.url
ogb.utils.url.decide_download = lambda url: True

# Disable all tqdm progress bars
os.environ["TQDM_DISABLE"] = "1"

# Set DGL backend and home directory before importing DGL to avoid permission errors on clusters
os.environ['DGLBACKEND'] = 'pytorch'
try:
    if not os.access(os.path.expanduser("~"), os.W_OK):
        os.environ['DGL_HOME'] = os.path.join(os.getcwd(), ".dgl")
except Exception:
    os.environ['DGL_HOME'] = os.path.join(os.getcwd(), ".dgl")

import argparse, time, torch, torch.nn as nn, torch.nn.functional as F, torchmetrics.functional as MF, dgl, os, tqdm, sys
# Suppress download progress
os.environ['DGL_DOWNLOAD_PROGRESS'] = '0'
import dgl.nn as dglnn
from dgl.dataloading import DataLoader, NeighborSampler, MultiLayerFullNeighborSampler
from dgl.data import AsNodePredDataset, RedditDataset, YelpDataset, FlickrDataset, PubmedGraphDataset, CoraGraphDataset, FraudYelpDataset as LegacyFraudYelpDataset
from ogb.nodeproppred import DglNodePropPredDataset

class GAT(nn.Module):
    def __init__(self, in_size, hid_size, out_size, num_layers, num_heads):
        super().__init__()
        self.layers = nn.ModuleList()
        # Input layer
        self.layers.append(dglnn.GATConv(in_size, hid_size // num_heads, num_heads, activation=F.relu, allow_zero_in_degree=True))
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(dglnn.GATConv(hid_size, hid_size // num_heads, num_heads, activation=F.relu, allow_zero_in_degree=True))
        # Output layer
        self.layers.append(dglnn.GATConv(hid_size, out_size, 1, allow_zero_in_degree=True))
        self.dropout = nn.Dropout(0.5)
        self.hid_size, self.out_size = hid_size, out_size

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = h.flatten(1)
                h = self.dropout(h)
            else:
                h = h.mean(1)
        return h

    def inference(self, g, device, batch_size):
        feat = g.ndata['feat'].cpu()
        sampler = MultiLayerFullNeighborSampler(1)
        dataloader = DataLoader(g, torch.arange(g.num_nodes()).to(g.device), sampler, device=device, 
                                batch_size=batch_size, shuffle=False, drop_last=False)
        for l, layer in enumerate(self.layers):
            out_dim = self.hid_size if l != len(self.layers) - 1 else self.out_size
            y = torch.empty(g.num_nodes(), out_dim, device="cpu")
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader, desc=f"Inference Layer {l+1}", disable=not sys.stdout.isatty()):
                x = feat[input_nodes.cpu()].to(device)
                h = layer(blocks[0], x)
                if l != len(self.layers) - 1:
                    h = h.flatten(1); h = self.dropout(h)
                else:
                    h = h.mean(1)
                y[output_nodes.cpu()] = h.cpu()
            feat = y
        return y

def train(args, device, g_coarse, model, num_classes, is_multilabel, run_id, fine_tune_g=None):
    train_idx = g_coarse.ndata['train_mask'].nonzero().squeeze()
    val_idx = g_coarse.ndata['val_mask'].nonzero().squeeze()
    sampler = NeighborSampler([int(f) for f in args.fan_out.split(",")[:args.num_layers]])
    train_loader = DataLoader(g_coarse, train_idx.to(device), sampler, device=device, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(g_coarse, val_idx.to(device), sampler, device=device, batch_size=args.batch_size, shuffle=False)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    best_val_acc = 0.0
    ckpt_path = f"best_model_gat_{args.dataset}_{run_id}.pt"
    for epoch in range(args.epoch):
        model.train(); total_loss = 0
        for it, (input_nodes, output_nodes, blocks) in enumerate(train_loader):
            x = blocks[0].srcdata["feat"]; y = blocks[-1].dstdata["label"]
            y_hat = model(blocks, x)
            loss = F.binary_cross_entropy_with_logits(y_hat, y.float()) if is_multilabel else F.cross_entropy(y_hat, y.squeeze().long())
            opt.zero_grad(); loss.backward(); opt.step(); total_loss += loss.item()
        model.eval(); ys, y_hats = [], []
        with torch.no_grad():
            for _, (_, _, blocks) in enumerate(val_loader):
                ys.append(blocks[-1].dstdata["label"]); y_hats.append(model(blocks, blocks[0].srcdata["feat"]))
        val_preds, val_labels = torch.cat(y_hats), torch.cat(ys)
        acc = MF.accuracy(val_preds, val_labels.squeeze(), task="multiclass", num_classes=num_classes).item()
        if acc > best_val_acc: best_val_acc = acc; torch.save(model.state_dict(), ckpt_path)
        print(f"Epoch {epoch:03d} | Loss {total_loss/(it+1):.4f} | Val Acc: {acc:.4f}")
    if fine_tune_g is not None and args.ft_epoch > 0:
        model.load_state_dict(torch.load(ckpt_path))
        train_idx_orig = fine_tune_g.ndata['train_mask'].nonzero().squeeze()
        val_idx_orig = fine_tune_g.ndata['val_mask'].nonzero().squeeze()
        train_loader_orig = DataLoader(fine_tune_g, train_idx_orig.to(device), sampler, device=device, batch_size=args.batch_size, shuffle=True)
        val_loader_orig = DataLoader(fine_tune_g, val_idx_orig.to(device), sampler, device=device, batch_size=args.batch_size, shuffle=False)
        opt_ft = torch.optim.Adam(model.parameters(), lr=args.lr * 0.1, weight_decay=5e-4)
        for epoch in range(args.ft_epoch):
            model.train(); total_loss = 0
            for it, (input_nodes, output_nodes, blocks) in enumerate(train_loader_orig):
                x = blocks[0].srcdata["feat"]; y = blocks[-1].dstdata["label"]
                y_hat = model(blocks, x); loss = F.binary_cross_entropy_with_logits(y_hat, y.float()) if is_multilabel else F.cross_entropy(y_hat, y.squeeze().long())
                opt_ft.zero_grad(); loss.backward(); opt_ft.step(); total_loss += loss.item()
            model.eval(); ys, y_hats = [], []
            with torch.no_grad():
                for _, (_, _, blocks) in enumerate(val_loader_orig):
                    ys.append(blocks[-1].dstdata["label"]); y_hats.append(model(blocks, blocks[0].srcdata["feat"]))
            val_preds, val_labels = torch.cat(y_hats), torch.cat(ys)
            acc = MF.accuracy(val_preds, val_labels.squeeze(), task="multiclass", num_classes=num_classes).item()
            if acc > best_val_acc: acc = best_val_acc; torch.save(model.state_dict(), ckpt_path)
            print(f"FT Epoch {epoch:03d} | Loss {total_loss/(it+1):.4f} | Val Acc (Orig): {acc:.4f}")
    return best_val_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="puregpu", choices=["cpu", "mixed", "puregpu"])
    parser.add_argument("--dataset", default="reddit")
    parser.add_argument("--epoch", type=int, default=100); parser.add_argument("--ft_epoch", type=int, default=0)
    parser.add_argument("--num_layers", type=int, default=3); parser.add_argument("--hid_size", type=int, default=256)
    parser.add_argument("--fan_out", type=str, default="10,10,10"); parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_heads", type=int, default=4); parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument('--path', type=str, default='/data/dgl_lab'); args = parser.parse_args()
    args.dataset = args.dataset.strip().strip(','); device = torch.device("cuda" if args.mode != "cpu" else "cpu")
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
    coarsened_graph_name = args.dataset.split("-")[-1] + ".dgl"
    g_coarse = dgl.load_graphs(coarsened_graph_name)[0][0]
    g_coarse.ndata['label'] = g_coarse.ndata['label'].long(); g.ndata['label'] = g.ndata['label'].long()
    if args.mode == "puregpu": g, g_coarse = g.to(device), g_coarse.to(device)
    else: g_coarse = g_coarse.to(device); g.create_formats_()
    model = GAT(g_coarse.ndata["feat"].shape[1], args.hid_size, num_classes, args.num_layers, args.num_heads).to(device)
    train_start = time.time(); best_val_acc = train(args, device, g_coarse, model, num_classes, is_multilabel, run_id, fine_tune_g=g); train_time = time.time() - train_start
    ckpt_path = f"best_model_gat_{args.dataset}_{run_id}.pt"
    if os.path.exists(ckpt_path): model.load_state_dict(torch.load(ckpt_path))
    model.eval(); test_start = time.time()
    with torch.no_grad():
        full_preds = model.inference(g, device, 4096); test_idx = g.ndata["test_mask"].nonzero().squeeze()
        test_preds = full_preds[test_idx.cpu()].to(device); test_labels = g.ndata["label"][test_idx].to(device)
        test_acc = MF.accuracy(test_preds, test_labels.squeeze(), task="multiclass", num_classes=num_classes).item()
    test_time = time.time() - test_start
    
    import json
    gpu_peak = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
    orig_nodes = g.num_nodes(); orig_edges = g.num_edges()
    coarse_nodes = g_coarse.num_nodes(); coarse_edges = g_coarse.num_edges()
    total_mem_access_train = coarse_edges * args.num_layers * args.epoch * 2
    
    result = {
        "dataset": args.dataset, "method": "ESSC_GAT", "run_id": run_id, "ratio": float(coarse_nodes / orig_nodes),
        "orig_nodes": int(orig_nodes), "orig_edges": int(orig_edges),
        "red_nodes": int(coarse_nodes), "red_edges": int(coarse_edges),
        "train_time": float(train_time), "test_time": float(test_time),
        "test_acc": float(test_acc), "best_val_acc": float(best_val_acc),
        "gpu_peak_mem_mb": float(gpu_peak), "total_mem_access_train": int(total_mem_access_train)
    }
    print(f"\n--- RESULTS SUMMARY ---\nBEST_VAL_ACC: {best_val_acc:.4f}\nFINAL_TEST_ACC: {test_acc:.4f}\nTRAIN_TIME: {train_time:.4f}s\nTEST_TIME: {test_time:.4f}s")
    print(f"\nPRINT_RESULT: {json.dumps(result, indent=4)}")
    if os.path.exists(ckpt_path): os.remove(ckpt_path)
