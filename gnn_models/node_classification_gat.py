import os

# Set DGL backend and home directory before importing DGL to avoid permission errors on clusters
os.environ['DGLBACKEND'] = 'pytorch'
# Redirect DGL config directory to a writable location if the home directory is restricted
try:
    if not os.access(os.path.expanduser("~"), os.W_OK):
        os.environ['DGL_HOME'] = os.path.join(os.getcwd(), ".dgl")
except Exception:
    os.environ['DGL_HOME'] = os.path.join(os.getcwd(), ".dgl")

import argparse, time, torch, torch.nn as nn, torch.nn.functional as F, torchmetrics.functional as MF, dgl, tqdm, sys
import dgl.nn as dglnn
from dgl.dataloading import DataLoader, NeighborSampler, MultiLayerFullNeighborSampler
from dgl.data import AsNodePredDataset, RedditDataset, YelpDataset, FlickrDataset, PubmedGraphDataset, CoraGraphDataset, FraudYelpDataset as LegacyFraudYelpDataset
from ogb.nodeproppred import DglNodePropPredDataset

class GAT(nn.Module):
    def __init__(self, in_size, hid_size, out_size, num_layers, num_heads=8):
        super().__init__()
        self.layers = nn.ModuleList()
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        if num_layers == 1:
            self.layers.append(dglnn.GATConv(in_size, out_size, 1, allow_zero_in_degree=True))
        else:
            # input layer
            self.layers.append(dglnn.GATConv(in_size, hid_size, num_heads, allow_zero_in_degree=True))
            # hidden layers
            for _ in range(num_layers - 2):
                self.layers.append(dglnn.GATConv(hid_size * num_heads, hid_size, num_heads, allow_zero_in_degree=True))
            # output layer
            self.layers.append(dglnn.GATConv(hid_size * num_heads, out_size, 1, allow_zero_in_degree=True))
        self.dropout = nn.Dropout(0.5)
        self.hid_size, self.out_size = hid_size, out_size

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = h.flatten(1)
                h = F.relu(h)
                h = self.dropout(h)
            else:
                h = h.mean(1)
        return h

    def inference(self, g, device, batch_size):
        """Layer-wise inference to save memory. Intermediate features stay on CPU."""
        feat = g.ndata["feat"]
        sampler = MultiLayerFullNeighborSampler(1)
        dataloader = DataLoader(g, torch.arange(g.num_nodes()).to(g.device), sampler, device=device, 
                                batch_size=batch_size, shuffle=False, drop_last=False)
        
        for l, layer in enumerate(self.layers):
            if l != len(self.layers) - 1:
                out_dim = self.hid_size * self.num_heads
            else:
                out_dim = self.out_size
            
            y = torch.empty(g.num_nodes(), out_dim, device="cpu")
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader, desc=f"Inference Layer {l+1}"):
                x = feat[input_nodes.cpu()].to(device)
                h = layer(blocks[0], x)
                if l != len(self.layers) - 1:
                    h = h.flatten(1)
                    h = F.relu(h)
                    h = self.dropout(h)
                else:
                    h = h.mean(1)
                y[output_nodes.cpu()] = h.cpu()
            feat = y
        return y

def train(args, device, g_coarse, model, num_classes, is_multilabel):
    train_idx = g_coarse.ndata['train_mask'].nonzero().squeeze()
    val_idx = g_coarse.ndata['val_mask'].nonzero().squeeze()
    
    sampler = NeighborSampler([int(f) for f in args.fan_out.split(",")[:args.num_layers]])
    train_loader = DataLoader(g_coarse, train_idx.to(device), sampler, device=device, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(g_coarse, val_idx.to(device), sampler, device=device, batch_size=args.batch_size, shuffle=False)
    
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    best_val_acc = torch.tensor(0.0).to(device)
    
    for epoch in range(args.epoch):
        model.train()
        total_loss = 0
        for it, (input_nodes, output_nodes, blocks) in enumerate(train_loader):
            x = blocks[0].srcdata["feat"]
            y = blocks[-1].dstdata["label"]
            y_hat = model(blocks, x)
            
            if is_multilabel:
                loss = F.binary_cross_entropy_with_logits(y_hat, y.float())
            else:
                loss = F.cross_entropy(y_hat, y.squeeze().long())
                
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()
        
        # Validation on Coarsened Graph
        model.eval(); ys, y_hats = [], []
        with torch.no_grad():
            for _, (_, _, blocks) in enumerate(val_loader):
                ys.append(blocks[-1].dstdata["label"])
                y_hats.append(model(blocks, blocks[0].srcdata["feat"]))
        
        val_preds, val_labels = torch.cat(y_hats), torch.cat(ys)
        acc = MF.accuracy(val_preds, val_labels.squeeze(), task="multiclass", num_classes=num_classes)
        
        if acc > best_val_acc: best_val_acc = acc
        print(f"Epoch {epoch:03d} | Loss {total_loss/(it+1):.4f} | Val Acc: {acc.item():.4f}")
        
    return best_val_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GAT training on coarsened graph, testing on original")
    parser.add_argument("--mode", default="puregpu", choices=["cpu", "mixed", "puregpu"])
    parser.add_argument("--dataset", default="reddit")
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--fan_out", type=str, default="10,10,10")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument('--path', type=str, default='/data/dgl_lab')
    args = parser.parse_args()
    args.dataset = args.dataset.strip().strip(',')

    if not torch.cuda.is_available():
        args.mode = "cpu"

    device = torch.device("cuda" if args.mode != "cpu" else "cpu")

    # --- Smart Path Detection ---
    server_path = "/data/dgl_lab"
    local_path = os.path.expanduser("~/.dgl")
    data_path = args.path if args.path else (server_path if os.path.exists(server_path) else local_path)
    print(f"--> [LOG] Data Root: {data_path}")

    # --- 1. Load Original Graph ---
    print(f"Loading original {args.dataset} dataset...")
    if args.dataset.startswith("ogbn-"):
        dataset = AsNodePredDataset(DglNodePropPredDataset(args.dataset, root=data_path))
        g, num_classes = dataset[0], dataset.num_classes
    elif args.dataset == "reddit":
        dataset = RedditDataset(raw_dir=data_path); g, num_classes = dataset[0], dataset.num_classes
    elif args.dataset == "flickr":
        dataset = FlickrDataset(raw_dir=data_path); g, num_classes = dataset[0], dataset.num_classes
    elif args.dataset == "yelp":
        dataset = LegacyFraudYelpDataset(raw_dir=data_path); g_raw = dataset[0]
        g = dgl.to_homogeneous(g_raw)
        g.ndata['feat'], g.ndata['label'] = g_raw.ndata['feature'], g_raw.ndata['label'].long()
        g.ndata['train_mask'], g.ndata['val_mask'], g.ndata['test_mask'] = g_raw.ndata['train_mask'], g_raw.ndata['val_mask'], g_raw.ndata['test_mask']
        num_classes = 2
    elif args.dataset == "cora":
        dataset = CoraGraphDataset(raw_dir=data_path); g, num_classes = dataset[0], dataset.num_classes
    elif args.dataset.startswith("igb"):
        dgl_path = os.path.join(data_path, args.dataset.replace('-', '_') + '.dgl')
        if not os.path.exists(dgl_path):
            raise FileNotFoundError(f"IGB DGL file not found at {dgl_path}")
        g = dgl.load_graphs(dgl_path)[0][0]; num_classes = 19
        if 'feat' not in g.ndata and 'features' in g.ndata: g.ndata['feat'] = g.ndata['features']
        if 'label' not in g.ndata and 'labels' in g.ndata: g.ndata['label'] = g.ndata['labels']
    else:
        dataset = PubmedGraphDataset(raw_dir=data_path); g, num_classes = dataset[0], dataset.num_classes

    is_multilabel = g.ndata['label'].ndim > 1 and g.ndata['label'].shape[1] > 1
    
    # --- 2. Load Coarsened Graph ---
    coarsened_graph_name = args.dataset.split("-")[-1] + ".dgl"
    if not os.path.exists(coarsened_graph_name):
        print(f"Error: Coarsened graph file '{coarsened_graph_name}' not found. Run 'rupam_file.py' first.")
        sys.exit(1)
    g_coarse = dgl.load_graphs(coarsened_graph_name)[0][0]
    g_coarse.ndata['label'] = g_coarse.ndata['label'].long()
    g.ndata['label'] = g.ndata['label'].long()
    
    if args.mode == "puregpu":
        g, g_coarse = g.to(device), g_coarse.to(device)
    else:
        g_coarse = g_coarse.to(device)
        g.create_formats_()
    
    # Note: hid_size=32 with 8 heads gives 256 features, similar to SAGE/GCN versions
    model = GAT(g_coarse.ndata["feat"].shape[1], 32, num_classes, args.num_layers, num_heads=8).to(device)

    # --- 3. Training ---
    print(f"\nTraining GAT on Coarsened Graph ({args.dataset})...")
    train_start = time.time()
    best_val_acc = train(args, device, g_coarse, model, num_classes, is_multilabel)
    train_time = time.time() - train_start

    # --- 4. Testing ---
    print("\nTesting on Original Graph via direct layer-wise inference...")
    test_start = time.time()
    model.eval()
    with torch.no_grad():
        full_preds = model.inference(g, device, 4096)
        test_idx = g.ndata["test_mask"].nonzero().squeeze()
        test_preds = full_preds[test_idx.cpu()].to(device)
        test_labels = g.ndata["label"][test_idx].to(device)
        test_acc = MF.accuracy(test_preds, test_labels.squeeze(), task="multiclass", num_classes=num_classes)
    test_time = time.time() - test_start

    print("\n--- GAT RESULTS SUMMARY ---")
    print(f"BEST_VAL_ACC: {best_val_acc.item():.4f}")
    print(f"FINAL_TEST_ACC: {test_acc.item():.4f}")
    print(f"TRAIN_TIME: {train_time:.4f}s")
    print(f"TEST_TIME: {test_time:.4f}s")
