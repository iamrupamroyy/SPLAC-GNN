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

class GCN(nn.Module):
    def __init__(self, in_size, hid_size, out_size, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.res_linears = nn.ModuleList()
        
        if num_layers == 1:
            self.layers.append(dglnn.GraphConv(in_size, out_size, allow_zero_in_degree=True))
        else:
            # Layer 1
            self.layers.append(dglnn.GraphConv(in_size, hid_size, activation=F.relu, allow_zero_in_degree=True))
            self.norms.append(nn.LayerNorm(hid_size))
            self.res_linears.append(nn.Linear(in_size, hid_size))
            
            # Middle Layers
            for _ in range(num_layers - 2):
                self.layers.append(dglnn.GraphConv(hid_size, hid_size, activation=F.relu, allow_zero_in_degree=True))
                self.norms.append(nn.LayerNorm(hid_size))
                self.res_linears.append(nn.Linear(hid_size, hid_size))
            
            # Output Layer
            self.layers.append(dglnn.GraphConv(hid_size, out_size, allow_zero_in_degree=True))
            self.res_linears.append(nn.Linear(hid_size, out_size))
            
        self.dropout = nn.Dropout(0.5)
        self.hid_size, self.out_size = hid_size, out_size

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h_res = h
            h = layer(block, h)
            
            # Apply Residual Connection
            res = self.res_linears[l](h_res[:block.num_dst_nodes()])
            h = h + res
            
            if l != len(self.layers) - 1:
                h = self.norms[l](h)
                h = F.relu(h)
                h = self.dropout(h)
        return h

    def inference(self, g, device, batch_size):
        """Layer-wise inference to save memory. Intermediate features stay on CPU."""
        feat = g.ndata['feat'].cpu()
        sampler = MultiLayerFullNeighborSampler(1)
        dataloader = DataLoader(g, torch.arange(g.num_nodes()).to(g.device), sampler, device=device, 
                                batch_size=batch_size, shuffle=False, drop_last=False)
        
        for l, layer in enumerate(self.layers):
            out_dim = self.hid_size if l != len(self.layers) - 1 else self.out_size
            y = torch.empty(g.num_nodes(), out_dim, device="cpu")
            
            # Linear and Norm must be on device
            res_lin = self.res_linears[l].to(device)
            if l != len(self.layers) - 1:
                norm = self.norms[l].to(device)
            
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader, desc=f"Inference Layer {l+1}", disable=not sys.stdout.isatty()):
                x = feat[input_nodes.cpu()].to(device)
                h_res = x[:blocks[0].num_dst_nodes()]
                h = layer(blocks[0], x)
                
                # Apply residual
                h = h + res_lin(h_res)
                
                if l != len(self.layers) - 1:
                    h = norm(h)
                    h = F.relu(h)
                    h = self.dropout(h)
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
    
    ckpt_path = f"best_model_gcn_{args.dataset}_{run_id}.pt"
    
    print(f"Training on Coarsened Graph for {args.epoch} epochs...")
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
        acc = MF.accuracy(val_preds, val_labels.squeeze(), task="multiclass", num_classes=num_classes).item()
        
        if acc > best_val_acc:
            best_val_acc = acc
            torch.save(model.state_dict(), ckpt_path)
        print(f"Epoch {epoch:03d} | Loss {total_loss/(it+1):.4f} | Val Acc: {acc:.4f}")

    # Fine-tuning on Original Graph
    if fine_tune_g is not None and args.ft_epoch > 0:
        print(f"\nFine-tuning on Original Graph for {args.ft_epoch} epochs...")
        model.load_state_dict(torch.load(ckpt_path))
        train_idx_orig = fine_tune_g.ndata['train_mask'].nonzero().squeeze()
        val_idx_orig = fine_tune_g.ndata['val_mask'].nonzero().squeeze()
        
        train_loader_orig = DataLoader(fine_tune_g, train_idx_orig.to(device), sampler, device=device, batch_size=args.batch_size, shuffle=True)
        val_loader_orig = DataLoader(fine_tune_g, val_idx_orig.to(device), sampler, device=device, batch_size=args.batch_size, shuffle=False)
        
        # Use a smaller learning rate for fine-tuning
        opt_ft = torch.optim.Adam(model.parameters(), lr=args.lr * 0.1, weight_decay=5e-4)
        
        for epoch in range(args.ft_epoch):
            model.train()
            total_loss = 0
            for it, (input_nodes, output_nodes, blocks) in enumerate(train_loader_orig):
                x = blocks[0].srcdata["feat"]
                y = blocks[-1].dstdata["label"]
                y_hat = model(blocks, x)
                if is_multilabel:
                    loss = F.binary_cross_entropy_with_logits(y_hat, y.float())
                else:
                    loss = F.cross_entropy(y_hat, y.squeeze().long())
                opt_ft.zero_grad(); loss.backward(); opt_ft.step()
                total_loss += loss.item()
            
            model.eval(); ys, y_hats = [], []
            with torch.no_grad():
                for _, (_, _, blocks) in enumerate(val_loader_orig):
                    ys.append(blocks[-1].dstdata["label"])
                    y_hats.append(model(blocks, blocks[0].srcdata["feat"]))
            val_preds, val_labels = torch.cat(y_hats), torch.cat(ys)
            acc = MF.accuracy(val_preds, val_labels.squeeze(), task="multiclass", num_classes=num_classes).item()
            
            if acc > best_val_acc:
                best_val_acc = acc
                torch.save(model.state_dict(), ckpt_path)
            print(f"FT Epoch {epoch:03d} | Loss {total_loss/(it+1):.4f} | Val Acc (Orig): {acc:.4f}")
            
    return best_val_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="puregpu", choices=["cpu", "mixed", "puregpu"])
    parser.add_argument("--dataset", default="reddit")
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--ft_epoch", type=int, default=0, help="Number of fine-tuning epochs on original graph")
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--hid_size", type=int, default=256)
    parser.add_argument("--fan_out", type=str, default="10,10,10")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument('--path', type=str, default='/data/dgl_lab')
    
    # Coarsening parameters to identify the correct graph file
    parser.add_argument("-r", "--retain_fraction", type=float, default=0.75)
    parser.add_argument("--boost_h", type=float, default=10000.0)
    parser.add_argument("--use_feature_sim", action="store_true", default=False)
    
    args = parser.parse_args()
    args.dataset = args.dataset.strip().strip(',')

    device = torch.device("cuda" if args.mode != "cpu" else "cpu")
    run_id = f"{int(time.time())}_{os.getpid()}"

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
        name_for_folders = args.dataset.replace("ogbn-", "")
    elif args.dataset == "reddit":
        dataset = RedditDataset(raw_dir=data_path); g, num_classes = dataset[0], dataset.num_classes
        name_for_folders = "reddit"
    elif args.dataset == "flickr":
        dataset = FlickrDataset(raw_dir=data_path); g, num_classes = dataset[0], dataset.num_classes
        name_for_folders = "flickr"
    elif args.dataset == "yelp":
        # Use Legacy Fraud Yelp (Single-label binary)
        dataset = LegacyFraudYelpDataset(raw_dir=data_path); g_raw = dataset[0]
        g = dgl.to_homogeneous(g_raw)
        g.ndata['feat'], g.ndata['label'] = g_raw.ndata['feature'], g_raw.ndata['label'].long()
        g.ndata['train_mask'], g.ndata['val_mask'], g.ndata['test_mask'] = g_raw.ndata['train_mask'], g_raw.ndata['val_mask'], g_raw.ndata['test_mask']
        num_classes = 2
        name_for_folders = "yelp"
    elif args.dataset == "cora":
        dataset = CoraGraphDataset(raw_dir=data_path); g, num_classes = dataset[0], dataset.num_classes
        name_for_folders = "cora"
    elif args.dataset.startswith("igb"):
        # Search for .dgl file
        dgl_path = os.path.join(data_path, args.dataset.replace('-', '_') + '.dgl')
        if not os.path.exists(dgl_path):
            raise FileNotFoundError(f"IGB DGL file not found at {dgl_path}")
        g = dgl.load_graphs(dgl_path)[0][0]; num_classes = 19
        # Mapping for IGB consistency
        if 'feat' not in g.ndata and 'features' in g.ndata: g.ndata['feat'] = g.ndata['features']
        if 'label' not in g.ndata and 'labels' in g.ndata: g.ndata['label'] = g.ndata['labels']
        name_for_folders = args.dataset.replace("igb-", "")
    else:
        dataset = PubmedGraphDataset(raw_dir=data_path); g, num_classes = dataset[0], dataset.num_classes
        name_for_folders = "pubmed"

    # Automatically detect multi-label based on label dimensions
    is_multilabel = g.ndata['label'].ndim > 1 and g.ndata['label'].shape[1] > 1
    print(f"--> [LOG] Multi-label mode: {is_multilabel}")
    
    print("--> [LOG] Adding self-loops to original graph...")
    g = dgl.add_self_loop(g)

    # --- 2. Load Coarsened Graph ---
    # Construct unique filename based on coarsening parameters
    feat_sim_str = "_sim" if args.use_feature_sim else ""
    coarsened_graph_name = f"{name_for_folders}_r{args.retain_fraction}_h{args.boost_h}{feat_sim_str}.dgl"
    
    if not os.path.exists(coarsened_graph_name):
        print(f"Error: Coarsened graph file '{coarsened_graph_name}' not found. Run 'rupam_file.py' first with matching parameters.")
        sys.exit(1)
    g_coarse = dgl.load_graphs(coarsened_graph_name)[0][0]
    
    print("--> [LOG] Adding self-loops to coarsened graph...")
    g_coarse = dgl.add_self_loop(g_coarse)
    
    # Ensure labels are long for multi-class cross entropy
    g_coarse.ndata['label'] = g_coarse.ndata['label'].long()
    g.ndata['label'] = g.ndata['label'].long()
    
    # Mode-based placement
    if args.mode == "puregpu":
        g, g_coarse = g.to(device), g_coarse.to(device)
    else:
        g_coarse = g_coarse.to(device)
        # For mixed mode, original graph stays on CPU but we optimize its format
        g.create_formats_()
    
    # --- 3. Training ---
    print(f"\nTraining on Coarsened Graph ({args.dataset})...")
    
    # Initialize model
    model = GCN(g_coarse.ndata["feat"].shape[1], args.hid_size, num_classes, args.num_layers).to(device)
    
    train_start = time.time()
    
    # Use the new train function which supports fine-tuning
    best_val_acc = train(args, device, g_coarse, model, num_classes, is_multilabel, run_id, fine_tune_g=g)
    
    train_time = time.time() - train_start

    # --- 4. Testing (Load Best Model) ---
    print("\nTesting on Original Graph...")
    test_start = time.time()
    ckpt_path = f"best_model_gcn_{args.dataset}_{run_id}.pt"
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path))
    model.eval()
    with torch.no_grad():
        # Full inference on ORIGINAL graph g
        full_preds = model.inference(g, device, 4096)
        test_idx = g.ndata["test_mask"].nonzero().squeeze()
        test_preds = full_preds[test_idx.cpu()].to(device)
        test_labels = g.ndata["label"][test_idx].to(device)
        test_acc = MF.accuracy(test_preds, test_labels.squeeze(), task="multiclass", num_classes=num_classes).item()
    test_time = time.time() - test_start

    # --- 5. Report ---
    import json
    # Reset peak memory stats for inference reporting
    gpu_peak = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
    
    # Calculate simulated memory access (Edges * Layers * Features)
    # Original Graph
    orig_nodes = g.num_nodes()
    orig_edges = g.num_edges()
    # Coarsened Graph
    coarse_nodes = g_coarse.num_nodes()
    coarse_edges = g_coarse.num_edges()
    
    # Total memory accesses during training (approximate)
    total_mem_access_train = coarse_edges * args.num_layers * args.epoch * 2
    
    result = {
        "dataset": args.dataset,
        "method": "ESSC_GCN",
        "run_id": run_id,
        "ratio": float(coarse_nodes / orig_nodes),
        "orig_nodes": int(orig_nodes),
        "orig_edges": int(orig_edges),
        "red_nodes": int(coarse_nodes),
        "red_edges": int(coarse_edges),
        "train_time": float(train_time),
        "test_time": float(test_time),
        "test_acc": float(test_acc),
        "best_val_acc": float(best_val_acc.item() if torch.is_tensor(best_val_acc) else best_val_acc),
        "gpu_peak_mem_mb": float(gpu_peak),
        "total_mem_access_train": int(total_mem_access_train)
    }

    print("\n--- RESULTS SUMMARY ---")
    print(f"BEST_VAL_ACC: {result['best_val_acc']:.4f}")
    print(f"FINAL_TEST_ACC: {result['test_acc']:.4f}")
    print(f"TRAIN_TIME: {result['train_time']:.4f}s")
    print(f"TEST_TIME: {result['test_time']:.4f}s")
    print(f"\nPRINT_RESULT: {json.dumps(result, indent=4)}")
    
    # Cleanup checkpoint to save space
    if 'ckpt_path' in locals() and os.path.exists(ckpt_path):
        os.remove(ckpt_path)

print("******************************* Rupam Roy ******************************")
print("**************** Indian Institute of Technology Bhilai ********************")
