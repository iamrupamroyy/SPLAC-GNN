import os
# Monkeypatch OGB to bypass download confirmation
import ogb.utils.url
ogb.utils.url.decide_download = lambda url: True

# Disable all tqdm progress bars
os.environ["TQDM_DISABLE"] = "1"

# Set DGL backend and home directory before importing DGL to avoid permission errors on clusters
os.environ['DGLBACKEND'] = 'pytorch'
# Redirect DGL config directory to a writable location if the home directory is restricted
try:
    if not os.access(os.path.expanduser("~"), os.W_OK):
        os.environ['DGL_HOME'] = os.path.join(os.getcwd(), ".dgl")
except Exception:
    os.environ['DGL_HOME'] = os.path.join(os.getcwd(), ".dgl")
import argparse
import time
import dgl
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
import tqdm
import sys
# Suppress download progress
import os
os.environ['DGL_DOWNLOAD_PROGRESS'] = '0'
from dgl.data import AsNodePredDataset
from dgl.dataloading import (
    DataLoader,
    MultiLayerFullNeighborSampler,
    NeighborSampler,
)
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.data import RedditDataset, YelpDataset, PubmedGraphDataset, FlickrDataset, CoraGraphDataset
import dgl.graphbolt as gb
from dgl.data import FraudYelpDataset as LegacyFraudYelpDataset # Rename to avoid conflict
import warnings
warnings.filterwarnings("ignore")
os.environ['DGL_LOG_LEVEL'] = 'warning'

class GCN(nn.Module):
    def __init__(self, in_size, hid_size, out_size, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        if num_layers == 1:
            self.layers.append(dglnn.GraphConv(in_size, out_size, allow_zero_in_degree=True))
        else:
            # input layer
            self.layers.append(dglnn.GraphConv(in_size, hid_size, allow_zero_in_degree=True))
            self.norms.append(nn.BatchNorm1d(hid_size))
            # hidden layers
            for _ in range(num_layers - 2):
                self.layers.append(dglnn.GraphConv(hid_size, hid_size, allow_zero_in_degree=True))
                self.norms.append(nn.BatchNorm1d(hid_size))
            # output layer
            self.layers.append(dglnn.GraphConv(hid_size, out_size, allow_zero_in_degree=True))
        self.dropout = nn.Dropout(0.5)
        self.hid_size = hid_size
        self.out_size = out_size

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = self.norms[l](h)
                h = F.relu(h)
                h = self.dropout(h)
        return h

    def inference(self, g, device, batch_size):
        feat = g.ndata["feat"]
        sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=["feat"])
        dataloader = DataLoader(
            g,
            torch.arange(g.num_nodes()).to(g.device),
            sampler,
            device=device,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )
        buffer_device = torch.device("cpu")
        pin_memory = buffer_device != device

        for l, layer in enumerate(self.layers):
            y = torch.empty(
                g.num_nodes(),
                self.hid_size if l != len(self.layers) - 1 else self.out_size,
                dtype=feat.dtype,
                device=buffer_device,
                pin_memory=pin_memory,
            )
            feat = feat.to(device)
            if l != len(self.layers) - 1:
                norm = self.norms[l].to(device)
            
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader, disable=not sys.stdout.isatty()):
                x = feat[input_nodes]
                h = layer(blocks[0], x)
                if l != len(self.layers) - 1:
                    h = norm(h)
                    h = F.relu(h)
                    h = self.dropout(h)
                y[output_nodes[0] : output_nodes[-1] + 1] = h.to(buffer_device)
            feat = y
        return y


def evaluate(model, graph, dataloader, num_classes, is_multilabel):
    model.eval()
    ys = []
    y_hats = []
    with torch.no_grad():
        for it, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
            x = blocks[0].srcdata["feat"]
            ys.append(blocks[-1].dstdata["label"])
            y_hats.append(model(blocks, x))
    
    y_hats_cat = torch.cat(y_hats)
    ys_cat = torch.cat(ys)

    if is_multilabel:
        preds = torch.sigmoid(y_hats_cat)
        return MF.f1_score(
            preds,
            ys_cat.int(),
            task="multilabel",
            num_labels=num_classes,
            average="macro",
        )
    else:
        return MF.accuracy(
            y_hats_cat,
            ys_cat,
            task="multiclass",
            num_classes=num_classes,
        )


def layerwise_infer(device, graph, nid, model, num_classes, batch_size, is_multilabel):
    model.eval()
    with torch.no_grad():
        pred = model.inference(graph, device, batch_size)
        pred = pred[nid]
        label = graph.ndata["label"][nid].to(pred.device)
        
        if is_multilabel:
            preds = torch.sigmoid(pred)
            return MF.f1_score(
                preds,
                label.int(),
                task="multilabel",
                num_labels=num_classes,
                average="macro",
            )
        else:
            return MF.accuracy(pred, label, task="multiclass", num_classes=num_classes)


def train(args, device, g, dataset, model, num_classes, is_multilabel):
    train_idx = dataset.train_idx.to(device)
    val_idx = dataset.val_idx.to(device)
    # Dynamically adjust fan_out based on num_layers
    fan_out_list = [int(fanout) for fanout in args.fan_out.split(",")]
    if len(fan_out_list) < args.num_layers:
        # Repeat the last fan_out value if not enough values are provided
        fan_out_list = fan_out_list + [fan_out_list[-1]] * (args.num_layers - len(fan_out_list))
    fan_out = fan_out_list[:args.num_layers]
    print(f"--> [LOG] Using fan_out: {fan_out} for {args.num_layers} layers")

    sampler = NeighborSampler(
        fan_out,
        prefetch_node_feats=["feat"],
        prefetch_labels=["label"],
    )
    use_uva = args.mode == "mixed"
    train_dataloader = DataLoader(
        g,
        train_idx,
        sampler,
        device=device,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_uva=use_uva,
    )
    val_dataloader = DataLoader(
        g,
        val_idx,
        sampler,
        device=device,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_uva=use_uva,
    )

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    best_val_metric = 0

    for epoch in range(args.epoch):
        model.train()
        total_loss = 0
        epoch_start_time = time.time()
        for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
            x = blocks[0].srcdata["feat"]
            y = blocks[-1].dstdata["label"]
            y_hat = model(blocks, x)
            if is_multilabel:
                loss = F.binary_cross_entropy_with_logits(y_hat, y.float())
            else:
                loss = F.cross_entropy(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        
        epoch_time = time.time() - epoch_start_time
        metric = evaluate(model, g, val_dataloader, num_classes, is_multilabel)
        if metric > best_val_metric:
            best_val_metric = metric

        metric_name = "Val F1" if is_multilabel else "Val Acc"
        print(
            "Epoch {:05d} | Loss {:.4f} | {} {:.4f} (Best {:.4f}) | Time {:.4f}".format(
                epoch, total_loss / (it + 1), metric_name, metric.item(), best_val_metric.item(), epoch_time
            )
        )
    return best_val_metric, model


if __name__ == "__main__":
    print("******************************* Rupam Roy ******************************")
    print("**************** Indian Institute of Technology Bhilai ********************")
    
    parser = argparse.ArgumentParser(description="GNN training on original graph")
    parser.add_argument("--mode", default="puregpu", choices=["cpu", "mixed", "puregpu"])
    parser.add_argument("--dataset", default="ogbn-arxiv", help="Dataset name")
    parser.add_argument("--epoch", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of layers")
    parser.add_argument("--fan_out", type=str, default="10,10,10", help="Neighbor sampling fan-out")
    parser.add_argument("--batch_size", type=int, default=1024, help="Training batch size")
    parser.add_argument('--path', type=str, default=None, help='Path to dataset directory.')
    parser.add_argument("--dt", type=str, default="float", help="Data type (float, bfloat16)")
    args = parser.parse_args()
    args.dataset = args.dataset.strip().strip(',')

    if not torch.cuda.is_available():
        args.mode = "cpu"
    print(f"Training in {args.mode} mode on original graph.")
    print(f"Using dataset: {args.dataset}")

    print("--- Loading Original Graph Data ---")
    load_start_time = time.time()

    # --- Smart Path Detection ---
    server_path = "/data/dgl_lab"
    local_path = os.path.expanduser("~/.dgl")
    data_path = args.path if args.path else (server_path if os.path.exists(server_path) else local_path)
    print(f"--> [LOG] Data Root: {data_path}")

    if args.dataset.startswith('ogbn-'):
        dataset = AsNodePredDataset(DglNodePropPredDataset(name=args.dataset, root=data_path))
        g = dataset[0]
        num_classes = dataset.num_classes
    elif args.dataset == 'yelp':
        # Yelp Legacy (Fraud Detection)
        print("--> [LOG] Loading Yelp Legacy (Fraud Detection) dataset...")
        d = LegacyFraudYelpDataset(raw_dir=data_path); g_raw = d[0]
        g = dgl.to_homogeneous(g_raw)
        g.ndata['feat'] = g_raw.ndata['feature']
        g.ndata['label'] = g_raw.ndata['label'].long()
        num_classes = 2 
        g.ndata['train_mask'] = g_raw.ndata['train_mask']
        g.ndata['val_mask'] = g_raw.ndata['val_mask']
        g.ndata['test_mask'] = g_raw.ndata['test_mask']
        dataset = d
    elif args.dataset.startswith('igb'):
        # For IGB, if path is not provided, check the common server path first
        igb_path = args.path if args.path else '/data/dgl_lab'
        dgl_file_path = os.path.join(igb_path, args.dataset.replace('-', '_') + '.dgl')
        
        if os.path.exists(dgl_file_path):
            print(f"--> [LOG] Loading pre-processed IGB graph from: {dgl_file_path}")
            graphs, _ = dgl.load_graphs(dgl_file_path)
            g = graphs[0]
        else:
            print(f"--> [LOG] Pre-processed IGB file not found at {dgl_file_path}. Trying 'igb' package...")
            try:
                from igb.dataset import IGBDataset
                dataset_size = args.dataset.split('-')[-1] if '-' in args.dataset else 'small'
                d = IGBDataset(name='igb', root=igb_path, dataset_size=dataset_size, synthetic=False)
                g = d[0]
                # Map keys to match script
                if 'feat' not in g.ndata and 'features' in g.ndata: g.ndata['feat'] = g.ndata['features']
                if 'label' not in g.ndata and 'labels' in g.ndata: g.ndata['label'] = g.ndata['labels']
            except ImportError:
                raise FileNotFoundError(f"IGB file not found at {dgl_file_path} and 'igb' package not installed. Please provide --path to the folder containing the .dgl file.")
        
        num_classes = 19 # IGB has 19 classes
        class DummyDataset: pass
        dataset = DummyDataset()
    else:
        if args.dataset == 'reddit': d = RedditDataset(raw_dir=data_path)
        elif args.dataset == 'pubmed': d = PubmedGraphDataset(raw_dir=data_path)
        elif args.dataset == 'cora': d = CoraGraphDataset(raw_dir=data_path)
        elif args.dataset in ['flicker', 'flickr']: d = FlickrDataset(raw_dir=data_path)
        else: raise ValueError(f"Unsupported dataset: {args.dataset}")
        g = d[0]
        num_classes = d.num_classes
        dataset = d # Assign to dataset for consistent access to indices
        
    # Determine if the dataset is multi-label
    is_multilabel = 'label' in g.ndata and g.ndata['label'].ndim > 1 and g.ndata['label'].shape[1] > 1
    if is_multilabel:
        print("Detected multi-label dataset. Using F1 score and BCE loss.")

    # Add self-loops to the graph (CRITICAL FOR GCN/GAT)
    print("--> [LOG] Adding self-loops to graph...")
    g = dgl.add_self_loop(g)

    # Attach final indices to the dataset object for the trainer to use
    if not hasattr(dataset, 'train_idx'):
        class DummyDataset:
            def __init__(self, train_idx, val_idx, test_idx, num_classes):
                self.train_idx = train_idx
                self.val_idx = val_idx
                self.test_idx = test_idx
                self.num_classes = num_classes
        
        train_mask_tensor = g.ndata.get('train_mask', torch.zeros(g.num_nodes(), dtype=torch.bool))
        val_mask_tensor = g.ndata.get('val_mask', torch.zeros(g.num_nodes(), dtype=torch.bool))
        test_mask_tensor = g.ndata.get('test_mask', torch.zeros(g.num_nodes(), dtype=torch.bool))
        train_idx_new = torch.nonzero(train_mask_tensor, as_tuple=True)[0]
        val_idx_new = torch.nonzero(val_mask_tensor, as_tuple=True)[0]
        test_idx_new = torch.nonzero(test_mask_tensor, as_tuple=True)[0]

        if val_idx_new.numel() == 0 and train_idx_new.numel() > 0:
            print("Validation set not found or is empty. Creating a new one from the training set (80/20 split).")
            perm = torch.randperm(train_idx_new.shape[0])
            split_point = int(train_idx_new.shape[0] * 0.8)
            new_train_indices = train_idx_new[perm[:split_point]]
            val_idx_new = train_idx_new[perm[split_point:]]
            train_idx_new = new_train_indices
        
        dataset = DummyDataset(train_idx_new, val_idx_new, test_idx_new, num_classes)
    
    load_time = time.time() - load_start_time
    print(f"Original Graph: {g}")
    print(f"Graph loading time: {load_time:.4f}s")

    if args.epoch == 0:
        print("Epoch is 0. Data loading complete. Exiting successfully.")
        sys.exit(0)

    device = torch.device("cpu" if args.mode == "cpu" else "cuda")
    if args.mode == "puregpu":
        g = g.to(device)

    in_size = g.ndata["feat"].shape[1]
    out_size = num_classes
    model = GCN(in_size, 256, out_size, args.num_layers).to(device)

    if args.dt == "bfloat16":
        g = dgl.to_bfloat16(g)
        model = model.to(dtype=torch.bfloat16)

    print(f"\n--- Training on Original Graph (on {args.mode}) (BASELINE GCN) ---")
    total_train_start_time = time.time()
    best_val_metric, model = train(args, device, g, dataset, model, num_classes, is_multilabel)
    total_train_time = time.time() - total_train_start_time
    
    print(f"\n--- Testing on Original Graph (on {args.mode}) ---")
    test_start_time = time.time()
    test_metric = layerwise_infer(device, g, dataset.test_idx, model, num_classes, batch_size=4096, is_multilabel=is_multilabel)
    test_time = time.time() - test_start_time

    val_metric_name = "Best Validation F1" if is_multilabel else "Best Validation Accuracy"
    test_metric_name = "Final Test F1" if is_multilabel else "Final Test Accuracy"
    print("\n------ FINAL BASELINE METRICS (GCN) ------")
    print(f"Dataset: {args.dataset}")
    print(f"{val_metric_name}: {best_val_metric.item():.4f}")
    print(f"{test_metric_name}: {test_metric.item():.4f}")
    print(f"Total Training Time: {total_train_time:.4f}s")
    print(f"Inference Time: {test_time:.4f}s")
    print("------------------------------------")
