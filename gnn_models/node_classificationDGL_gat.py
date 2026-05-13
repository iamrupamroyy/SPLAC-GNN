import os

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
from dgl.data import AsNodePredDataset
from dgl.dataloading import (
    DataLoader,
    MultiLayerFullNeighborSampler,
    NeighborSampler,
)
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.data import RedditDataset, YelpDataset, PubmedGraphDataset, FlickrDataset, CoraGraphDataset
import warnings
warnings.filterwarnings("ignore")
os.environ['DGL_LOG_LEVEL'] = 'warning'
from dgl.data import FraudYelpDataset as LegacyFraudYelpDataset

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
            if l != len(self.layers) - 1:
                out_dim = self.hid_size * self.num_heads
            else:
                out_dim = self.out_size
                
            y = torch.empty(
                g.num_nodes(),
                out_dim,
                dtype=feat.dtype,
                device=buffer_device,
                pin_memory=pin_memory,
            )
            feat = feat.to(device)
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader, desc=f"Inference Layer {l+1}"):
                x = feat[input_nodes]
                h = layer(blocks[0], x)
                if l != len(self.layers) - 1:
                    h = h.flatten(1)
                    h = F.relu(h)
                    h = self.dropout(h)
                else:
                    h = h.mean(1)
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
    sampler = NeighborSampler(
        [int(fanout) for fanout in args.fan_out.split(",")[:args.num_layers]],
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
    parser = argparse.ArgumentParser(description="GAT training on original graph")
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
    print(f"Training GAT in {args.mode} mode on original graph.")
    print(f"Using dataset: {args.dataset}")

    print("--- Loading Original Graph Data ---")
    load_start_time = time.time()
    data_path = args.path if args.path else os.path.expanduser("~/.dgl")

    if args.dataset.startswith('ogbn-'):
        dataset = AsNodePredDataset(DglNodePropPredDataset(name=args.dataset, root=data_path))
        g = dataset[0]
        num_classes = dataset.num_classes
    elif args.dataset == 'yelp':
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
        igb_path = args.path if args.path else '/data/dgl_lab'
        dgl_file_path = os.path.join(igb_path, args.dataset.replace('-', '_') + '.dgl')
        if os.path.exists(dgl_file_path):
            graphs, _ = dgl.load_graphs(dgl_file_path)
            g = graphs[0]
        else:
            from igb.dataset import IGBDataset
            dataset_size = args.dataset.split('-')[-1] if '-' in args.dataset else 'small'
            d = IGBDataset(name='igb', root=igb_path, dataset_size=dataset_size, synthetic=False)
            g = d[0]
            if 'feat' not in g.ndata and 'features' in g.ndata: g.ndata['feat'] = g.ndata['features']
            if 'label' not in g.ndata and 'labels' in g.ndata: g.ndata['label'] = g.ndata['labels']
        num_classes = 19
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
        dataset = d
        
    is_multilabel = 'label' in g.ndata and g.ndata['label'].ndim > 1 and g.ndata['label'].shape[1] > 1

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
            perm = torch.randperm(train_idx_new.shape[0])
            split_point = int(train_idx_new.shape[0] * 0.8)
            new_train_indices = train_idx_new[perm[:split_point]]
            val_idx_new = train_idx_new[perm[split_point:]]
            train_idx_new = new_train_indices
        dataset = DummyDataset(train_idx_new, val_idx_new, test_idx_new, num_classes)
    
    load_time = time.time() - load_start_time
    device = torch.device("cpu" if args.mode == "cpu" else "cuda")
    if args.mode == "puregpu":
        g = g.to(device)

    in_size = g.ndata["feat"].shape[1]
    # Note: hid_size=32 with 8 heads gives 256 features
    model = GAT(in_size, 32, num_classes, args.num_layers, num_heads=8).to(device)

    if args.dt == "bfloat16":
        g = dgl.to_bfloat16(g)
        model = model.to(dtype=torch.bfloat16)

    print(f"\n--- Training GAT on Original Graph (on {args.mode}) ---")
    total_train_start_time = time.time()
    best_val_metric, model = train(args, device, g, dataset, model, num_classes, is_multilabel)
    total_train_time = time.time() - total_train_start_time
    
    print(f"\n--- Testing GAT on Original Graph (on {args.mode}) ---")
    test_start_time = time.time()
    test_metric = layerwise_infer(device, g, dataset.test_idx, model, num_classes, batch_size=4096, is_multilabel=is_multilabel)
    test_time = time.time() - test_start_time

    val_metric_name = "Best Validation F1" if is_multilabel else "Best Validation Accuracy"
    test_metric_name = "Final Test F1" if is_multilabel else "Final Test Accuracy"
    print("\n------ FINAL GAT METRICS ------")
    print(f"Dataset: {args.dataset}")
    print(f"{val_metric_name}: {best_val_metric.item():.4f}")
    print(f"{test_metric_name}: {test_metric.item():.4f}")
    print(f"Total Training Time: {total_train_time:.4f}s")
    print(f"Inference Time: {test_time:.4f}s")
    print("------------------------------------")
