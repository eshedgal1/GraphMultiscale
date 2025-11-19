import torch
from utils import level_mats
import torch.nn.functional as F
from models.gcn import GCN
from models.gat import GAT
from models.gin import GIN


def init_model(model_name, dataset, data, device):
    if model_name == "GCN":
        model = GCN(
            in_channels=data.x.size(1),
            hidden_channels=192,
            out_channels=dataset.num_classes,
            num_layers=4,
            dropout=0.3
        ).to(device)
    elif model_name == "GAT":
        model = GAT(
            in_channels=data.x.size(1),
            hidden_channels=64,
            out_channels=dataset.num_classes,
            heads=2,
            num_layers=3,
            dropout=0.3
        ).to(device)
    elif model_name == "GIN":
        model = GIN(
            in_channels=data.x.size(1),
            hidden_channels=256,
            out_channels=dataset.num_classes,
            num_layers=3,
            dropout=0.3
        ).to(device)
    else:
        raise ValueError("Unsupported architecture")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    return model, optimizer


@torch.no_grad()
def evaluate(model, data):
    model.eval()
    pred = model(data.x, data.adj).argmax(dim=-1)

    if data.y.dim() > 1:
        labels = data.y.argmax(dim=-1)
    else:
        labels = data.y

    accs = []
    for mask_name in ["train_mask", "val_mask", "test_mask"]:
        mask = getattr(data, mask_name, None)
        if mask is None or mask.sum() == 0:
            accs.append(0.0)
        else:
            accs.append(int((pred[mask] == labels[mask]).sum()) / int(mask.sum()))

    return accs


def training_single_level(model, data, optimizer, level=0,
                          num_epochs=20, method="random", k=1, device='cuda', dataset=None):
    total_iters = 0
    take_random = 0  # for RnS
    max_train, max_val, max_test = 0, 0, 0
    model.train()

    for epoch in range(num_epochs):
        if method == "RnS":
            if take_random % 2 == 0:
                Mc = level_mats(data.edge_index, num_level=level, method="random", k=k)
            else:
                Mc = level_mats(data.edge_index, num_level=level, method="subgraph", k=k)
            take_random += 1
        else:
            Mc = level_mats(data.edge_index, num_level=level, method=method, k=k)

        P = Mc.P.to(device)

        if level == 0:
            edge_index_coarse = data.edge_index.to(device)  # use the fine graph's edge_index directly
        else:
            edge_index_coarse = Mc.edges_index_c.to(device)

        with torch.no_grad():
            y = data.y.to(device)
            x = data.x.to(device)

        optimizer.zero_grad()
        xC = P.T @ x
        yC = (P.T @ y.float()).long()

        if level == 0:
            xC = x
            yC = y
            adjC = data.adj
            train_mask_coarse = data.train_mask.to(device)
        else:
            num_nodes = xC.shape[0]
            values = torch.ones(edge_index_coarse.shape[1], device=device)
            adjC = torch.sparse_coo_tensor(edge_index_coarse, values, (num_nodes, num_nodes)).to(device)
            adjC = adjC + adjC.t()
            train_mask_coarse = P.T @ data.train_mask.float().to(device)
            train_mask_coarse = train_mask_coarse > 0  # convert to boolean

        outC = model(xC, adjC)
        if dataset == "facebook" or dataset == "ppi":
            lossC = torch.nn.CrossEntropyLoss()(outC[train_mask_coarse], yC[train_mask_coarse].argmax(dim=-1))
        else:
            lossC = F.nll_loss(outC[train_mask_coarse], yC[train_mask_coarse])


        lossC.backward()
        optimizer.step()
        total_iters += 1

        # Test model
        train_acc, val_acc, test_acc = evaluate(model, data)
        max_train = max(max_train, train_acc)
        max_val = max(max_val, val_acc)
        max_test = max(max_test, test_acc)

    return model, max_train, max_val, max_test


def train_network(n_levels, n_fine_epochs, model, data, optimizer, method, k, device, dt):
    for i in range(n_levels - 1, -1, -1):
        model, max_train, max_val, max_test = training_single_level(model, data,
                                                                     optimizer, level=i,
                                                                     num_epochs=n_fine_epochs * (2 ** i),
                                                                     method=method,
                                                                     k=k,
                                                                     device=device,
                                                                     dataset=dt)

    return max_train, max_val, max_test