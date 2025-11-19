from utils import graph_coarsening
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score
import torch_geometric.utils as pyg_utils
import networkx as nx
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data, Dataset, DataLoader

class level_mats:
    def __init__(self, edge_index, num_level=3, method="random", k=1):
        m = torch.max(edge_index) + 1
        m = int(m // ((4/3) ** num_level))
        if num_level == 0:
            columns, edge_index_c, Pmat = graph_coarsening(edge_index, m=m, k=1, method=method, level=num_level)
        else:
            columns, edge_index_c, Pmat = graph_coarsening(edge_index, m=m, k=k, method=method, level=num_level)

        self.P = Pmat
        self.cols = columns
        self.edges_index_c = edge_index_c


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, preds = [], []
    for data in loader:
        data.to(device)
        num_nodes = data.x.size(0)
        ei = _edge_index_from_data(data)
        data.edge_index = ei
        values = torch.ones(data.edge_index.shape[1], device=device)
        adj = torch.sparse_coo_tensor(data.edge_index, values, (num_nodes, num_nodes)).to(device)
        data.adj = adj
        adj = data.adj
        ys.append(data.y)
        out = model(data.x.to(device), adj.to(device))
        preds.append((out > 0).float().cpu())

    y, pred = torch.cat(ys, dim=0).cpu().numpy(), torch.cat(preds, dim=0).cpu().numpy()
    return f1_score(y, pred, average='micro') if pred.sum() > 0 else 0


@torch.no_grad()
def evaluate_molhiv(model, loader, device):
    model.eval()

    ys, preds = [], []
    for data in loader:
        data = data.to(device)

        num_nodes = data.x.size(0)
        values = torch.ones(data.edge_index.shape[1], device=device)
        adj = torch.sparse_coo_tensor(data.edge_index, values,
                                      (num_nodes, num_nodes)).to(device)

        out = model(data.x.float().to(device), adj)   # [N, C]
        out = out.mean(dim=0, keepdim=True)           # [1, C]

        prob = torch.softmax(out, dim=1)[:, 1].cpu()
        ys.append(data.y.view(-1).cpu())
        preds.append(prob)

    y = torch.cat(ys, dim=0).numpy()
    p = torch.cat(preds, dim=0).numpy()
    return roc_auc_score(y, p)


@torch.no_grad()
def evaluate_nci1(model, loader, device):
    model.eval()

    correct = 0
    total = 0
    for data in loader:
        data = data.to(device)

        num_nodes = data.x.size(0)
        values = torch.ones(data.edge_index.shape[1], device=device)
        adj = torch.sparse_coo_tensor(data.edge_index, values,
                                      (num_nodes, num_nodes)).to(device)

        out = model(data.x.float().to(device), adj)  # [N, C]
        out = out.mean(dim=0, keepdim=True)          # [1, C]
        pred = out.argmax(dim=1)                     # predicted class index

        correct += (pred == data.y.view(-1)).sum().item()
        total += data.y.size(0)

    return correct / total if total > 0 else 0.0


def graph_classification_training_single_level(model, train_loader, test_loader, optimizer, level=0,
                                               num_epochs=20, method="random", k=1, device='cuda', data_name='MolHIV'):
    total_iters = 0
    max_test_score = 0
    total_loss = total_examples = 0
    model.train()
    take_random = 0

    for epoch in range(num_epochs):
        for batch in train_loader:

            if method == "RnS":
                if take_random % 2 == 0:
                    Mc = level_mats(batch.edge_index, num_level=level, method="random", k=k)
                else:
                    Mc = level_mats(batch.edge_index, num_level=level, method="subgraph", k=k)
                take_random += 1
            else:
                Mc = level_mats(batch.edge_index, num_level=level, method=method, k=k)

            P = Mc.P.to(device)
            if level == 0:
                edge_index_coarse = batch.edge_index.to(device)
            else:
                edge_index_coarse = Mc.edges_index_c.to(device)

            y = batch.y.to(device)
            x = batch.x.to(device).float()

            batch.to(device)
            optimizer.zero_grad()

            if P.shape[0] != x.shape[0]:
                P_dense = P.to_dense()
                if P_dense.shape[0] < x.shape[0]:
                    pad_rows = x.shape[0] - P_dense.shape[0]
                    P_dense = torch.cat([P_dense, torch.zeros(pad_rows, P_dense.shape[1], device=P_dense.device)],
                                        dim=0)
                elif P_dense.shape[0] > x.shape[0]:
                    P_dense = P_dense[:x.shape[0], :]
                P = P_dense.to_sparse()

            xC = torch.matmul(P.T, x)
            if xC.size(0) < 2:
                xC = x

            if level == 0:
                xC = x
                num_nodes = batch.x.size(0)
                values = torch.ones(batch.edge_index.shape[1], device=device)
                adj = torch.sparse_coo_tensor(batch.edge_index, values, (num_nodes, num_nodes)).to(device)
            else:
                adj = pyg_utils.to_torch_csr_tensor(edge_index_coarse, size=(xC.shape[0], xC.shape[0]))

            out = model(xC, adj)  # [num_nodes, C]
            out = out.mean(dim=0, keepdim=True)  # [1, C]
            y = y.view(-1).long()  # [1]
            loss = F.nll_loss(out, y)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_iters += 1

            total_loss += loss.item() * batch.num_nodes
            total_examples += batch.num_nodes

        # Test model
        if data_name == "MolHIV":
            test_score = evaluate_molhiv(model, test_loader, device)
        else:
            test_score = evaluate_nci1(model, test_loader, device)
        max_test_score = max(max_test_score, test_score)

    return model, max_test_score


# Function to comlete the multiscale gradient training process
def fine_level_training(model, train_loader, test_loader, optimizer, num_epochs=20, device='cuda'):
    total_iters = 0
    max_test_f1 = 0
    total_loss = total_examples = 0
    model.train()

    for epoch in range(num_epochs):
        for batch in train_loader:
            y = batch.y.to(device)
            x = batch.x.to(device)
            batch.to(device)

            ei = _edge_index_from_data(batch)
            batch.edge_index = ei

            optimizer.zero_grad()
            num_nodes = batch.x.size(0)
            values = torch.ones(batch.edge_index.shape[1], device=device)
            adj = torch.sparse_coo_tensor(batch.edge_index, values, (num_nodes, num_nodes)).to(device)
            out = model(x, adj)

            criterion = torch.nn.BCEWithLogitsLoss()
            loss = criterion(out, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_iters += 1
            total_loss += loss.item() * batch.num_nodes
            total_examples += batch.num_nodes

        # Test model
        test_f1 = evaluate(model, test_loader, device)
        max_test_f1 = max(max_test_f1, test_f1)

    return max_test_f1


def _edge_index_from_data(data):
    if hasattr(data, "edge_index") and data.edge_index is not None:
        return data.edge_index
    if hasattr(data, "adj_t"):
        if data.adj_t.layout == torch.sparse_coo:
            return data.adj_t.coalesce().indices()
        elif data.adj_t.layout == torch.sparse_csr:
            crow_indices = data.adj_t.crow_indices()
            col_indices = data.adj_t.col_indices()
            row_indices = torch.arange(len(crow_indices) - 1, device=crow_indices.device)
            row_indices = row_indices.repeat_interleave(crow_indices[1:] - crow_indices[:-1])
            return torch.stack([row_indices, col_indices], dim=0)
        else:
            raise TypeError(f"Unsupported sparse layout: {data.adj_t.layout}")
    raise AttributeError("Data object has neither edge_index nor adj_t")


def multiscale_gradient_training(model, train_loader, test_loader, optimizer,
                                 num_epochs=20, method="random", device='cuda'):
    total_iters = 0  # Initialize the iteration counter
    take_random = 0
    max_test_f1 = 0
    total_loss = total_examples = 0
    k = 1
    model.train()

    for epoch in range(num_epochs):
        data_iter = iter(train_loader)  # Convert DataLoader to an iterator

        while True:
            try:
                batch1 = next(data_iter)  # First batch
                batch2 = next(data_iter)  # Second batch

                batch1, batch2 = batch1.to(device), batch2.to(device)
                ei1 = _edge_index_from_data(batch1)
                ei2 = _edge_index_from_data(batch2)
                G1 = to_networkx(Data(edge_index=ei1), to_undirected=True)
                G2 = to_networkx(Data(edge_index=ei2), to_undirected=True)
                num_components1 = nx.number_connected_components(G1)
                num_components2 = nx.number_connected_components(G2)
                if num_components1 > 1 or num_components2 > 1:
                    continue
                batch1.edge_index = ei1
                batch2.edge_index = ei2
            except StopIteration:
                break

            lossC = torch.tensor(0.0, device=device, requires_grad=True)
            optimizer.zero_grad()

            # Fine level (batch1)
            xF, yF = batch1.x.to(device), batch1.y.to(device)
            adjF = pyg_utils.to_torch_csr_tensor(batch1.edge_index, size=(xF.shape[0], xF.shape[0]))

            if method == "RnS":
                if take_random % 2 == 0:
                    Mc = level_mats(batch1.edge_index, num_level=1, method="random", k=k)
                else:
                    Mc = level_mats(batch1.edge_index, num_level=1, method="subgraph", k=k)
                take_random += 1
            else:
                Mc = level_mats(batch1.edge_index, num_level=1, method=method, k=k)

            P = Mc.P.to(device)
            xC = P.T @ xF
            yC = (P.T @ yF.float()).float()
            adjC = pyg_utils.to_torch_csr_tensor(Mc.edges_index_c, size=(xC.shape[0], xC.shape[0]))

            outF = model(xF, adjF)
            outC = model(xC, adjC)
            criterion = torch.nn.BCEWithLogitsLoss()
            lossC = lossC + torch.abs(criterion(outF, yF) - criterion(outC, yC))

            # Fine level (batch2)
            xF, yF = batch2.x.to(device), batch2.y.to(device)
            adjF = pyg_utils.to_torch_csr_tensor(batch2.edge_index, size=(xF.shape[0], xF.shape[0]))

            if method == "RnS":
                if take_random % 2 == 0:
                    Mc = level_mats(batch2.edge_index, num_level=1, method="random", k=k)
                else:
                    Mc = level_mats(batch2.edge_index, num_level=1, method="subgraph", k=k)
                take_random += 1
            else:
                Mc = level_mats(batch2.edge_index, num_level=1, method=method, k=k)

            P = Mc.P.to(device)
            xC = P.T @ xF
            yC = (P.T @ yF.float()).float()
            adjC = pyg_utils.to_torch_csr_tensor(Mc.edges_index_c, size=(xC.shape[0], xC.shape[0]))

            outC = model(xC, adjC)
            lossC = lossC + torch.abs(criterion(outC, yC))

            lossC.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_iters += 1
            total_loss += lossC.item() * batch1.num_nodes
            total_examples += batch1.num_nodes

        # Test model
        test_f1 = evaluate(model, test_loader, device)
        max_test_f1 = max(max_test_f1, test_f1)

    return model, max_test_f1


def train_network(n_levels, n_fine_epochs, model, train_loader, test_loader,
                  optimizer, method, k, device, dt):
    for i in range(n_levels - 1, -1, -1):
        model, max_test_score = graph_classification_training_single_level(model, train_loader, test_loader,
                                                                            optimizer, level=i,
                                                                            num_epochs=n_fine_epochs * (2 ** i),
                                                                            method=method,
                                                                            k=k,
                                                                            device=device,
                                                                            data_name=dt)

    return max_test_score