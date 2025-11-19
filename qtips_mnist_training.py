import torch
import torch.nn.functional as F
from utils import level_mats


@torch.no_grad()
def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    for batch in test_loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index)
        pred = out.argmax(dim=1)
        correct += (pred == batch.y).sum().item()
        total += batch.y.size(0)
    return correct / total


def training_single_level(model, train_loader, test_loader, optimizer, level=0,
                          num_epochs=20,  method="random", k=1, device='cuda'):
    total_iters = 0
    take_random = 0
    max_test = 0
    model.train()

    for epoch in range(num_epochs):
        for batch in train_loader:
            batch = batch.to(device)

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
                edge_index_coarse = batch.edge_index.to(device)  # Use the fine graph's edge_index directly
            else:
                edge_index_coarse = Mc.edges_index_c.to(device)

            with torch.no_grad():
                y = batch.y.to(device)
                x = batch.x.to(device)

            optimizer.zero_grad()
            xC = P.T @ x
            yC = (P.T @ y.float()).long()

            if level == 0:
                xC = x
                yC = y

            outC = model(xC, edge_index_coarse)
            lossC = F.nll_loss(outC, yC)
            lossC.backward()
            optimizer.step()
            total_iters += 1

        # Test model
        test_acc = evaluate(model, test_loader, device)
        max_test = max(max_test, test_acc)

    return model, max_test


def train_network(n_levels, n_fine_epochs, model, train_loader, test_loader, optimizer, method, k, device):
    for i in range(n_levels - 1, -1, -1):
        model, max_test = training_single_level(model, train_loader, test_loader,
                                                                     optimizer, level=i,
                                                                     num_epochs=n_fine_epochs * (2 ** i),
                                                                     method=method,
                                                                     k=k,
                                                                     device=device)

    return max_test