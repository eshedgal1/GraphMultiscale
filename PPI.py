import torch
from torch_geometric.datasets import PPI
from torch_geometric.data import DataLoader
import torch_geometric.transforms as T
from models.gcnii import GCNII
from inductive_training import *


# Determine device and set num_workers
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_workers = 0 if device.type == 'cpu' else 6
print(f"Using device: {device}")

path = "data/inductive/PPI"

pre_transform = T.Compose([T.GCNNorm(), T.ToSparseTensor()])
train_dataset = PPI(path, split='train', pre_transform=pre_transform)
val_dataset = PPI(path, split='val', pre_transform=pre_transform)
test_dataset = PPI(path, split='test', pre_transform=pre_transform)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
dt = "PPI"
num_levels = 2


n_runs = 3
methods = ["random"]
ks = [1]
num_epochs = 500


for k in ks:
    for method in methods:

        test_f1s = []
        for run in range(n_runs):
            model = GCNII(hidden_channels=2048, num_layers=9, alpha=0.5, theta=1.0,
                          shared_weights=False, dropout=0.2).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
            model.reset_parameters()

            # Multiscale gradients
            model, run_max_test_f1_mc = multiscale_gradient_training(model, train_loader, test_loader, optimizer,
                                                                     num_epochs=num_epochs, method=method, device=device)
            # Complete training with fine grid
            level = 1
            run_max_test_f1_ms = fine_level_training(model, train_loader, test_loader, optimizer,
                                                     num_epochs=num_epochs, device=device)
            run_max_test_f1 = max(run_max_test_f1_mc, run_max_test_f1_ms)

            # Append results
            test_f1s.append(run_max_test_f1)


        max_test_f1 = max(test_f1s)
        std_test_f1 = torch.tensor(test_f1s).std().item()

        # Print results
        print("\n--- Final Results Across Runs ---", flush=True)
        print(f"Max Test F1: {100 * max_test_f1:.2f}% (std: {100 * std_test_f1:.2f}%)", flush=True)
