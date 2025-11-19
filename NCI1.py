import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from models.gcn import GCN
from torch_geometric.datasets import TUDataset
from inductive_training import *


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--methods", nargs="+", default=["random"],
        choices=["random", "topk", "subgraph", "RnS"],
        help="Coarsening or pooling method to use in hierarchical training."
    )

    parser.add_argument(
        "--levels", nargs="+", type=int, default=[3],
        choices=[1, 2],
        help="Number of multilevel hierarchy levels (e.g., 1, 2, 3)."
    )

    parser.add_argument(
        "--p_vals", nargs="+", type=int, default=[1],
        help="Connectivity enhancement factors (use larger integers to strengthen graph connections)."
    )

    parser.add_argument(
        "--n_runs", type=int, default=3,
        help="Number of repeated runs for each setting."
    )

    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_workers = 0 if device.type == 'cpu' else 6

    dataset = TUDataset(root="data/inductive", name="NCI1")
    train_dataset, test_dataset = dataset[:3000], dataset[3000:]
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    dt = "NCI1"

    model = GCN(in_channels=train_dataset.x.size(1),
                hidden_channels=192,
                out_channels=train_dataset.num_classes,
                num_layers=4,
                dropout=0.5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    for p in args.p_vals:
        for n_levels in args.levels:
            for method in args.methods:

                if n_levels == 1:
                    n_fine_epochs = 100
                else:
                    # 2 levels
                    n_fine_epochs = 40

                print(f"number of fine epochs:", n_fine_epochs, flush=True)
                print(f"Using method {method} with {n_levels} levels and p {p}:", flush=True)

                # Get results across multiple runs
                test_accs = []
                for run in range(args.n_runs):
                    model.reset_parameters()
                    run_max_test_acc = train_network(n_levels, n_fine_epochs, model, train_loader, test_loader,
                                                     optimizer, method, p, device, dt)
                    test_accs.append(run_max_test_acc)

                # Compute overall statistics
                max_test_acc = max(test_accs)
                std_test_acc = torch.tensor(test_accs).std().item()

                # Print results
                print("\n--- Final Results Across Runs ---", flush=True)
                print(f"Max Test Accuracy: {100 * max_test_acc:.2f}% (std: {100 * std_test_acc:.2f}%)", flush=True)


if __name__ == "__main__":
    main()
