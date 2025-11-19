import torch
import argparse
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

from models.gcn import GCN
from qtips_mnist_training import train_network
from generate_mnist_data import *

# Run this script to train the synthetic MNIST dataset


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--methods", nargs="+", default=["random"],
        choices=["random", "topk", "subgraph", "RnS"],
        help="Coarsening or pooling method to use in hierarchical training."
    )

    parser.add_argument(
        "--levels", nargs="+", type=int, default=[3],
        choices=[1, 2, 3, 4],
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
    num_workers = 0 if device.type == 'cpu' else 4


    full_train_dataset = MNISTGraphDataset(train=True, k=3)
    train_dataset = torch.utils.data.Subset(full_train_dataset, range(1000))
    full_test_dataset = MNISTGraphDataset(train=False, k=3)
    test_dataset = torch.utils.data.Subset(full_test_dataset, range(100))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    first_graph = train_dataset[0]
    model = GCN(
        in_channels=first_graph.x.size(1),
        hidden_channels=256,
        out_channels=11,
        num_layers=4,
        dropout=0.5
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for p in args.p_vals:
        for n_levels in args.levels:
            for method in args.methods:
                if n_levels == 1:
                    n_fine_epochs = 500
                else:
                    # 3 levels
                    n_fine_epochs = 100

                print(f"number of fine epochs:", n_fine_epochs, flush=True)
                print(f"Using method {method} with {n_levels} levels and connectity {p}:", flush=True)

                test_accs = []
                for run in range(args.n_runs):
                    model.reset_parameters()
                    run_max_test = \
                        train_network(n_levels, n_fine_epochs, model, train_loader, test_loader, optimizer, method, p, device)
                    test_accs.append(run_max_test)

                max_test = max(test_accs)
                std_test = torch.tensor(test_accs, dtype=torch.float32).std().item()

                # Print results
                print("\n--- Final Results Across Runs ---", flush=True)
                print(f"Max Test Accuracy: {100 * max_test:.2f}% (std: {100 * std_test:.2f}%)", flush=True)


if __name__ == "__main__":
    main()
