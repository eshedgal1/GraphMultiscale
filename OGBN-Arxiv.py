import torch
import argparse
from torch_geometric.data import Data
from ogb.nodeproppred import NodePropPredDataset, Evaluator
from transductive_training import train_network, init_model

# This file may be used for the training of OGBN-Arxiv dataset


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--nets", nargs="+", default=["GCN"],
        choices=["GCN", "GIN", "GAT"],
        help="GNN architectures to train (choose one or more)."
    )

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


def init_dataset(dataset, device):
    graph, labels = dataset[0]
    data = Data(
        x=torch.tensor(graph['node_feat'], dtype=torch.float),
        edge_index=torch.tensor(graph['edge_index'], dtype=torch.long),
        y=torch.tensor(labels, dtype=torch.long).squeeze(),
    )
    data = data.to(device)

    # Get train/validation/test splits
    split_idx = dataset.get_idx_split()
    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.train_mask[split_idx['train']] = True
    data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.val_mask[split_idx['valid']] = True
    data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.test_mask[split_idx['test']] = True

    num_nodes = data.x.size(0)
    values = torch.ones(data.edge_index.shape[1], device=device)
    adj = torch.sparse_coo_tensor(data.edge_index, values, (num_nodes, num_nodes)).to(device)
    adj = adj + adj.t()
    data.adj = adj

    return data

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_workers = 0 if device.type == 'cpu' else 4

    # Load dataset
    dataset = NodePropPredDataset(name='ogbn-arxiv')
    data = init_dataset(dataset, device)
    evaluator = Evaluator(name='ogbn-arxiv')


    for net in args.nets:
        model, optimizer = init_model(net, dataset, data, device)

        for p in args.p_vals:
            for n_levels in args.levels:
                for method in args.methods:
                    if n_levels == 1:
                        n_fine_epochs = 2000
                    elif n_levels == 2:
                        n_fine_epochs = 1000
                    elif n_levels == 3:
                        n_fine_epochs = 800
                    else:
                        # 4 levels
                        n_fine_epochs = 600

                    print(f"number of fine epochs:", n_fine_epochs, flush=True)
                    print(f"Model: {net}", flush=True)
                    print(f"Using method {method} with {n_levels} levels and connectivity {p}:", flush=True)

                    # Get results across multiple runs
                    train_accs, val_accs, test_accs = [], [], []
                    for run in range(args.n_runs):
                        model.reset_parameters()
                        run_max_train, run_max_val, run_max_test = \
                            train_network(n_levels, n_fine_epochs, model, data, optimizer, method, p, device, "ogbn-arxiv")
                        train_accs.append(run_max_train)
                        val_accs.append(run_max_val)
                        test_accs.append(run_max_test)

                    # Compute overall statistics
                    max_train = max(train_accs)
                    max_val = max(val_accs)
                    max_test = max(test_accs)
                    std_train = torch.tensor(train_accs).std().item()
                    std_val = torch.tensor(val_accs).std().item()
                    std_test = torch.tensor(test_accs).std().item()

                    # Print results
                    print("\n--- Final Results Across Runs ---", flush=True)
                    print(f"Max Train Accuracy: {100 * max_train:.2f}% (std: {100 * std_train:.2f}%)", flush=True)
                    print(f"Max Validation Accuracy: {100 * max_val:.2f}% (std: {100 * std_val:.2f}%)", flush=True)
                    print(f"Max Test Accuracy: {100 * max_test:.2f}% (std: {100 * std_test:.2f}%)", flush=True)


if __name__ == "__main__":
    main()

