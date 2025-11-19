import torch
import argparse
from transductive_training import train_network, init_model
from torch_geometric.datasets import Planetoid, WikiCS, Flickr, AttributedGraphDataset, CitationFull

# This file may be used for the training of the following datasets:
# Cora, CiteCeer, PubMed, WikiCS, Flickr, DBLP, PPI (trunsductive), BlogCatalog, Facebook


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--nets", nargs="+", default=["GCN", "GIN", "GAT"],
        choices=["GCN", "GIN", "GAT"],
        help="GNN architectures to train (choose one or more)."
    )

    parser.add_argument(
        "--datas", nargs="+", default=["pub"],
        choices=["cora", "cite", "pub", "wiki", "flickr", "dblp", "facebook", "blog", "ppi"],
        help="Datasets to run experiments on (choose one or more)."
    )

    parser.add_argument(
        "--methods", nargs="+", default=["random"],
        choices=["random", "topk", "subgraph", "RnS"],
        help="Coarsening or pooling method to use in hierarchical training."
    )

    parser.add_argument(
        "--levels", nargs="+", type=int, default=[2],
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

    for net in args.nets:
        for dt in args.datas:
            if dt == "cite":
                dataset = Planetoid(root='/tmp/CiteSeer', name='CiteSeer')
            elif dt == "pub":
                dataset = Planetoid(root='/tmp/PubMed', name='PubMed')
            elif dt == "wiki":
                dataset = WikiCS(root="data/WikiCS")
            elif dt == "flickr":
                dataset = Flickr(root="data/Flickr")
            elif dt == "dblp":
                dataset = CitationFull(root="data/CitationFull", name="DBLP")
            elif dt == "facebook":
                dataset = AttributedGraphDataset(root='data/Facebook', name='facebook')
            elif dt == "blog":
                dataset = AttributedGraphDataset(root='data/BlogCatalog', name='blogcatalog')
            elif dt == "ppi":
                dataset = AttributedGraphDataset(root='data/PPI', name='ppi')
            elif dt == "cora":
                dataset = Planetoid(root='/tmp/Cora', name='Cora')
            else:
                raise RuntimeError("unsupported dataset")

            data = dataset[0].to(device)

            if dt == "facebook" or dt == "blog" or dt == "ppi" or dt == "dblp":
                num_nodes = data.num_nodes
                perm = torch.randperm(num_nodes)
                train_size = int(0.8 * num_nodes)
                data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
                data.train_mask[perm[:train_size]] = True
                data.test_mask = ~data.train_mask
                data.val_mask = None
            elif dt == "wiki":
                data.train_mask = data.train_mask[:, 0]
                data.val_mask = data.val_mask[:, 0]

            model, optimizer = init_model(net, dataset, data, device)

            num_nodes = data.x.size(0)
            values = torch.ones(data.edge_index.shape[1], device=device)
            adj = torch.sparse_coo_tensor(data.edge_index, values, (num_nodes, num_nodes)).to(device)
            adj = adj + adj.t()
            data.adj = adj

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
                        print(f"data: {dt}, model: {net}", flush=True)
                        print(f"Using method {method} with {n_levels} levels and connectity {p}:", flush=True)

                        # Get results across multiple runs
                        train_accs, test_accs = [], []
                        for run in range(args.n_runs):
                            model.reset_parameters()
                            run_max_train, _, run_max_test = \
                                train_network(n_levels, n_fine_epochs, model, data, optimizer, method, p, device, dt)
                            train_accs.append(run_max_train)
                            test_accs.append(run_max_test)

                        # Compute overall statistics
                        max_train = max(train_accs)
                        max_test = max(test_accs)
                        std_train = torch.tensor(train_accs).std().item()
                        std_test = torch.tensor(test_accs).std().item()

                        # Print results
                        print("\n--- Final Results Across Runs ---", flush=True)
                        print(f"Max Train Accuracy: {100 * max_train:.2f}% ({100 * std_train:.1f}%)", flush=True)
                        print(f"Max Test Accuracy: {100 * max_test:.2f}% ({100 * std_test:.1f}%)", flush=True)


if __name__ == "__main__":
    main()
