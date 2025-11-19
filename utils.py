import torch
from torch_geometric.utils import k_hop_subgraph


class level_mats:
    def __init__(self, edge_index, num_level=3, method="random", k=1):
        m = torch.max(edge_index) + 1
        m = m // (2 ** num_level)
        if num_level == 0:
            columns, edge_index_c, Pmat = graph_coarsening(edge_index, m=m, k=1, method=method, level=num_level)
        else:
            columns, edge_index_c, Pmat = graph_coarsening(edge_index, m=m, k=k, method=method, level=num_level)

        self.P = Pmat
        self.cols = columns
        self.edges_index_c = edge_index_c


def graph_coarsening(edge_index, m, k, method, level=2):
    device = edge_index.device
    num_nodes = torch.max(edge_index) + 1
    values = torch.ones(edge_index.shape[1], device=device)
    adj = torch.sparse_coo_tensor(edge_index, values, (num_nodes, num_nodes)).to(device)

    Ak = adj.clone()
    for _ in range(k - 1):
        Ak = Ak @ adj

    if method == "random":
        random_columns = torch.randperm(num_nodes, device=device)[:m].sort()[0]
        indices = torch.stack([random_columns, torch.arange(m, device=device)])
    elif method == "topk":
        scores = torch.sparse.sum(adj, dim=1).to_dense()
        topk_indices = torch.topk(scores, m).indices
        random_columns = topk_indices
        indices = torch.stack([random_columns, torch.arange(m, device=device)])
    elif method == "subgraph":
        if level == 1:
            n_hops = 6
        elif level == 2:
            n_hops = 4
        else:
            n_hops = 2

        retries = 0
        max_retries = 50
        random_columns = torch.tensor([], device=device)
        # make sure to avoid choosing single isolated node
        while len(random_columns) <= 1 and retries < max_retries:
            start_node = torch.randint(0, num_nodes, (1,), device=device).item()
            random_columns, edge_index_khop, _, _ = k_hop_subgraph(
                node_idx=start_node,
                num_hops=n_hops,
                edge_index=edge_index,
                num_nodes=num_nodes
            )
            retries += 1
        indices = torch.stack([random_columns, torch.arange(len(random_columns), device=device)])
    else:
        raise ValueError("Unsupported pooling method")

    if method == "subgraph":
        values = torch.ones(indices.size(1), device=device)
        P = torch.sparse_coo_tensor(indices, values, size=(num_nodes, len(random_columns))).to(device)
    else:
        values = torch.ones(m, device=device)
        P = torch.sparse_coo_tensor(indices, values, size=(num_nodes, m)).to(device)

    A = P.T @ (Ak @ P)
    edge_index_c = A._indices()
    return random_columns, edge_index_c, P
