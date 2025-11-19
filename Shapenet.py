import torch
import argparse
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.datasets import ShapeNet
from torchmetrics.functional import jaccard_index
from torch_geometric.utils import scatter
import torch_geometric.transforms as T
from models.dgcnn import *
import os


def parse_args():
    parser = argparse.ArgumentParser()

    # Dataset / loader parameters
    parser.add_argument("--path", type=str,
                        default=None,
                        help="Path to ShapeNet dataset root directory.")

    parser.add_argument("--category", type=str, default="Airplane",
                        help="ShapeNet category (e.g., Airplane, Chair). Use 'None' for all categories.")

    parser.add_argument("--points_per_shape", type=int, default=2048,
                        help="Number of points sampled from each shape.")

    parser.add_argument("--batch_size", type=int, default=12,
                        help="Batch size for dataloaders.")

    # Model hyperparameters
    parser.add_argument("--ks", nargs="+", type=int, default=[10],
                        help="Number of nearest neighbors for DGCNN layers.")

    # Training parameters
    parser.add_argument("--n_runs", type=int, default=3,
                        help="How many repeated runs to execute.")

    parser.add_argument("--levels", nargs="+", type=int, default=[3],
                        help="Multilevel hierarchy levels.")

    parser.add_argument("--methods", nargs="+", default=["random"],
                        choices=["random", "subgraph", "RnS"],
                        help="Coarsening method for point selection.")

    return parser.parse_args()


class LocalShapeNet(ShapeNet):
    def download(self):
        # Override download method to prevent downloading
        if not os.path.exists(self.raw_dir):
            raise RuntimeError(f"Dataset not found in {self.raw_dir}. Ensure the dataset is placed correctly.")


@torch.no_grad()
def test_model(loader):
    model.eval()

    ious, categories = [], []
    y_map = torch.empty(loader.dataset.num_classes, device=device).long()
    for data in loader:
        data = data.to(device)
        outs = model(data.pos, data.batch, data)

        sizes = (data.ptr[1:] - data.ptr[:-1]).tolist()
        for out, y, category in zip(outs.split(sizes), data.y.split(sizes),
                                    data.category.tolist()):
            category = list(ShapeNet.seg_classes.keys())[category]
            part = ShapeNet.seg_classes[category]
            part = torch.tensor(part, device=device)

            y_map[part] = torch.arange(part.size(0), device=device)

            iou = jaccard_index(
                out[:, part].argmax(dim=-1),
                y_map[y],
                num_classes=part.size(0),
                task='multiclass'
            )
            ious.append(iou)

        categories.append(data.category)

    iou = torch.tensor(ious, device=device)
    category = torch.cat(categories, dim=0)

    mean_iou = scatter(iou, category, reduce='mean')  # Per-category IoU.
    return float(mean_iou.mean())  # Global IoU.


def random_coarsening(points_per_shape, num_coarse_points, batch):
    selected_indices = torch.randperm(points_per_shape)[:num_coarse_points].to(device)
    selected_indices_value = selected_indices.repeat(repeats=[len(batch.ptr) - 1])
    selected_indices_offset = torch.repeat_interleave(batch.ptr[:-1], repeats=num_coarse_points, dim=0)
    selected_indices = selected_indices_value + selected_indices_offset
    return selected_indices


def subgraph_coarsening(points_per_shape, num_coarse_points, pos, batch):
    selected_shape_indices = []
    for i in range(len(batch.ptr) - 1):
        start, end = batch.ptr[i].item(), batch.ptr[i + 1].item()
        random_idx = torch.randint(start, end, (1,), device=device)
        distances = torch.cdist(pos[random_idx], pos[start:end]).squeeze()
        _, indices = torch.topk(distances, num_coarse_points, largest=False)
        offset = points_per_shape * i
        offset_indices = [j + offset for j in indices.tolist()]
        selected_shape_indices = selected_shape_indices + offset_indices
    selected_indices = torch.tensor(selected_shape_indices)
    return selected_indices


def training_single_level(model, train_loader, test_loader, optimizer, level=0,
                          num_epochs=20, method="random", points_per_shape=2048):
    total_iters = 0
    take_random = 0
    max_test_iou = 0
    model.train()

    for epoch in range(num_epochs):
        for batch in train_loader:

            with torch.no_grad():
                y_oh = batch.y.to(device)
                pos = batch.pos.to(device)
                b = batch.batch.to(device)
                batch = batch.to(device)

            optimizer.zero_grad()

            if level != 0:
                points_per_shape = points_per_shape
                num_coarse_points = int(points_per_shape // 2 ** level)

                if method == "random":
                    selected_indices = random_coarsening(points_per_shape, num_coarse_points, batch)
                elif method == "subgraph":
                    selected_indices = subgraph_coarsening(points_per_shape, num_coarse_points, pos, batch)
                elif method == "RnS":
                    if take_random % 2 == 0:
                        # Use random
                        selected_indices = random_coarsening(points_per_shape, num_coarse_points, batch)
                    else:
                        # Use subgraph
                        selected_indices = subgraph_coarsening(points_per_shape, num_coarse_points, pos, batch)
                    take_random += 1
                else:
                    raise ValueError("unsupported pooling method")

                # Sort indices
                selected_indices = torch.sort(selected_indices).values
                # Filter the data based on selected indices
                posC = pos[selected_indices]
                y_ohC = y_oh[selected_indices]
                bC = b[selected_indices]

            else:
                posC = pos
                bC = b
                y_ohC = y_oh

            outC = model(posC, bC, batch)
            lossC = F.nll_loss(outC, y_ohC)
            lossC.backward()
            optimizer.step()
            total_iters += 1

        # Test model
        iou = test_model(test_loader)
        max_test_iou = max(max_test_iou, iou)
    return model, max_test_iou


def get_selected_indices(method, points_per_shape, num_coarse_points, cur_batch):
    if method == "random":
        selected_indices = random_coarsening(points_per_shape, num_coarse_points, cur_batch)
    elif method == "subgraph":
        selected_indices = subgraph_coarsening(points_per_shape, num_coarse_points, cur_batch.pos, cur_batch)
    else:
        raise ValueError("unsupported pooling method")
    return selected_indices


def multiscale_gradient_training(model, train_loader, test_loader, optimizer,
                                  num_epochs=20,
                                 num_levels=3, fine_batch_samples=4, method="random",
                                 points_per_shape=2048, batch_size=12):

    total_iters = 0
    take_random = 0
    max_test_iou = 0
    model.train()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        for batch in train_loader:
            if batch.batch.max().item() + 1 != batch_size:
                continue  # Skip this batch

            # Define sub-batch sizes
            sub_batch_sizes = [fine_batch_samples]
            for i in range(1, num_levels):
                sub_batch_sizes.append(2 * sub_batch_sizes[i - 1])

            split_batches = split_batch(batch, batch.batch, sub_batch_sizes)
            lossC = torch.tensor(0.0, device=device, requires_grad=True)

            optimizer.zero_grad()
            points_per_shape = points_per_shape

            for i in range(num_levels - 1):
                cur_batch = split_batches[i].to(device)

                # Fine level
                if i == 0:
                    posF = cur_batch.pos
                    bF = cur_batch.batch
                    y_ohF = cur_batch.y
                else:
                    num_coarse_points = int(points_per_shape // 2 ** i)
                    if method == "RnS":
                        if take_random % 2 == 0:
                            # Use random
                            selected_indices = get_selected_indices("random", points_per_shape, num_coarse_points, cur_batch)
                        else:
                            # Use subgraph
                            selected_indices = get_selected_indices("subgraph", points_per_shape, num_coarse_points, cur_batch)
                        take_random += 1
                    else:
                        selected_indices = get_selected_indices(method, points_per_shape, num_coarse_points, cur_batch)

                    # Sort indices:
                    selected_indices = torch.sort(selected_indices).values

                    posF = cur_batch.pos[selected_indices]
                    bF = cur_batch.batch[selected_indices]
                    y_ohF = cur_batch.y[selected_indices]

                # Coarse level
                num_coarse_points = int(points_per_shape // 2 ** (i+1))
                if method == "RnS":
                    if take_random % 2 == 0:
                        # Use random
                        selected_indices = get_selected_indices("random", points_per_shape, num_coarse_points,
                                                                cur_batch)
                    else:
                        # Use subgraph
                        selected_indices = get_selected_indices("subgraph", points_per_shape, num_coarse_points,
                                                                cur_batch)
                    take_random += 1
                else:
                    selected_indices = get_selected_indices(method, points_per_shape, num_coarse_points, cur_batch)

                # Sort indices:
                selected_indices = torch.sort(selected_indices).values

                posC = cur_batch.pos[selected_indices]
                bC = cur_batch.batch[selected_indices]
                y_ohC = cur_batch.y[selected_indices]

                outF = model(posF, bF, cur_batch)
                outC = model(posC, bC, cur_batch)
                tmp_lossF = F.nll_loss(outF, y_ohF)
                tmp_lossC = F.nll_loss(outC, y_ohC)

                lossC = lossC + torch.abs(tmp_lossF - tmp_lossC)

            # Final level
            cur_batch = split_batches[-1].to(device)

            num_coarse_points = int(points_per_shape // 2 ** (num_levels + 1))
            if method == "RnS":
                if take_random % 2 == 0:
                    # Use random
                    selected_indices = get_selected_indices("random", points_per_shape, num_coarse_points, cur_batch)
                else:
                    # Use subgraph
                    selected_indices = get_selected_indices("subgraph", points_per_shape, num_coarse_points, cur_batch)
                take_random += 1
            else:
                selected_indices = get_selected_indices(method, points_per_shape, num_coarse_points, cur_batch)

            # Sort indices:
            selected_indices = torch.sort(selected_indices).values

            posC = cur_batch.pos[selected_indices]
            bC = cur_batch.batch[selected_indices]
            y_ohC = cur_batch.y[selected_indices]

            outC = model(posC, bC, cur_batch)
            tmp_lossC = F.nll_loss(outC, y_ohC)
            lossC = lossC + torch.abs(tmp_lossC)
            lossC.backward()
            optimizer.step()
            total_iters += 1

        # Test model
        iou = test_model(test_loader)
        max_test_iou = max(max_test_iou, iou)

    return model, max_test_iou


def split_batch(data, batch, split_sizes):
    assert sum(split_sizes) == batch.max().item() + 1, "Split sizes must match the batch size."

    # Split indices based on batch tensor
    split_batches = []
    start_idx = 0
    for size in split_sizes:
        # Identify nodes belonging to the current split
        mask = (batch >= start_idx) & (batch < start_idx + size)
        split_data = data.clone()

        # Filter data fields
        split_data.pos = data.pos[mask]
        split_data.x = data.x[mask] if data.x is not None else None
        split_data.y = data.y[mask]
        split_data.batch = batch[mask] - start_idx
        split_data.edge_index = None

        # Update ptr for the split
        num_nodes_per_graph = torch.bincount(split_data.batch)
        split_data.ptr = torch.cat([torch.tensor([0]), torch.cumsum(num_nodes_per_graph, dim=0)])
        graph_indices = torch.arange(start_idx, start_idx + size)
        split_data.category = data.category[graph_indices]
        split_batches.append(split_data)
        start_idx += size

    return split_batches


def initiate_dataset(path, category, batch_size, points_per_shape):
    train_transform = T.Compose([
        T.RandomJitter(0.01),
        T.RandomRotate(15, axis=0),
        T.RandomRotate(15, axis=1),
        T.RandomRotate(15, axis=2),
        T.FixedPoints(points_per_shape, allow_duplicates=True, replace=False)
    ])
    train_pre_transform = T.NormalizeScale()

    test_pre_transform, test_transform = (
        T.NormalizeScale(),
        T.FixedPoints(points_per_shape, allow_duplicates=True, replace=False),
    )

    train_dataset = LocalShapeNet(path, category, split='trainval', transform=train_transform, pre_transform=train_pre_transform)
    test_dataset = LocalShapeNet(path, category, split='test', transform=test_transform, pre_transform=test_pre_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_dataset, test_dataset, train_loader, test_loader


def initiate_model(train_dataset, k):
    model = DynamicEdgeConvNet(train_dataset.num_classes, k=k).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    return model, optimizer


def train_network(n_levels, n_fine_epochs, model, train_loader, test_loader,
                  optimizer, method):
    for i in range(n_levels - 1, -1, -1):
        print("level:", i)
        model, max_test_iou = training_single_level(model, train_loader,
                                                    test_loader,
                                                    optimizer, level=i,
                                                    num_epochs=n_fine_epochs*(2 ** i),
                                                    method=method)

    return max_test_iou


if __name__ == '__main__':
    args = parse_args()
    model_name = "DGCNN"
    print("Model:", model_name)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_workers = 0 if device.type == 'cpu' else 6

    train_dataset, test_dataset, train_loader, test_loader = \
        initiate_dataset(path=args.path, category=args.category,
                         batch_size=args.batch_size, points_per_shape=args.points_per_shape)

    for k in args.ks:
        for n_levels in args.levels:
            for method in args.methods:
                if n_levels == 1:
                    n_fine_epochs = 100
                else:
                    # 3 levels
                    n_fine_epochs = 40
                multiscale_fine_epochs = 50
                multiscale_levels = 1

                print(f"number of fine epochs:", n_fine_epochs, flush=True)
                print("category:", args.category, flush=True)
                print(f"Using method {method} with {n_levels} levels and k {k}:", flush=True)

                test_ious = []
                for run in range(args.n_runs):
                    model, optimizer = initiate_model(train_dataset, k)


                    # Multilevel training
                    run_max_test_iou = train_network(n_levels, n_fine_epochs, model, train_loader, test_loader,
                                                      optimizer, method)

                    # Multiscale gradients training
                    model, run_max_test_iou_mc = multiscale_gradient_training(model, train_loader, test_loader,
                                                                              optimizer,
                                                                              num_epochs=multiscale_fine_epochs,
                                                                              num_levels=2, fine_batch_samples=4,
                                                                              method=method,
                                                                              points_per_shape=args.points_per_shape,
                                                                              batch_size=args.batch_size)
                    # Complete training with fine grid
                    run_max_test_iou_ms = train_network(multiscale_levels, multiscale_fine_epochs, model,
                                                        train_loader, test_loader,
                                                          optimizer, method)
                    run_max_test_iou = max(run_max_test_iou_mc, run_max_test_iou_ms)

                    test_ious.append(run_max_test_iou)

                max_test_iou = max(test_ious)
                std_test_iou = torch.tensor(test_ious).std().item()

                # Print results
                print("\n--- Final Results Across Runs ---", flush=True)
                print(f"Max Test IoU: {100 * max_test_iou:.2f}% (std: {100 * std_test_iou:.2f}%)", flush=True)