import torch
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data
from torchvision.transforms import ToTensor


class MNISTGraphDataset(Dataset):
    def __init__(self, root='.', train=True, k=8, threshold=30):
        self.k = k
        self.threshold = threshold
        self.base_dataset = MNIST(root=root, train=train, download=True, transform=ToTensor())

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        return mnist_to_graph_batch(img, label, self.k, self.threshold)


def mnist_to_graph_batch(img_tensor, label, k=8, threshold=30):
    img = img_tensor.squeeze(0)  # (28, 28)
    H, W = img.shape
    N = H * W

    x = img.flatten().unsqueeze(1)  # (784, 1)
    pos = torch.stack(torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij'), dim=-1).reshape(-1, 2)
    knn = NearestNeighbors(n_neighbors=k + 1).fit(pos)
    _, indices = knn.kneighbors(pos)
    edge_index = []

    for i, neighbors in enumerate(indices):
        for j in neighbors[1:]:
            edge_index.append([i, j])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    y = torch.full((N,), 10, dtype=torch.long)
    y[img.flatten() > (threshold / 255.0)] = label

    return Data(x=x, edge_index=edge_index, y=y)
