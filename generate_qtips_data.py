import matplotlib
import numpy as np
import torch
import scipy.sparse as sp
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from scipy.sparse.linalg import eigsh
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from tqdm import tqdm  # For progress bar
from torch_geometric.nn import MessagePassing
import torch.nn as nn
from torch_geometric.data import Data, Dataset, DataLoader
import matplotlib
from torchvision import transforms
import torchvision.transforms.functional as FF
from sklearn.neighbors import KDTree
from sklearn.neighbors import BallTree
import torch_geometric as tge


def get_random_graph(n=1024, plotGraph=False):
    # Generate random graph points
    x = np.random.rand(n)
    y = np.random.rand(n)

    # Delaunay triangulation to get connectivity
    tri = Delaunay(np.vstack([x, y]).T)
    T = tri.simplices

    # Generate edge list
    edges = []
    for simplex in T:
        edges.append((simplex[0], simplex[1]))
        edges.append((simplex[1], simplex[2]))
    edges = np.array(edges)

    # Sort edges and remove duplicates
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)

    # PyTorch Geometric expects edge index to be a 2xE tensor
    edge_index = torch.tensor(edges.T, dtype=torch.long)

    # Create adjacency matrix A as sparse matrix
    row, col = edges[:, 0], edges[:, 1]
    A = sp.coo_matrix((np.ones(len(row)), (row, col)), shape=(n, n))
    A = A + A.T  # Make symmetric

    # Add self-loops
    I = sp.eye(n)
    A = A + I

    # Graph Laplacian
    D = sp.diags(A.sum(axis=1).A1)  # Degree matrix
    D_inv_sqrt = sp.diags(1.0 / np.sqrt(D.diagonal()))
    L = D_inv_sqrt @ (D - A) @ D_inv_sqrt

    # Get eigenvalues and eigenvectors
    # Use eigsh for sparse matrix L and real eigenvalues
    num_eig = min(10, n)  # Adjust based on need, here taking a few for demonstration
    eigenvalues, V = eigsh(L, k=num_eig, which='SM')

    # Generate input data
    s = np.random.randn(num_eig) / np.arange(1, num_eig + 1)
    s[0] = 0
    c = V @ s

    # Generate the output data
    sout = np.tanh(10 * s)  # np.flip(s)   #np.maximum(s,0) #

    cout = V @ sout

    # Construct PyTorch Geometric Data object
    input_data = c
    output_data = cout
    if plotGraph:
        # Plot the graph with node values
        plt.subplot(2, 2, 1)
        plt.scatter(x, y, s=500, c=c, cmap='viridis')
        for edge in edges:
            plt.plot([x[edge[0]], x[edge[1]]], [y[edge[0]], y[edge[1]]], color='b', linewidth=2)
        plt.axis('off')
        plt.title("Input data")

        # Plot the input data s
        plt.subplot(2, 2, 3)
        plt.plot(s, linewidth=3)
        plt.title("Input spectrum")

        # Plot the transformed output data
        plt.subplot(2, 2, 2)
        plt.scatter(x, y, s=500, c=cout, cmap='viridis')
        for edge in edges:
            plt.plot([x[edge[0]], x[edge[1]]], [y[edge[0]], y[edge[1]]], color='b', linewidth=2)
        plt.axis('off')

        plt.title("Output data")
        plt.axis('off')

        # Plot the input data s
        plt.subplot(2, 2, 4)
        plt.plot(sout, linewidth=3)
        plt.title("Output spectrum")

    return input_data.reshape(n, 1), output_data.reshape(n, 1), edges


def generate_ktips_image():
    # generate Qtips image
    I = torch.zeros(1, 1, 64 * 2, 64 * 2)
    I[0, 0, 22 * 2:42 * 2, 30 * 2:34 * 2] = 1.0
    # The tips
    type = torch.randint(3, (1,))
    if type == 0:  # same color (2)
        I[0, 0, 18 * 2:22 * 2, 28 * 2:36 * 2] = 2
        I[0, 0, 40 * 2:44 * 2, 28 * 2:36 * 2] = 2
    elif type == 1:  # same color (3)
        I[0, 0, 18 * 2:22 * 2, 28 * 2:36 * 2] = 3
        I[0, 0, 40 * 2:44 * 2, 28 * 2:36 * 2] = 3
    else:  # same color (3)
        I[0, 0, 18 * 2:22 * 2, 28 * 2:36 * 2] = 2
        I[0, 0, 40 * 2:44 * 2, 28 * 2:36 * 2] = 3

    # transform
    a = (torch.randn(1) * 180).item()
    t = torch.randn(2) * 5
    s = 2 * torch.rand(1).item()
    c = 1 + 0.5 * torch.rand(1).item()
    X = FF.affine(I, angle=a, translate=(t[0].item(), t[1].item()), scale=c, shear=s)
    Y = torch.zeros_like(X)
    if type == 0:
        Y[X > 0] = 1
    elif type == 1:
        Y[X > 0] = 2
    else:
        Y[X > 0] = 3
    return X, Y, type


def get_graph_from_quadTree(S, A, Astd, k=9):
    b, c, n1, n2 = A.shape
    X = S.nonzero()
    I, J, K, L = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    V = A[I, J, K, L]
    B = S[I, J, K, L]
    Q = Astd[I, J, K, L]

    # Get the cell-center grid and scale [0,1]
    Kc = (K + B / 2) / n1
    Lc = (L + B / 2) / n2

    # put it all together
    data = torch.stack((Kc, Lc, 0 * V, 0 * Q / Q.max(), 0 * 1 / B.to(torch.float32)))
    data = data.t()
    DD = torch.cdist(data, data)
    G = torch.exp(-DD)
    _, indices = torch.topk(G, k, dim=1, largest=True)

    ii = torch.zeros(0)
    jj = torch.zeros(0)
    e = torch.arange(G.shape[0])
    for i in range(k):
        ii = torch.cat((ii, e))
        jj = torch.cat((jj, indices[:, i]))

    edge_index = torch.stack([ii, jj])

    data[:, 2] = V
    return edge_index.to(torch.long), 0, data


def create_quadtree_from_image(A, tol, method=0):
    # Initialize QuadTree
    S = torch.ones_like(A)
    Aot = A.clone()
    Astd = torch.ones_like(A)

    # Arrays of indices to know which ones we have not looked at
    bINDo = torch.zeros_like(A)
    bINDo = False

    # At the moment only for square images
    imsize = A.shape
    bsz = imsize[-1] // 2
    while bsz > 1:
        if method == 0:  # Max(A)-Min(A)<tol
            Akp = F.max_pool2d(A, kernel_size=bsz)
            Akm = -F.max_pool2d(-A, kernel_size=bsz)
            dA = Akp - Akm
            Ak = (Akp + Akm) / 2
        else:  # std(A)<tol
            Ak = F.avg_pool2d(A, kernel_size=bsz)
            AkF = F.interpolate(Ak, scale_factor=bsz)
            dA = (A - AkF) ** 2
            dA = F.avg_pool2d(dA, kernel_size=bsz)

        # Test if its a candidate for coarsening
        bIND = (dA < tol) * (bINDo == False)
        bINDf = F.interpolate(bIND.to(torch.float32), scale_factor=bsz).to(torch.bool)

        # Assign values to the coarse OcTree
        V = F.interpolate(Ak, scale_factor=bsz)
        Q = F.interpolate(dA, scale_factor=bsz)
        Aot[bINDf] = V[bINDf]
        Astd[bINDf] = Q[bINDf]

        # Assign values to the block size
        S[bINDf] = 0
        I, J, K, L = torch.where(bIND)
        S[I, J, K * bsz, L * bsz] = bsz

        # Update the indeces we have already coarsen
        bINDo = F.interpolate((bINDo + bIND).to(torch.float32), scale_factor=2).to(torch.bool)

        # Next resolution
        bsz = bsz // 2

    return Aot, Astd, S


def get_ktips_data(plotGraph=False, tol=1e-2):
    X, Y, type = generate_ktips_image()
    Aot, Astd, S = create_quadtree_from_image(X, tol=tol)
    edge_index, E, data = get_graph_from_quadTree(S, Aot, Astd, k=5)
    X = data[:, 2]

    Y = torch.zeros_like(X)
    if type == 0:
        Y[X > 0] = 1
    elif type == 1:
        Y[X > 0] = 2
    else:
        Y[X > 0] = 3
    xy = data[:, :2]

    if plotGraph:
        plot_graph(edge_index, X, xy)
    return X.unsqueeze(1), Y.unsqueeze(1), edge_index, xy


def plot_graph(edge_index, X, xy):
    # plot connectivity
    for i in range(edge_index.shape[1]):
        indL = edge_index[0, i]
        indR = edge_index[1, i]

        xx = [xy[indL, 0], xy[indR, 0]]
        yy = [xy[indL, 1], xy[indR, 1]]
        plt.plot(xx, yy, 'k', linewidth=1)

    plt.scatter(xy[:, 0], xy[:, 1], s=80, c=X, cmap="viridis", alpha=1)
    plt.show()


X, Y, edge_index, xy = get_ktips_data(plotGraph=False, tol=1e-1)
print(' ')
