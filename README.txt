This repository supplies the sufficient code to run the experiments outlined in the paper: Towards Efficient Training of Graph Neural Networks: A Multiscale Approach.

To run the experiments, select the task and data to run.
Dataset locations:
* QTips, MSINT: 'qtips_mnist/'
* Cora, CiteCeer, PubMed, WikiCS, Flickr, DBLP, PPI (transductive), BlogCatalog, Facebook, OGBN-Arxiv, OGBN-Mag: 'transductive_learning/'
* PPI-inductive, NCI1, MolHIV: 'inductive_learning/'
* Shapenet: 'shapenet/'

Specific details of each dataset are detailed below.



### Using 'qtips_mnist/'

To run multilevel training of QTips dataset, run QTips.py. For MNIST, run MNIST.py.
Inside the script, you may update the following parameters:
--n_runs : (int) number of runs of the experiment. Default = 3
--levels : (list of int) number of multiscale levels, out od [1,2,3,4]. Default = [3]
--methods : (list of str) methods described in the paper, out of ["random", "topk", "subgraph", "RnS"]. Default = ["random"]
--nets : (list of str) models to be used for training. Default = ["GCN"]
--ks : (list of int) connectivity enhancement factor. 1 is the original graph. Default = [3].

The datasets are generated using generate_mnist_data.py and generate_qtips_data.py


### Using 'transductive_learning/'

To run multilevel training of Cora, CiteCeer, PubMed, WikiCS, Flickr, DBLP, PPI (trunsductive), BlogCatalog, Facebook, run transductive_datasets.py.
Inside the script, you may update the following parameters:
--n_runs : (int) number of runs of the experiment. Default = 3
--levels : (list of int) number of multiscale levels, out od [1,2,3,4]. Default = [2]
--methods : (list of str) methods described in the paper, out of ["random", "topk", "subgraph", "RnS"]. Default = ["random"]
--nets : (list of str) models to be used for training, out of ["GCN", "GIN", "GAT"]. Default: ["GCN"]
--datas : (list of str) datasets chosen for experiments. Out of ["flicker", "wiki", "dblp", "ppi", "blog", "facebook", "pub", "cora", "cite"]. Default = ["pub"]
--p_vals : (list of int) connectivity enhancement factor. 1 is the original graph. Default = [1].

To run multilevel training of OGBN-Arxiv, run OGBN-Arxiv.py.
Inside the script, you may update the following parameters:
--n_runs : (int) number of runs of the experiment. Default = 3
--levels : (list of int) number of multiscale levels, out od [1,2,3,4]. Default = [3]
--methods : (list of str) methods described in the paper, out of ["random", "topk", "subgraph", "RnS"]. Default = ["random"]
--nets : (list of str) models to be used for training, out of ["GCN", "GIN", "GAT"]. Default = ["GCN"]
--p_vals : (list of int) connectivity enhancement factor. 1 is the original graph. Default = [1].

To run multilevel training of OGBN-Mag, run OGBN-Mag.py.
Inside the script, you may update the following parameters:
--n_runs : (int) number of runs of the experiment. Default = 3
--levels : (list of int) number of multiscale levels, out od [1,2,3,4]. Default = [3]
--methods : (list of str) methods described in the paper, out of ["random", "topk", "subgraph", "RnS"]. Default = ["random"]
--nets : (list of str) models to be used for training, out of ["GCN", "GIN", "GAT"]. Default = ["GCN"]
--p_vals : (list of int) connectivity enhancement factor. 1 is the original graph. Default = [1].


### Using 'inductive_learning/'

To run multilevel training of PPI-inductive, run PPI.py.
Inside the script, you may update the following parameters:
--n_runs : (int) number of runs of the experiment. Default = 3
--methods : (list of str) methods described in the paper, out of ["random", "topk", "subgraph", "RnS"]. Default = ["random"]
--num_epochs : (intr) Number of epochs. Default: 500
--ks : (list of int) connectivity enhancement factor. 1 is the original graph. Default = [1].

To run multilevel training of NCI1, run NCI1.py.
Inside the script, you may update the following parameters:
--n_runs : (int) number of runs of the experiment. Default = 3
--levels : (list of int) number of multiscale levels. Default = [2]
--methods : (list of str) methods described in the paper, out of ["random", "topk", "subgraph", "RnS"]. Default = ["random"]
--nets : (list of str) models to be used for training. Default: ["GCN"]
--ks : (list of int) connectivity enhancement factor. 1 is the original graph. Default = [1].

To run multilevel training of MolHIV, run MolHIV.py.
Inside the script, you may update the following parameters:
--n_runs : (int) number of runs of the experiment. Default = 3
--levels : (list of int) number of multiscale levels. Default = [2]
--methods : (list of str) methods described in the paper, out of ["random", "topk", "subgraph", "RnS"]. Default = ["random"]
--nets : (list of str) models to be used for training. Default = ["GCN"]
--ks : (list of int) connectivity enhancement factor. 1 is the original graph. Default = [1].


### Using 'shapenet/'

To run multilevel training of Shapenet, run Shapenet.py.
Inside the script, you may update the following parameters:
--n_runs : (int) number of runs of the experiment. Default = 3
--levels : (list of int) number of multiscale levels. Default = [3]
--methods : (list of str) methods described in the paper, out of ["random", "subgraph", "RnS"]. Default = ["random"]
--ks : (list of int) connectivity factor used for the knn algorithm, paper shows [6, 10, 20]. Default = [10].
Note: this script requires the possession of the 'Shapenet' dataset which is not available directly threw pytorch-geometric. Update the "path" for proper data location.



### Notes

- Tested on Linux (Ubuntu 2.04) and Windows 11.
- Use a virtual environment (venv or conda) to avoid conflicts.
- GPU users: ensure CUDA version matches the installed PyTorch version.
