# Towards Efficient Training of Graph Neural Networks: A Multiscale Approach

This repository provides the code required to reproduce the experiments from the paper:  
**“Towards Efficient Training of Graph Neural Networks: A Multiscale Approach.”**  
<https://arxiv.org/pdf/2503.19666>

---

## Setup

First, create an environment with the required dependencies:

```bash
pip install -r requirements.txt
```

**Notes:**
- Tested on **Linux (Ubuntu 20.04)** and **Windows 11**.
- Recommended to use a virtual environment (**venv** or **conda**) to avoid dependency conflicts.
- **GPU users:** ensure your CUDA version matches the installed PyTorch version.

---

## Multiscale Training

Below are the entry points for running multiscale training on all supported datasets.

### Synthetic Datasets
- **QTips**:  
  ```bash
  python QTips.py
  ```

- **MNIST**:  
  ```bash
  python MNIST.py
  ```
  Datasets are generated using:
  - `generate_mnist_data.py`
  - `generate_qtips_data.py`

---

### Transductive Benchmark Datasets
Cora, CiteSeer, PubMed, WikiCS, Flickr, DBLP, BlogCatalog, Facebook, and PPI (transductive):

```bash
python transductive_datasets.py
```

---

### OGB Datasets
- **OGBN-Arxiv:**
  ```bash
  python OGBN-Arxiv.py
  ```

- **OGBN-Mag:**
  ```bash
  python OGBN-Mag.py
  ```

---

### Inductive / Molecule / Point Cloud Datasets
- **PPI (inductive):**
  ```bash
  python PPI.py
  ```

- **NCI1:**
  ```bash
  python NCI1.py
  ```

- **MolHIV:**
  ```bash
  python MolHIV.py
  ```

- **ShapeNet:**
  ```bash
  python Shapenet.py
  ```
  **Note:** ShapeNet is *not* available directly through PyTorch Geometric.  
  Download it separately and update the `"path"` argument accordingly.

---

## BibTeX

```bibtex
@article{Gal2025GraphMultiscale,
  title         = {Towards Efficient Training of Graph Neural Networks: A Multiscale Approach},
  author        = {Gal, Eshed and Eliasof, Moshe and Sch{\"o}nlieb, Carola-Bibiane and Kyrchei, Ivan I. and Haber, Eldad and Treister, Eran},
  journal       = {Transactions on Machine Learning Research},
  year          = {2025},
  url           = {https://arxiv.org/abs/2503.19666},
}
```
