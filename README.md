# Cellgroup

<div align="center">

# 🔬 Cellgroup

*A sophisticated library for clustering and analyzing cells in fluorescent microscopy well-plate images*

[![Tests](https://github.com/username/cellgroup/actions/workflows/test_pr.yml/badge.svg)](https://github.com/username/cellgroup/actions/workflows/test_pr.yml)
[![Tests Full](https://github.com/username/cellgroup/actions/workflows/test_full.yml/badge.svg)](https://github.com/username/cellgroup/actions/workflows/test_full.yml)
[![Documentation Status](https://readthedocs.org/projects/cellgroup/badge/?version=latest)](https://cellgroup.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/cellgroup.svg)](https://badge.fury.io/py/cellgroup)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Slack](https://img.shields.io/badge/chat-on%20slack-purple?logo=slack)](https://join.slack.com/t/cellgroup/shared_invite/...)

[Documentation](https://cellgroup.readthedocs.io/) |
[Examples](examples/) |
[Contributing](CONTRIBUTING.md) |
[Paper](https://arxiv.org/abs/...)

</div>

![Cellgroup Overview](docs/images/overview.png)

## 🌟 Highlights

- 🧬 **Advanced Cell Analysis**: State-of-the-art segmentation and clustering
- 🔧 **Production Ready**: Extensively tested, documented, and optimized
- 🚀 **High Performance**: Cython-accelerated clustering algorithms
- 📊 **Rich Visualization**: Comprehensive plotting and analysis tools
- 🤝 **Easy Integration**: Works seamlessly with existing pipelines

## ⚡️ Quick Start

```bash
# Install using pip
pip install cellgroup

# Or using conda
conda install -c conda-forge cellgroup
```

```python
import cellgroup as cg

# Load and preprocess your image
image = cg.load_image("my_cells.tif")
preprocessed = cg.preprocess(image)

# Segment and cluster cells
cells = cg.segment(preprocessed)
clusters = cg.cluster(cells)

# Analyze results
analysis = cg.analyze(clusters)
cg.plot_results(analysis)
```

## 🎯 Key Features

### 🔍 Image Preprocessing
```python
# Advanced noise reduction
from cellgroup.preprocessing import analyze_noise, denoise

# Measure noise levels
noise_metrics = analyze_noise(image, 
    methods=['gradients', 'intensity', 'frequencies'])

# Apply state-of-the-art denoising
denoised = denoise(image, method='deep_learning')
```

### 🧩 Cell Segmentation
```python
from cellgroup.segmentation import segment_cells

# Configure segmentation parameters
config = {
    'model': 'stardist',
    'threshold': 0.5,
    'min_size': 100
}

# Perform segmentation
masks = segment_cells(image, config=config)
```

### 🎭 Clustering
```python
from cellgroup.clustering import cluster_cells

# Perform temporal clustering
clusters = cluster_cells(
    cells,
    method='density',
    temporal=True,
    min_cluster_size=5
)
```

## 🛠 Configuration

```python
import os

# Configure through environment variables
os.environ['CELLGROUP_CLUSTERS'] = 'all'
os.environ['CELLGROUP_BACKEND'] = 'cuda'  # For GPU acceleration
```

| Variable | Type | Description |
|----------|------|-------------|
| `CELLGROUP_CLUSTERS` | str | Clustering method (`'all'`, `'density'`, `'temporal'`) |
| `CELLGROUP_BACKEND` | str | Computation backend (`'cpu'`, `'cuda'`) |
| `CELLGROUP_PLOT` | bool | Enable visualization generation |
| `CELLGROUP_ANALYZE` | bool | Enable statistical analysis |

## 📊 Example Applications

<table>
<tr>
<td>
<img src="docs/images/cell_tracking.gif" width="200"/>
</td>
<td>

```python
# Track cells over time
tracked = cg.track_cells(
    images,
    temporal=True,
    track_method='flow'
)
```

</td>
</tr>
</table>

## 📚 Documentation

Visit our [documentation](https://cellgroup.readthedocs.io/) for:
- 📖 Detailed API reference
- 🎓 Tutorials and examples
- 🔧 Advanced configuration
- 💡 Best practices

## 🤝 Contributing

We welcome contributions! See our [Contributing Guide](CONTRIBUTING.md) for details.

```bash
# Set up development environment
git clone https://github.com/username/cellgroup.git
cd cellgroup
pip install -e ".[dev]"

# Run tests
pytest tests/
```

## 📜 Citation

```bibtex
@article{putignano2024cellgroup,
  title={Cellgroup - A library to cluster and analyse cells in wells},
  author={Putignano, Guido; Carrara, Federico; D'Ascenzo, Davide},
  journal={ETH Zürich},
  year={2024}
}
```

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=username/cellgroup&type=Date)](https://star-history.com/#username/cellgroup&Date)

---

<div align="center">
Made with ❤️ for the scientific community
</div>