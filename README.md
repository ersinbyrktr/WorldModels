# World Models Implementation

This is an implementation of the [World Models](https://worldmodels.github.io/) paper by Ha and Schmidhuber (2018). The implementation consists of the following components:

1. **VAE (Vision Module)**: Compresses high-dimensional observations into a compact latent representation.
2. **MDN-RNN (Dynamics Module)**: Predicts future latent states given past latent states.
3. **World Model**: Combines the VAE and MDN-RNN for environment simulation and planning.

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd world-models

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Data Preparation

Store your environment frames (e.g., from CarRacing-v0) in a directory structure as follows:

```
data/
└── carracing/
    ├── episode1.npy
    ├── episode2.npy
    └── ...
```

Each `.npy` file should contain an array of shape `(N, H, W, 3)` where `N` is the number of frames, and `H`, `W` are the height and width of each frame.

## Model Architecture

### VAE
- Encoder: 4 convolutional layers
- Latent dimension: 32 (default)
- Decoder: 4 transposed convolutional layers

### MDN-RNN
- LSTM with 768 hidden units (default)
- Mixture Density Network with 5 Gaussian mixtures
- Takes latent vectors as input and predicts next latent state

## Citation

```
@article{ha2018worldmodels,
  title={World Models},
  author={Ha, David and Schmidhuber, J{\"u}rgen},
  journal={arXiv preprint arXiv:1803.10122},
  year={2018}
}
```
