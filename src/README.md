# Source Code Structure

This directory contains the organized source code for the revealing-attention project.

## Module Organization

### `replay_buffer.py`
Contains classes and functions related to replay buffer management:
- `Transition`: Named tuple for storing transition data
- `HDF5ReplayBufferRAM`: Main replay buffer class for HDF5 storage with RAM caching

### `models.py`
Contains neural network model definitions:
- `Autoencoder`: Convolutional autoencoder for state representation
- `Motor_predictor_fwd`: Motor prediction network
- `CTR_Attention_SA`: Self-attention based contextualized task-relevant attention
- `CTR_Attention_dil`: Dilated convolution based attention
- `Gaze_predictor_pool`: Gaze prediction network

### `utils.py`
Contains utility functions:
- `count_and_list_parameters()`: Model parameter counting
- `print_AE_test_weights()`: Autoencoder weight debugging
- `get_lr()`: Learning rate extraction
- `dropout_no_scaling()`: Custom dropout implementation
- `BernoulliSTE`: Straight-through estimator for binary attention
- `sample_binary_attention()`: Binary attention sampling
- `redistribute_lam()`: Lambda redistribution utility
- `topk_binarize()`: Top-k binarization

## Usage

### In Notebooks
```python
import sys
sys.path.insert(0, '../src')

from replay_buffer import HDF5ReplayBufferRAM, Transition
from models import Autoencoder, CTR_Attention_SA
from utils import count_and_list_parameters, get_lr
```

### As Installed Package
```python
# After running: pip install -e .
from src.replay_buffer import HDF5ReplayBufferRAM, Transition
from src.models import Autoencoder, CTR_Attention_SA
from src.utils import count_and_list_parameters, get_lr
```

## Dependencies

See `setup.py` for full dependency list. Main requirements:
- torch >= 1.9.0
- numpy >= 1.21.0
- h5py >= 3.1.0
- antialiased-cnns >= 0.0.4
