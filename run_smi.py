import torch
import numpy as np


def compute_smi(_repr, _label, n_samples):
    _repr = _repr.detach().cpu().numpy()
    _repr = _repr.reshape((_repr.shape[0], -1))
    _repr = np.expand_dims(_repr, axis=1)

    _label = _label.detach().cpu().numpy()

    samples = np.random.randn(_repr.shape[-1], n_samples)
    samples /= np.linalg.norm(samples, axis=0)

    smi = samples[np.newaxis, :]
    return
