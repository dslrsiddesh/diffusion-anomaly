"""
Single machine training helpers (non-distributed version).
"""

import io
import torch as th
import blobfile as bf

def setup_dist():
    """
    Setup for single GPU/CPU usage.
    """
    # Nothing to setup for single machine training
    if th.cuda.is_available():
        # Set default GPU device
        th.cuda.set_device(0)

def dev():
    """
    Get the device to use (CPU or single GPU).
    """
    if th.cuda.is_available():
        return th.device("cuda:0")  # Use first GPU if available
    return th.device("cpu")

def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file directly.
    """
    with bf.BlobFile(path, "rb") as f:
        data = f.read()
    return th.load(io.BytesIO(data), **kwargs)

def sync_params(params):
    """
    No need to sync parameters for single machine training.
    """
    pass  # Do nothing since we're not distributing