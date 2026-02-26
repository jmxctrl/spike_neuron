"""
Rate-based spike encoding and decoding functions

This module provides functions to convert continuous values into spike trains
using rate coding, where higher values produce higher spike rates.

Supports both:
- Single-value encoding: rate_encode(), rate_decode()
- Vector encoding: rate_encode_vector(), rate_decode_vector()
"""
import numpy as np

# Global random number generator
rng = np.random.default_rng(0)


# ========== Single Value Encoding ==========

def rate_encode(x, T=200, p_max=0.2):
    """
    Encode a single value as a spike train using rate coding.
    
    Args:
        x: value to encode in [0, 1]
        T: simulate T chances to spike (timesteps)
        p_max: if input maxed out, each timestep neuron spikes with prob p_max
    
    Returns:
        spikes: np.ndarray shape (T,) with 0/1 values
    
    Example:
        >>> spikes = rate_encode(0.5, T=200, p_max=0.2)
        >>> len(spikes)
        200
    """
    x = float(np.clip(x, 0, 1))  # 0: never 1: always
    p = x * p_max  # expectation per timestamp
    spikes = (rng.random(T) < p).astype(np.int8)  # if < p --> (1), if > p -->(0)
    return spikes


def rate_decode(spikes, p_max=0.2):
    """
    Decode a spike train back to a continuous value.
    
    Args:
        spikes: np.ndarray shape (T,) with 0/1 values
        p_max: maximum spike probability used during encoding
    
    Returns:
        estimated value in [0, 1] range
    
    Example:
        >>> spikes = np.array([0,1,0,0,1,0,0,1,0])
        >>> rate_decode(spikes, p_max=0.2)
        1.66...  # estimated input value after rescaling by p_max
    """
    T = len(spikes)
    return spikes.sum() / (T * p_max)  # average of spike rate / P(max)


# ========== Vector Encoding (Multiple Values) ==========

def rate_encode_vector(x_vec, T=200, p_max=0.2, seed=None):
    """
    Encode a vector of values as spike trains using rate coding.
    
    Args:
        x_vec: array-like shape (N,), values in [0, 1]
        T: number of timesteps
        p_max: maximum spike probability per timestep
        seed: optional random seed for reproducibility
    
    Returns:
        spikes: np.ndarray shape (N, T) with 0/1 values
    
    Example:
        >>> x_vec = [0.2, 0.6, 0.9]
        >>> spikes = rate_encode_vector(x_vec, T=100)
        >>> spikes.shape
        (3, 100)
    """
    local_rng = np.random.default_rng(seed) if seed is not None else rng
    
    # Clip values to [0, 1] range
    x = np.clip(np.asarray(x_vec, dtype=float), 0, 1)
    
    # Compute spike probability for each neuron: (N, 1)
    p = (x * p_max)[:, None]
    
    # Generate spikes: if random value < probability, spike = 1
    spikes = (local_rng.random((len(x), T)) < p).astype(np.int8)
    
    return spikes


def rate_decode_vector(spikes, p_max=0.2):
    """
    Decode spike trains back to continuous values.
    
    Args:
        spikes: np.ndarray shape (N, T) with 0/1 values
        p_max: maximum spike probability used in encoding
    
    Returns:
        x_hat: np.ndarray shape (N,) with decoded values in [0, 1]
    
    Example:
        >>> spikes = np.array([[1,0,1,0], [1,1,1,1]])
        >>> rate_decode_vector(spikes, p_max=0.2)
        array([0.5, 1.0])
    """
    T = spikes.shape[1]
    
    # Average spike rate / max possible rate
    # If neuron spikes 14% of time and max is 20%, value is 14/20 = 0.7
    return spikes.sum(axis=1) / (T * p_max)


# ========== Test and Demo ==========

if __name__ == "__main__":
    print("="*60)
    print("TESTING RATE ENCODING")
    print("="*60)
    
    # Test vector encoding
    x_vec = [0.2, 0.6, 0.7, 0.9]
    S = rate_encode_vector(x_vec, T=200, p_max=0.2, seed=0)
    x_hat = rate_decode_vector(S, p_max=0.2)
    
    print("\nOriginal values:", x_vec)
    print("Spike counts:", S.sum(axis=1).tolist())
    print("Decoded values:", x_hat.tolist())
    print("Reconstruction error:", np.abs(np.array(x_vec) - x_hat).tolist())
    
    print("\n✓ Test complete")
