import numpy as np 

rng = np.random.default_rng(0)

def rate_encode_vector(x_vec, T=200, p_max=0.2, seed=None):

    """
    params:
      x_vec: array-like shape (N,), values in [0,1]
      T: number of time steps
      p_max: max spike probability per timestep
      seed: optional random seed for reproducibility

    returns:
      spikes: np.ndarray shape (N, T) with 0/1
    """

    local_rng = np.random.default_rng(seed) if  seed is not None else rng 

    x = np.clip(np.asarray(x_vec, dtype=float), 0, 1) # turn x_vec into float, then clip between 0 and 1
    p = (x * p_max)[:, None] # compute neuron spike probability, reshape into a column 

    R = local_rng.random((len(x), T))
    spikes = (local_rng.random((len(x), T)) < p).astype(np.int8) ## each neuron i, time t cell 
        # if R[i, t] < p spikes [i, t] = 1, otherwise = 0
    return spikes 

    """
    [[0.50, 0.01, 0.90, 0.30, 0.02],
    [0.10, 0.20, 0.05, 0.80, 0.14],
    [0.17, 0.99, 0.04, 0.50, 0.60]]
    """

def rate_decode_vector(spikes, p_max=0.2):
    """params: 
        spikes: matrix of shape(N, T), x ∈ (0, 1)
            N: # of neurons 
            T: number of timesteps 
        returns_hat shape (N, )
    """

    T = spikes.shape[1]
    
    ## spike count per neuron (axis=1 --> row) / T * p_max 
    ## T * p_max: max expected blinks in that observed T window 
    ## it blinked 14% of the time, since max possible is 20%, brightness is 14/20 = 70%

    return spikes.sum(axis=1) / (T * p_max) 


## encode/decode tester 
## S: spike matrix (N, T)
## x_hat: length (N, )
if __name__== "__main__":
    x_vec = [0.2, 0.6, 0.7, 0.9]
    S = rate_encode_vector(x_vec, T=200, p_max=0.2, seed=0)
    x_hat = rate_decode_vector(S, p_max=0.2)

    print("x_vec:", x_vec)
    print("spike counts:", S.sum(axis=1).tolist())
    print("decoded:", x_hat.tolist())