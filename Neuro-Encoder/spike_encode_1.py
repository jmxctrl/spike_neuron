import numpy as np 
import matplotlib.pyplot as plt 

rng = np.random.default_rng(0)

def rate_encode(x, T=200, p_max=0.2):
    ## params ## 
        ## x: value to encode [0, 1]
        ## T=200: simulate 200 chances to spike  
        ## p_max: if input maxxed out, each timestep neuron spikes with prob 20% 

    x = float(np.clip(x, 0, 1)) # 0: never 1: always 
    p = x * p_max ## expectation per timestamp 
    spikes = (rng.random(T) < p).astype(np.int8) ## if < p --> (1), if > p -->(0)
    return spikes 


def rate_decode(spikes, p_max=0.2):
    T = len(spikes)
    return spikes.sum() / (T * p_max) ## average of spike rate / P(max)


"""""""""""""""""""""""""""""""""""""""""""""
# spikes = [0,1,0,0,1,0,0,1,0]
# len(spikes)   # 9  (timesteps)
# sum(spikes)   # 3  (spikes)

returns 1.66 -- estimated input value x after rescaling p(max)
"""""""""""""""""""""""""""""""""""""""""""""

