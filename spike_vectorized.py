"""
spike_vectorized.py implements a simple vectorized Leaky Integrate-and-Fire (LIF) neuron simulation.

Run this script to generate randomized, noisy action potential spikes and membrane potential traces for a population of neurons.

"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib 
matplotlib.use('TkAgg')

"""
Synaptic Weights: 
   Excitatory weights - positive values to increase potential, driving it to spike 
   Inhibitary weights - negative values to decrease membrane potential (GABA)

   Weighted Matrix 
        N neurons = N x N matrix weight 
        W[i, j] represents weight from j to i 

    Synaptic Weights 
        Input current depends on spikes of other neurones and synaptic weights 
        Input current to neuron i at t+1: 
            I(t+1)=∑​j W[i,j]⋅S[j,t]
            (sum of weighted spikes from all other neurons)
"""

# set parameters 
N = 100 # nuber of neurons 

def create_weight_matrix(N, inhibitory_ratio=0.2, seed=None):
    """
    Create a weight matrix with excitatory and inhibitory connections.
    
    Parameters:
    -----------
    N : int
        Number of neurons
    inhibitory_ratio : float
        Ratio of inhibitory connections (default 0.2 for 20%)
    seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    W : ndarray
        Weight matrix of shape (N, N) with positive (excitatory) and negative (inhibitory) values
    """
    if seed is not None:
        np.random.seed(seed)
    
    W = np.random.uniform(0, 1, (N, N))  # Start with all positive (excitatory)
    inhibitory_mask = np.random.rand(N, N) >= (1 - inhibitory_ratio)  # Select inhibitory neurons
    W[inhibitory_mask] = -W[inhibitory_mask]  # Flip to negative (inhibitory)
    
    return W

def run_vectorized_lif(
    num_steps = 100, 
    num_neurons = 100,
    threshold = 1.0,
    reset_value = 0.0,
    tau = 20.0,
    dt=1.0,
    input_current=0.08, 
    plot=True
): 

# initialize membrane potentials and spikes 
    V = np.zeros(num_neurons) 
    spikes = np.zeros((num_steps, num_neurons))
    I_base = np.full(num_neurons, input_current)
    V_record = np.zeros((num_steps, num_neurons))  # To record membrane potential over time

    # Create weight matrix for synaptic connections
    W = create_weight_matrix(num_neurons, inhibitory_ratio=0.2, seed=42)
    
    # Print weight matrix statistics (once, before simulation)
    print("\n=== Weight Matrix Statistics ===")
    print(f"Matrix shape: {W.shape}")
    print(f"Excitatory connections: {np.sum(W > 0)} ({np.sum(W > 0) / W.size * 100:.1f}%)")
    print(f"Inhibitory connections: {np.sum(W < 0)} ({np.sum(W < 0) / W.size * 100:.1f}%)")
    print(f"Weight range: [{W.min():.3f}, {W.max():.3f}]")
    print(f"\nSample weights (5x5):")
    print(W[:5, :5].round(3))

    # simulation loop 
    for t in range(num_steps): 
        # Add random noise to the input current at each time step
        noise = np.random.normal(0, 0.2, num_neurons)
        
        # Calculate synaptic input 
        if t > 0:
            I_synaptic = (1.0 / num_neurons) * (W @ spikes[t-1, :])  # Scale by 1/N
        else: 
            I_synaptic = np.zeros(num_neurons)
        
        I = I_base + noise + I_synaptic 
        dV = (dt / tau) * (-V + I) # change in membrane potential at this time step 
        V += dV # updates membrane potential with electrical signals
        spiked = V >= threshold # checks which neurons have reached threshold (usually boolean)
        spikes[t, spiked] = 1 # records spike that fired at this time step to 1 
        V[spiked] = reset_value # resets membrane potential to 0
        V_record[t] = V  # Record the membrane potential for all neurons

    if plot:
        print("About to show plot")
        fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        # Plot membrane potential of the first neuron
        axs[0].plot(V_record[:, 5], label='Neuron 5')
        axs[0].set_ylabel('Membrane Potential (V)')
        axs[0].set_title('Membrane Potential of Neuron 0')
        axs[0].legend()
        # Plot spike raster
        axs[1].imshow(spikes.T, aspect='auto', cmap='Greys', interpolation='nearest')
        axs[1].set_xlabel('Time step')
        axs[1].set_ylabel('Neuron index')
        axs[1].set_title('Spike Raster Plot (Vectorized LIF)')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Set parameters for visible spiking - simulate all 100 neurons
    # Strong drive to ensure sustained activity
    run_vectorized_lif(num_neurons=100, input_current=5.0, threshold=0.5, plot=True) # measured in millivolts 


