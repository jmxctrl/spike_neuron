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

def calculate_stdp(spikes, W, A_plus=0.02, A_minus=0.022, tau_plus=20.0, tau_minus=20.0):
    """
    Calculate weight updates based on spike timing 

    spikes: (T, N) spike time at each neuron i 
    W: (N, N) current weight matrix 

    Returns: Delta_w(N, N) - weight changes applied 
    """
    # 1. extract spike times from spikes array  
    # 2. get W (N, N) of weight matrix 
    # 3. calculate the difference of pre and post neuron 
    # 4. calculate magnitude 
    # 5. if Δt>0: Long Term Potentiation 
            # else: Δt<0: Long Term Depressio
    # 6. store dw in delta_W[i, j]
    # 7. sum all ΔW from spike apirs 
    # 8. repeat for every connection W[i, j]
    # 9. return Delta_W matrix 

    N = W.shape[0]
    delta_W = np.zeros((N, N)) 
    
    # loop through (N, N) and get spikes of i and j 
    for i in range(N):
        for j in range(N):
            t_pre = np.where(spikes[:, j] == 1)[0]
            t_post = np.where(spikes[:, i] == 1)[0] 

            if len(t_pre) == 0 or len(t_post) == 0: 
                continue 
            
            total_dw = 0.0  # Initialize for this connection
            
            for t_post_spike in t_post: 
                for t_pre_spike in t_pre:
                    # calculate delta_t
                    delta_t = t_post_spike - t_pre_spike 
                    
                    if abs(delta_t) > 100:  # Skip if too far apart
                        continue
            
                    if delta_t > 0:  # LTP 
                        magnitude = np.exp(-delta_t / tau_plus)
                        dw = A_plus * magnitude
                    else:  # LTD 
                        magnitude = np.exp(-delta_t / tau_minus) 
                        dw = -A_minus * magnitude 
                    
                    # accumulate weight changes 
                    total_dw += dw 

            delta_W[i, j] = total_dw
    return delta_W

def train_with_stdp(num_episodes=10, num_steps=100, num_neurons=100):
    """
    Train network using STDP over multiple episodes 
    """

    # 1. Initialize weights 
    # 2. Loop Episodes x times: 
        # run netowrk with current weights
        # get spike patterns 
        # calculate how weights should change 
        # update weights 

    # 3. return trained network 

    W = create_weight_matrix(num_neurons)  # start point 
    
    # Save original neuron identity (doesn't change during training)
    excitatory_mask = W > 0
    inhibitory_mask = W < 0

    for episode in range(num_episodes):
        print(f"Episode {episode+1}/{num_episodes}")
        
        spikes, voltages = run_vectorized_lif(
            W=W,
            num_steps=num_steps,  # Fixed typo: was numsteps
            num_neurons=num_neurons, 
            threshold=0.5,
            input_current=5.0, 
            plot=False
        )  # run network 

        delta_W = calculate_stdp(spikes, W, A_plus=0.0001, A_minus=0.00012)
        delta_W = delta_W * 0.001  # Scale down to prevent saturation
        
        W = W + delta_W 

        # Enforce original neuron identity (Dale's principle)
        W[excitatory_mask] = np.clip(W[excitatory_mask], 0.01, 0.9)  # Stay positive
        W[inhibitory_mask] = np.clip(W[inhibitory_mask], -0.9, -0.01)  # Stay negative
        
        print(f"  Weight change: {np.abs(delta_W).mean():.6f}")
        print(f"  W range: [{W.min():.3f}, {W.max():.3f}]")
        print(f"  W mean (exc): {W[excitatory_mask].mean():.3f}, W mean (inh): {W[inhibitory_mask].mean():.3f}")

    return W, spikes, voltages

def run_vectorized_lif(
    W=None,  # Optional: pass in weight matrix for training
    num_steps=100, 
    num_neurons=100,
    threshold=1.0,
    reset_value=0.0,
    tau=20.0,
    dt=1.0,
    input_current=0.08, 
    plot=True
): 

    """
    Run Leaky Integrate and Fire 

    Params: 
        threshold: 
        tau (τ): time constant it takes for voltage to decay of initial value (how fast neuron returns to reset)
            tau = 20ms, voltage decays 63% in 20 milliseconds
            i.e. dt / dau = 1.0/20.0 = 0.05 --> 5% decay rate per millisecond 
        V: membrane voltage is mV
        dt: time step, how much time passes between updates 
        dv: (decay rate) * (distance from target)

    Return spike data and membrane voltages
    """

    # initialize membrane potentials and spikes 
    V = np.zeros(num_neurons) 
    spikes = np.zeros((num_steps, num_neurons))
    I_base = np.full(num_neurons, input_current)
    V_record = np.zeros((num_steps, num_neurons))  # To record membrane potential over time

    # Create weight matrix if not provided
    if W is None:
        W = create_weight_matrix(num_neurons, inhibitory_ratio=0.2, seed=42)
    else:
        # Verify W has correct shape
        assert W.shape == (num_neurons, num_neurons), f"W shape {W.shape} doesn't match num_neurons {num_neurons}"
    
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
    
    return spikes, V_record  # Return spike data and membrane voltages


if __name__ == "__main__":
    # Train network with STDP
    print("=== Training Network with STDP ===")
    trained_W, final_spikes, final_voltages = train_with_stdp(
        num_episodes=5,
        num_steps=100,
        num_neurons=100
    )
    
    # Visualize final trained network
    print("\n=== Final Trained Network ===")
    run_vectorized_lif(
        W=trained_W,
        num_neurons=100,
        num_steps=100,
        input_current=5.0,
        threshold=0.5,
        plot=True
    ) 


