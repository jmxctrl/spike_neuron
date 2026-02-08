"""
spike_vectorized.py implements a simple vectorized Leaky Integrate-and-Fire (LIF) neuron simulation.

Run this script to generate randomized, noisy action potential spikes and membrane potential traces for a population of neurons.

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib 
matplotlib.use('TkAgg')

# set parameters 

def run_vectorized_lif(
    num_neurons = 10,
    num_steps = 100, 
    threshold = 1.0,
    reset_value = 0.0,
    tau = 20.0,
    dt=1.0,
    input_current=0.08, 
    plot=True
): 

# initialize state variables 
    V = np.zeros(num_neurons) 
    spikes = np.zeros((num_steps, num_neurons))
    I_base = np.full(num_neurons, input_current)
    V_record = np.zeros((num_steps, num_neurons))  # To record membrane potential over time


    # simulation loop 
    for t in range(num_steps): 
        # Add random noise to the input current at each time step
        noise = np.random.normal(0, 0.2, num_neurons)
        I = I_base + noise
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
    # Set parameters for visible spiking
    run_vectorized_lif(input_current=1.0, threshold=0.1, plot=True) # measured in millivolts 


