from spike_vectorized import create_weight_matrix, calculate_stdp, train_with_stdp, run_vectorized_lif
import numpy as np
import matplotlib.pyplot as plt

# define light sensors 
sensor_neurons = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ,10]

def run_with_light(W, light_on=True):
    """
    Run simulation with light input to sensor neurons 
        W: neuron connection of weights 
        light_on: determins input current sensor receive 
    """

    if light_on:
        input_current = 10.0
    else: 
        input_current = 0.5 

    
    spikes, voltages = run_vectorized_lif(
        W=W, 
        num_steps=num_steps, 
        input_current=input_current, 
        threshold=0.5,
        plot=False
    )

    return spikes, voltages 