"""
    Deploy trained SNN on robot with real sensors (left, center, right)
        BEST WEIGHTS NOW: 267
        normalize distances to [0, 1] scale for SNN inference 
        SNN inference to classify actions 
"""
import numpy as np 
import sys 
import os
import time 

from spike_vectorized import run_vectorized_lif
from light_task import encode_sensors_to_spikes
from neuromodulation import classify_actions
from rasp_sensors import SensorFilter, cleanup

NUM_NEURONS = 100
NUM_STEPS = 10 

class SNNController: 
    def __init__(self, weights_path):
        """initialize robo controller with trained weights"""
        
        #load trained weights 
        self.weights = np.load(weights_path)
        print(f"✓ Loaded weights from: {weights_path}")
        print(f"  Shape: {self.weights.shape}")

        # create sensor filter 
        self.sensor_filter = SensorFilter(window_size=5)
        print(f"✓ Initialized sensor filter (window=5)")

    def normalize_sensors(self, distances, max_distance=400):
        """convert distances to [0, 1]"""
        return [min(d, max_distance) / max_distance for d in distances]

    def run_inference(self):
        # 1. read all 3 sensors returned from filtered 
        filtered_distances = self.sensor_filter.read_all_sensors()

        # 2. normalize all 3 sensors at once 
        normalized_inputs = self.normalize_sensors(filtered_distances)

        # 3. feed to SNN 
        input_spikes = encode_sensors_to_spikes(
            normalized_inputs, 
            T=NUM_STEPS,
            p_max=0.2
        )

        # 4. run SNN forward pass 
        spikes, voltages, pathway_history = run_vectorized_lif(
            W=self.weights, 
            external_spikes=input_spikes, 
            num_steps=NUM_STEPS,
            num_neurons=NUM_NEURONS, 
            plot=False
        )
        
        # 5. classify actions from 
        # 1) pathway_history 
        # 2) averages last 10 timesteps 
        # 3) divide neurons into 3 pools 
        #4) extracts winning pool actions      
        action = classify_actions(pathway_history)

        return action 


    def run(self, test_mode=True, num_iterations=50):
        """main control loop"""
        print(f"\nRunning in {'TEST' if test_mode else 'FULL'} mode for {num_iterations} iteraitons\n")

        try: 
            # loop num_iterations times 
            for i in range(num_iterations):
                action = self.run_inference()
                print(f"[{i+1}/{num_iterations}] Actions: {action}")
                time.sleep(0.2)
        except KeyboardInterrupt:
            print("\nStopped by user")

        finally: 
            cleanup() 
        
    
if __name__ == "__main__":
    weights_path = "../trained_weights/weights_reward267.4_20260317_160041.npy"
    controller = SNNController(weights_path)
    controller.run(test_mode=True, num_iterations=50)
