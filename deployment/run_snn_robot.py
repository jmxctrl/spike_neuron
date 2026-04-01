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
import serial

# Add parent directory to path so we can import SNN modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spike_vectorized import run_vectorized_lif
from light_task import encode_sensors_to_spikes
from neuromodulation import classify_actions
from rasp_sensors import SensorFilter, cleanup

NUM_NEURONS = 100
NUM_STEPS = 10 
SERIAL_PORT = "/dev/ttyACM0"
SERIAL_BAUD = 115200

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

        # connect to microbit via serial
        self.ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=1)
        time.sleep(2)  # wait for connection to stabilize
        print(f"✓ Connected to microbit on {SERIAL_PORT}")

    def send_command(self, cmd):
        """send command to microbit via serial"""
        self.ser.write((cmd + "\n").encode())

    def normalize_sensors(self, distances, max_distance=400):
        """ convert distances to [0, 1]"""
        return [min(d, max_distance) / max_distance for d in distances]

    def action_to_command(self, action):
        if action == -1:
            return "L"
        elif action == 0:
            return "F"
        elif action == 1:
            return "R"
        else:
            return "S"

    def run_inference(self):
        # 1. read all 3 sensors returned from filtered 
        filtered_distances = self.sensor_filter.read_all_sensors_filtered()

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
        command = self.action_to_command(action)

        return action, command


    def run(self, test_mode=True, num_iterations=50):
        """main control loop"""
        print(f"\nRunning in {'TEST' if test_mode else 'FULL'} mode for {num_iterations} iterations\n")

        try: 
            for i in range(num_iterations):
                action, command = self.run_inference()
                print(f"[{i+1}/{num_iterations}] Action: {action} -> {command}")
                self.send_command(command)
                time.sleep(0.2)
        except KeyboardInterrupt:
            print("\nStopped by user")

        finally: 
            self.send_command("S")  # stop the robot
            self.ser.close()
            cleanup()
            print("✓ Serial connection closed") 
        
    
if __name__ == "__main__":
    weights_path = "../trained_weights/weights_reward267.4_20260317_160041.npy"
    controller = SNNController(weights_path)
    controller.run(test_mode=True, num_iterations=50)
