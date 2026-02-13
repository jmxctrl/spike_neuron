from spike_vectorized import create_weight_matrix, calculate_stdp, train_with_stdp, run_vectorized_lif
import numpy as np
import matplotlib.pyplot as plt

"""
Learning objective: network learns how to self drive 

Input neurons (3): left, center, right lane sensors [0.0, 1.0]
Hidden neurons (94): processing layer 
Output neurons (3): steering commands (left, straight, right)

Task:  
    1. Car starts in lane center (position = 0, far left = -1, far right = +1) 
    2. Every timestep t: 
        - Read 3 sensor values [0.0, 1.0]
        - Convert to spike rates -> inject into input neurons 
        - run SNN forward pass 
        - Read output neuron spikes -> steering command 
        - update car position 
    3. crash if position > threshold 
    4. Reward for staying centered 

STDP learning: 
    Senesors: 
        - left sensor: 0.2 
        - center sensor: 0.5
        - right sensor: 0.9 (close to right edge)
    Encode -> Spike rates: 
        - Neuron 0: fires slowly (0.2 rate)
        - Neuron 1: fires medium (0.4 rate)
        - Neuron 2: fires fast (0.9 rate)

Expected Outcome: turn LEFT 
new position: 0.6 - 0.1 = 0.5 

"""

# ============================================
# STEP 1: Car simulation basics
# ============================================

def update_position(position, action, step_size=0.1): 
    """
    Update car position based on steering action
    
    Args:
        position: current position (-1=left edge, 0=center, +1=right edge)
        action: steering command (-1=left, 0=straight, +1=right)
        step_size: how much to move per action (default 0.1)
    
    Returns:
        new_position: updated position
    """
    new_position = position + (action * step_size)
    return new_position


def is_crashed(position, threshold=1.0): 
    """
    Check if car has left the lane
    
    Args:
        position: current position
        threshold: crash boundary (default ±1.0)
    
    Returns:
        True if crashed, False otherwise
    """
    return abs(position) > threshold

# ============================================
# STEP 2: Get sensors 
# ============================================

def get_sensor_values(position): 
    """
    Calculate sensor readings based on the car position 

    Params:
        position: car position [-1 to +1]

    Returns left center right sensor values 
    """
    distance_to_left = abs(position - (-1.0)) 
    left_sensor = max(0.0, 1.0 - distance_to_left)

    distance_to_center = abs(position - 0.0)
    center_sensor = max(0.0, 1.0 - distance_to_center)

    distance_to_right = abs(position - 1.0)
    right_sensor = max(0.0, 1.0 - distance_to_right)

    return [left_sensor, center_sensor, right_sensor]

# Test Step 1
if __name__ == "__main__":
    print("=== STEP 1: Car Simulation Test ===")
    
    position = 0.0  # Start centered
    print(f"Start position: {position:.2f}")
    
    # Simulate a few actions
    position = update_position(position, action=1)  # Turn right
    print(f"After turn right: {position:.2f}")
    
    position = update_position(position, action=1)  # Turn right again
    print(f"After turn right again: {position:.2f}")
    
    position = update_position(position, action=-1)  # Turn left
    print(f"After turn left: {position:.2f}")
    
    # Check crash detection
    print(f"\nPosition 1.5 - Crashed: {is_crashed(1.5)}")
    print(f"Position 0.5 - Crashed: {is_crashed(0.5)}")
    
    print("\n=== STEP 2: Sensor Test ===")
    
    # Test sensors at different positions
    positions_test = [0.0, 0.6, -0.6, 0.9, -0.9]
    for pos in positions_test:
        sensors = get_sensor_values(pos)
        print(f"Position {pos:5.1f} -> Left: {sensors[0]:.2f}, Center: {sensors[1]:.2f}, Right: {sensors[2]:.2f}")
    
    # Graph the sensor responses
    print("\n=== Graphing Sensor Responses ===")
    positions = np.linspace(-1.5, 1.5, 100)  # Car positions from -1.5 to 1.5
    
    left_values = []
    center_values = []
    right_values = []
    
    for pos in positions:
        sensors = get_sensor_values(pos)
        left_values.append(sensors[0])
        center_values.append(sensors[1])
        right_values.append(sensors[2])
    
    plt.figure(figsize=(10, 6))
    plt.plot(positions, left_values, label='Left Sensor', color='blue', linewidth=2)
    plt.plot(positions, center_values, label='Center Sensor', color='green', linewidth=2)
    plt.plot(positions, right_values, label='Right Sensor', color='red', linewidth=2)
    
    # Mark boundaries
    plt.axvline(x=-1.0, color='black', linestyle='--', alpha=0.3, label='Lane Edges')
    plt.axvline(x=1.0, color='black', linestyle='--', alpha=0.3)
    plt.axvline(x=0.0, color='gray', linestyle=':', alpha=0.5, label='Center')
    
    plt.xlabel('Car Position', fontsize=12)
    plt.ylabel('Sensor Value', fontsize=12)
    plt.title('Sensor Response vs Car Position', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(-1.5, 1.5)
    plt.ylim(-0.1, 1.1)
    
    plt.show()
        sensors = get_sensor_values(pos)
        print(f"Position {pos:5.1f} -> Left: {sensors[0]:.2f}, Center: {sensors[1]:.2f}, Right: {sensors[2]:.2f}")




