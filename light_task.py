import numpy as np

from rate_encoder import rate_encode_vector

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

class CarState:
    """
    Car state with position, velocity, and history.
    """
    def __init__(self, position=0.0, speed=0.1):
        self.position = position
        self.speed = speed  # Forward speed (constant for now)
        self.prev_position = position
        
    def update(self, action, step_size=0.1):
        """Update position based on steering action."""
        self.prev_position = self.position
        self.position = self.position + (action * step_size)
        
    def is_crashed(self, threshold=1.0):
        """Check if car has left the lane."""
        return abs(self.position) > threshold
    
    def get_centering(self):
        """Get distance from lane center (normalized to [0, 1])."""
        return abs(self.position)  # 0 = centered, 1 = at edge


def update_position(position, action, step_size=0.1): 
    """
    Update car position based on steering action (legacy function)
    
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
    Check if car has left the lane (legacy function)
    
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

def get_sensor_values(car_state, enhanced=True): 
    """
    Calculate sensor readings based on car state.

    Args:
        car_state: CarState object with position, speed, etc.
        enhanced: If True, return 5 sensors; if False, return 3 sensors (basic)

    Returns:
        List of sensor values [left, center, right] or 
        [left, center, right, speed, centering]
        All normalized to [0, 1] range
    """
    position = car_state.position if hasattr(car_state, 'position') else car_state
    
    # Distance sensors
    distance_to_left = abs(position - (-1.0))
    left_sensor = max(0.0, 1.0 - distance_to_left)
    
    distance_to_center = abs(position - 0.0)
    center_sensor = max(0.0, 1.0 - distance_to_center)
    
    distance_to_right = abs(position - 1.0)
    right_sensor = max(0.0, 1.0 - distance_to_right)
    
    # Basic mode: return only 3 sensors
    if not enhanced or not hasattr(car_state, 'speed'):
        return [left_sensor, center_sensor, right_sensor]
    
    # Enhanced mode: add speed and centering
    speed_sensor = min(1.0, car_state.speed / 0.2)
    centering_sensor = car_state.get_centering()
    
    return [left_sensor, center_sensor, right_sensor, speed_sensor, centering_sensor]


# ============================================
# STEP 3: Encode spike values 
# ============================================

def encode_sensors_to_spikes(sensor_values, T=100, p_max=0.2):
    """
    Convert sensors into spikes using rate encoding.

    Params:
        sensor_values: list of sensor values (3 or 5 values)
        T: number of timesteps 
        p_max: maximum spike probability 

    Returns:
        spike matrix of shape (num_sensors, T)
    """
    spikes = rate_encode_vector(sensor_values, T=T, p_max=p_max)
    return spikes 
    

# Test Step 1
if __name__ == "__main__":
    print("=== STEP 1: Car Simulation Test ===")
    
    # Test with new CarState class
    car = CarState(position=0.0, speed=0.1)
    print(f"Start position: {car.position:.2f}, speed: {car.speed:.2f}")
    
    # Simulate a few actions
    car.update(action=1, step_size=0.1)  # Turn right
    print(f"After turn right: {car.position:.2f}")
    
    car.update(action=1, step_size=0.1)  # Turn right again
    print(f"After turn right again: {car.position:.2f}")
    
    car.update(action=-1, step_size=0.1)  # Turn left
    print(f"After turn left: {car.position:.2f}")
    
    # Check crash detection
    print(f"\nCar at {car.position:.2f} - Crashed: {car.is_crashed()}")
    car.position = 1.5
    print(f"Car at {car.position:.2f} - Crashed: {car.is_crashed()}")
    
    print("\n=== STEP 2: Sensor Test ===")
    
    # Test sensors at different positions
    positions = [0.0, 0.6, -0.6, 0.9, -0.9]
    
    print("\n-- Basic Sensors (3 values) --")
    for pos in positions:
        sensors = get_sensor_values(pos, enhanced=False)
        print(f"Position {pos:5.1f} -> Left: {sensors[0]:.2f}, Center: {sensors[1]:.2f}, Right: {sensors[2]:.2f}")
    
    print("\n-- Enhanced Sensors (5 values) --")
    for pos in positions:
        car.position = pos
        sensors = get_sensor_values(car, enhanced=True)
        print(f"Position {pos:5.1f} -> Left: {sensors[0]:.2f}, Center: {sensors[1]:.2f}, "
              f"Right: {sensors[2]:.2f}, Speed: {sensors[3]:.2f}, Centering: {sensors[4]:.2f}")
    
    print("\n=== STEP 3: Spike Encoding Test ===")
    
    # Test encoding 5 sensors to spikes
    car.position = 0.6
    sensors = get_sensor_values(car, enhanced=True)
    print(f"\nPosition: {car.position}")
    print(f"Sensor values: Left={sensors[0]:.2f}, Center={sensors[1]:.2f}, Right={sensors[2]:.2f}, "
          f"Speed={sensors[3]:.2f}, Centering={sensors[4]:.2f}")
    
    spikes = encode_sensors_to_spikes(sensors, T=100, p_max=0.2)
    print(f"\nSpike matrix shape: {spikes.shape}")
    print(f"Spike counts: Left={spikes[0].sum()}, Center={spikes[1].sum()}, Right={spikes[2].sum()}, "
          f"Speed={spikes[3].sum()}, Centering={spikes[4].sum()}")
    print(f"Spike rates: Left={spikes[0].sum()/100:.2f}, Center={spikes[1].sum()/100:.2f}, "
          f"Right={spikes[2].sum()/100:.2f}, Speed={spikes[3].sum()/100:.2f}, Centering={spikes[4].sum()/100:.2f}")
    print(f"\nFirst 20 timesteps of spike matrix (5 sensors):")
    print(spikes[:, :20])




