"""
    ultrasonic sensor interface for Raspberry Pi
    Key Elements: 
        distance calculation 
        noise filtering with median value
""" 

import RPi.GPIO as GPIO 
import time 
from collections import deque 
import statistics 

# set GPIO mode 
GPIO.setmode(GPIO.BCM) # BCM numbering 

# define pin numbers for 3 sensors 
LEFT_TRIG = 13
LEFT_ECHO = 19 
CENTER_TRIG = 23
CENTER_ECHO = 24 
RIGHT_TRIG = 20 
RIGHT_ECHO = 21 

# set up GPIO pins 
GPIO.setup(LEFT_TRIG, GPIO.OUT)
GPIO.setup(LEFT_ECHO, GPIO.IN)
GPIO.setup(CENTER_TRIG, GPIO.OUT)
GPIO.setup(CENTER_ECHO, GPIO.IN)
GPIO.setup(RIGHT_TRIG, GPIO.OUT)
GPIO.setup(RIGHT_ECHO, GPIO.IN)

# filter class for smoothing sensor readings by taking median value 
class SensorFilter: 
    """
        Median value of all output values
        window size configurable    
    """

    def __init__(self, window_size=5):
        """ create storage for each sensor""" 
        # 3 buffers, one per each sensor 
        # each buffer holds 5 readings 
        self.left_buffer = deque(maxlen=window_size)
        self.center_buffer = deque(maxlen=window_size)
        self.right_buffer = deque(maxlen=window_size)


    def update(self, left, center, right):
        """add new readings, return filtered value"""
        # add new readings to buffer 
        self.left_buffer.append(left)
        self.center_buffer.append(center)
        self.right_buffer.append(right)

        # calculate and return median value 
        return [
            statistics.median(self.left_buffer),
            statistics.median(self.center_buffer),
            statistics.median(self.right_buffer)
        ]

    def read_all_sensors_filtered(self):
        """read all sensors and return median filtered values""" 
        raw = read_all_sensors()
        return self.update(raw[0], raw[1], raw[2])

# read_distance() function t * speed / 2
def read_distance(trig_pin, echo_pin, timeout=0.03):
    # send trigger pulse 
    GPIO.output(trig_pin, True)
    time.sleep(0.00001)
    GPIO.output(trig_pin, False)

    # Initialize variables BEFORE loops to prevent undefined variable crashes
    pulse_start = time.time()
    pulse_end = time.time()

    # wait for echo start 
    start_wait = time.time()
    while GPIO.input(echo_pin) == 0:
        if time.time() - start_wait > timeout:
            return 400 # max range 
        pulse_start = time.time()
    
    # wait for echo end (reset timeout timer for independent timing)
    start_wait = time.time()
    while GPIO.input(echo_pin) == 1:
        if time.time() - start_wait > timeout:
            return 400 # max range 
        pulse_end = time.time()

    # calculate distance 
    duration = pulse_end - pulse_start 
    distance = duration * 17150 

    # return value distance 
    return round(distance, 1)

# read_all_sensors function 
def read_all_sensors():
    # Read sensors with delays to prevent ultrasonic interference
    left = read_distance(LEFT_TRIG, LEFT_ECHO)
    time.sleep(0.01)  # 10ms settling time
    
    center = read_distance(CENTER_TRIG, CENTER_ECHO)
    time.sleep(0.01)  # 10ms settling time
    
    right = read_distance(RIGHT_TRIG, RIGHT_ECHO)

    return [left, center, right] # SNN expects continuous floats  

def cleanup():
    """Call this when shutting down"""
    GPIO.cleanup()


if __name__ == "__main__":
    """Test sensors independently"""
    try:
        sensor_filter = SensorFilter(window_size=5) # create median filter with window size 5
        while True:
            filtered = sensor_filter.read_all_sensors_filtered()
            
            # Show buffer contents (last 5 readings) and median result
            print(f"Left buffer:   {list(sensor_filter.left_buffer)} → Median: {filtered[0]:.1f}")
            print(f"Center buffer: {list(sensor_filter.center_buffer)} → Median: {filtered[1]:.1f}")
            print(f"Right buffer:  {list(sensor_filter.right_buffer)} → Median: {filtered[2]:.1f}")
            print("-" * 60)
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\nStopping...")
        cleanup()