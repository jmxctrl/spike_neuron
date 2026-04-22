"""
    ultrasonic sensor interface for Raspberry Pi
    Key Elements: 
        distance calculation 
        noise filtering with median value
""" 

import cv2 
import time 
from collections import deque 
import statistics 

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

        # open LeKiwi camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    def _read_frame(self):
        ret, frame = self.cap.read()
        if not ret: 
            return 400, 400, 400 # fallback = clear path 

        w = frame.shape[1] # gets the width (320 px)
        third = w // 3 
        left_col   = frame[:, :third] # slice out the left 
        center_col = frame[:, third:2*third] 
        right_col  = frame[:, 2*third:] 

        # converts a column to a distance value 
        def col_to_distance(col):
            gray = cv2.cvtColor(col, cv2.COLOR_BGR2GRAY) # grayscale, obstalces detected based on brightness 
            
            # pixel darker than 60 becomes white 255, brighter than 60 becomes black 0
            _, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV) 

            score = thresh.sum() / 255 / thresh.size # convert to pixel count [0, 1]
            
            # convert to distance 
            return 400 * (1.0 - score)

        return col_to_distance(left_col), col_to_distance(center_col), col_to_distance(right_col)


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
        l, c, r = self._read_frame()
        return self.update(l, c, r)

# read_all_sensors function 
def read_all_sensors():
   
    temp = SensorFilter(window_size=1) # temporary window to open camera internally 
    
    # one frame, 
    #split into 3 columns
    #scores each
    # return [left, center, rigtht] distances
    result = temp.read_all_sensors_filtered() 

    temp.cap.release() # close camera after reading,  one-shot function 
    return result # pass 3 distances value to read_all_sensors()

def cleanup():
    pass


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