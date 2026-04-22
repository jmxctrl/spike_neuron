"""
    LeKiwi motor control via Feetech SCS serial bus 
    Sends speed packets over /dev/ttyUSB0 - Waveshare board -> 3 servos 
"""

import serial 
import time 

W1 = 1 # front  
W2 = 2 # left 
W3 = 3 # right 

DRIVE_SPEED = 300 
TURN_SPEED = 200 
STOP = 0 

ADDR_SPEED = 46 

class MotorController: 
    def __init__(self, port='/dev/ttyUSB0', baudrate=1000000):
        """ open serial connection to Waveshare board """
        self.ser = serial.Serial(port, baudrate, timeout=0.1) # connection to waveshare board 
        time.sleep(0.1) # wait for connection to settle 
        print(f"✓ Motors connected on {port}")

    def _send_speed(self, servo_id, speed):
        """ build and send Feetech SCS speed packet to one servo """ 
        if speed < 0:
            speed_val = (1 << 15) | abs(speed)
        else: 
            speed_val = speed 

        speed_low = speed_val & 0xFF # lower byte 
        speed_high = (speed_val >> 8) & 0xFF # upper byte 

        length = 5 # how many bytes to read after length byte itself 
        checksum = (~(servo_id + length + 0x03 + ADDR_SPEED + speed_low + speed_high)) & 0xFF 

        # communication protocol between USB and waveboard 
        # payload -- speed which servo spins 
        packet = bytes([0xFF, 0xFF, servo_id, length, 0x03, ADDR_SPEED, speed_low, speed_high, checksum])
        self.ser.write(packet)


    def execute_action(self, action):
        """ translate SNN output to wheel speeds and send to all 3 servos """
        
        # forward
        if action == 0: 
            self._send_speed(W1,  DRIVE_SPEED)
            self._send_speed(W2,  DRIVE_SPEED)
            self._send_speed(W3,  DRIVE_SPEED)
    
        elif action == -1:  # turn left
            self._send_speed(W1, -TURN_SPEED)
            self._send_speed(W2,  TURN_SPEED)
            self._send_speed(W3, -TURN_SPEED)

        elif action == 1:   # turn right
            self._send_speed(W1,  TURN_SPEED)
            self._send_speed(W2, -TURN_SPEED)
            self._send_speed(W3,  TURN_SPEED)

    
    def stop(self):
        """stop all wheels"""
        self._send_speed(W1, STOP)
        self._send_speed(W2, STOP)
        self._send_speed(W3, STOP)

    def close(self):
        """close serial connection"""
        self.stop()
        self.ser.close()
        print("Motors disconnected")

        
if __name__ == "__main__":
    """Test motors independently"""
    motors = None
    try:
        motors = MotorController(port="/dev/ttyUSB0")

        print("Testing FORWARD...")
        motors.execute_action(0)
        time.sleep(2)

        print("Testing TURN LEFT...")
        motors.execute_action(-1)
        time.sleep(2)

        print("Testing TURN RIGHT...")
        motors.execute_action(1)
        time.sleep(2)

        motors.stop()
        print("Done")

    except KeyboardInterrupt:
        if motors:
            motors.stop()

    finally:
        if motors:
            motors.close()