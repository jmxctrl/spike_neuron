"""motor control translates discrete to motor commands"""

import RPi.GPIO as GPIO 

class MotorController:
    def __init__(self):

        pass 

    def execute_action(self, action):
        # converts -1/0/1 to motor commands 

        if action == -1: # left 
            self.set_motors(left_speed=3-, right_speed=70)