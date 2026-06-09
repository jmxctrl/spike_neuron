"""
Yahboom Raspbot V1 Control Library

Based on the original YB_Pcb_Car.py from Yahboom.
Motors and servos use I2C (address 0x16).
Sensors, buzzer, and LEDs use GPIO.

GPIO Pin Mappings (BOARD mode -> BCM conversion):
- Buzzer: BOARD 32 = BCM 12 (PWM)
- LED1 (Red): BOARD 40 = BCM 21
- LED2 (Blue): BOARD 38 = BCM 20
- Ultrasonic: Trig BOARD 16 = BCM 23, Echo BOARD 18 = BCM 24
- Line Trackers: L1 BOARD 13 = BCM 27, L2 BOARD 15 = BCM 22, 
                 R1 BOARD 11 = BCM 17, R2 BOARD 7 = BCM 4
- IR Obstacle: L BOARD 21 = BCM 9, R BOARD 19 = BCM 10, Enable BOARD 22 = BCM 25
"""

import time
from typing import Tuple
from dataclasses import dataclass

# Try to import gpiozero (works on all Pi versions including Pi 5)
GPIO_AVAILABLE = False
try:
    from gpiozero import OutputDevice, InputDevice, PWMOutputDevice, DistanceSensor
    from gpiozero.pins.lgpio import LGPIOFactory
    from gpiozero import Device
    # Try to use lgpio backend for Pi 5 compatibility
    try:
        Device.pin_factory = LGPIOFactory()
    except:
        pass  # Fall back to default
    GPIO_AVAILABLE = True
except ImportError:
    try:
        from gpiozero import OutputDevice, InputDevice, PWMOutputDevice, DistanceSensor
        GPIO_AVAILABLE = True
    except ImportError:
        OutputDevice = None
        InputDevice = None
        PWMOutputDevice = None
        DistanceSensor = None

# Try to import smbus
smbus = None
SMBUS_AVAILABLE = False
try:
    import smbus
    SMBUS_AVAILABLE = True
except ImportError:
    try:
        import smbus2 as smbus
        SMBUS_AVAILABLE = True
    except ImportError:
        pass

RPI_AVAILABLE = GPIO_AVAILABLE or SMBUS_AVAILABLE

if not RPI_AVAILABLE:
    print("Warning: Neither gpiozero nor smbus available. Running in simulation mode.")


# BOARD to BCM pin mapping
def board_to_bcm(board_pin: int) -> int:
    """Convert BOARD pin number to BCM GPIO number."""
    mapping = {
        7: 4, 11: 17, 12: 18, 13: 27, 15: 22, 16: 23, 18: 24,
        19: 10, 21: 9, 22: 25, 23: 11, 24: 8, 26: 7,
        29: 5, 31: 6, 32: 12, 33: 13, 35: 19, 36: 16,
        37: 26, 38: 20, 40: 21
    }
    return mapping.get(board_pin, board_pin)


@dataclass
class LineTrackerReading:
    """Line tracker sensor readings (True = line detected)."""
    left1: bool
    left2: bool
    right1: bool
    right2: bool
    
    @property
    def all_on_line(self) -> bool:
        return all([self.left1, self.left2, self.right1, self.right2])
    
    @property
    def centered(self) -> bool:
        return self.left2 and self.right1 and not self.left1 and not self.right2


class Raspbot:
    """
    Yahboom Raspbot V1 Control Class.
    Based on original YB_Pcb_Car implementation.
    """
    
    # I2C Configuration
    I2C_ADDRESS = 0x16
    I2C_BUS = 1
    
    # I2C Registers
    REG_MOTOR = 0x01
    REG_STOP = 0x02
    REG_SERVO = 0x03
    
    # GPIO Pins (BCM numbering for gpiozero)
    PIN_BUZZER = board_to_bcm(32)        # BCM 12
    PIN_LED_RED = board_to_bcm(40)       # BCM 21
    PIN_LED_BLUE = board_to_bcm(38)      # BCM 20
    PIN_ULTRASONIC_TRIG = board_to_bcm(16)  # BCM 23
    PIN_ULTRASONIC_ECHO = board_to_bcm(18)  # BCM 24
    PIN_TRACKING_LEFT1 = board_to_bcm(13)   # BCM 27
    PIN_TRACKING_LEFT2 = board_to_bcm(15)   # BCM 22
    PIN_TRACKING_RIGHT1 = board_to_bcm(11)  # BCM 17
    PIN_TRACKING_RIGHT2 = board_to_bcm(7)   # BCM 4
    PIN_IR_AVOID_LEFT = board_to_bcm(21)    # BCM 9
    PIN_IR_AVOID_RIGHT = board_to_bcm(19)   # BCM 10
    PIN_IR_AVOID_ENABLE = board_to_bcm(22)  # BCM 25
    
    def __init__(self):
        """Initialize the Raspbot controller."""
        self._device = None
        self._addr = self.I2C_ADDRESS
        self._initialized = False
        
        # GPIO devices (gpiozero)
        self._buzzer = None
        self._led_red = None
        self._led_blue = None
        self._ultrasonic_trig = None
        self._ultrasonic_echo = None
        self._tracker_l1 = None
        self._tracker_l2 = None
        self._tracker_r1 = None
        self._tracker_r2 = None
        self._ir_left = None
        self._ir_right = None
        self._ir_enable = None
        
    def __enter__(self):
        """Context manager entry."""
        self.init()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
        return False
    
    def init(self):
        """Initialize I2C and GPIO."""
        if not RPI_AVAILABLE:
            print("Simulation mode: Hardware initialization skipped")
            self._initialized = True
            return
        
        # Initialize I2C
        if SMBUS_AVAILABLE:
            try:
                self._device = smbus.SMBus(self.I2C_BUS)
                print("I2C initialized successfully")
            except Exception as e:
                print(f"Warning: I2C initialization failed: {e}")
                print("  Enable I2C with: sudo raspi-config -> Interface Options -> I2C")
                self._device = None
        
        # Initialize GPIO with gpiozero
        if not GPIO_AVAILABLE:
            print("Warning: gpiozero not available - GPIO features disabled")
            self._initialized = True
            return
        
        # Setup buzzer (PWM)
        try:
            self._buzzer = PWMOutputDevice(self.PIN_BUZZER, frequency=440)
            self._buzzer.value = 0
        except Exception as e:
            print(f"Warning: Buzzer init failed: {e}")
        
        # Setup LEDs
        try:
            self._led_red = OutputDevice(self.PIN_LED_RED, initial_value=False)
        except Exception as e:
            print(f"Warning: LED Red init failed: {e}")
        
        try:
            self._led_blue = OutputDevice(self.PIN_LED_BLUE, initial_value=False)
        except Exception as e:
            print(f"Warning: LED Blue init failed: {e}")
        
        # Setup ultrasonic
        try:
            self._ultrasonic_trig = OutputDevice(self.PIN_ULTRASONIC_TRIG)
            self._ultrasonic_echo = InputDevice(self.PIN_ULTRASONIC_ECHO)
        except Exception as e:
            print(f"Warning: Ultrasonic init failed: {e}")
        
        # Setup tracking sensors
        try:
            self._tracker_l1 = InputDevice(self.PIN_TRACKING_LEFT1)
            self._tracker_l2 = InputDevice(self.PIN_TRACKING_LEFT2)
            self._tracker_r1 = InputDevice(self.PIN_TRACKING_RIGHT1)
            self._tracker_r2 = InputDevice(self.PIN_TRACKING_RIGHT2)
        except Exception as e:
            print(f"Warning: Line trackers init failed: {e}")
        
        # Setup IR obstacle sensors
        try:
            self._ir_left = InputDevice(self.PIN_IR_AVOID_LEFT)
            self._ir_right = InputDevice(self.PIN_IR_AVOID_RIGHT)
            self._ir_enable = OutputDevice(self.PIN_IR_AVOID_ENABLE)
            self._ir_enable.on()
        except Exception as e:
            print(f"Warning: IR sensors init failed: {e}")
        
        print("GPIO initialized")
        
        self._initialized = True
        
    def cleanup(self):
        """Clean up resources."""
        if self._initialized:
            self.stop()
        
        # Close gpiozero devices
        for dev in [self._buzzer, self._led_red, self._led_blue,
                    self._ultrasonic_trig, self._ultrasonic_echo,
                    self._tracker_l1, self._tracker_l2, self._tracker_r1, self._tracker_r2,
                    self._ir_left, self._ir_right, self._ir_enable]:
            if dev:
                try:
                    dev.close()
                except:
                    pass
        
        if self._device:
            try:
                self._device.close()
            except:
                pass
        
        self._initialized = False
    
    # ========== I2C Helper Methods ==========
    
    def _write_u8(self, reg: int, data: int):
        """Write single byte to I2C register."""
        if self._device is None:
            return
        try:
            self._device.write_byte_data(self._addr, reg, data)
        except Exception as e:
            print(f"I2C write_u8 error: {e}")
    
    def _write_array(self, reg: int, data: list):
        """Write array to I2C register."""
        if self._device is None:
            return
        try:
            self._device.write_i2c_block_data(self._addr, reg, data)
        except Exception as e:
            print(f"I2C write_array error: {e}")
    
    # ========== Motor Control ==========
    
    def _ctrl_car(self, l_dir: int, l_speed: int, r_dir: int, r_speed: int):
        """Low-level motor control matching original YB_Pcb_Car.Ctrl_Car()."""
        data = [l_dir, l_speed, r_dir, r_speed]
        self._write_array(self.REG_MOTOR, data)
    
    def forward(self, speed: int = 100):
        """Move forward."""
        speed = max(0, min(255, speed))
        self._ctrl_car(1, speed, 1, speed)
    
    def backward(self, speed: int = 100):
        """Move backward."""
        speed = max(0, min(255, speed))
        self._ctrl_car(0, speed, 0, speed)
    
    def turn_left(self, speed: int = 100):
        """Turn/spin left."""
        speed = max(0, min(255, speed))
        self._ctrl_car(0, speed, 1, speed)
    
    def turn_right(self, speed: int = 100):
        """Turn/spin right."""
        speed = max(0, min(255, speed))
        self._ctrl_car(1, speed, 0, speed)
    
    def stop(self):
        """Stop all motors."""
        self._write_u8(self.REG_STOP, 0x00)
    
    def control_car(self, left_speed: int, right_speed: int):
        """Control car with signed speeds. Positive = forward, negative = backward."""
        l_dir = 1 if left_speed >= 0 else 0
        r_dir = 1 if right_speed >= 0 else 0
        self._ctrl_car(l_dir, int(abs(left_speed)), r_dir, int(abs(right_speed)))
    
    # ========== Servo Control ==========
    
    def set_servo(self, servo_id: int, angle: int):
        """Control servo angle. servo_id: 1=pan, 2=tilt. angle: 0-180."""
        angle = max(0, min(180, angle))
        self._write_array(self.REG_SERVO, [servo_id, angle])
    
    def set_pan(self, angle: int):
        """Set pan servo (horizontal)."""
        self.set_servo(2, angle)  # Servo 2 = pan (horizontal)
    
    def set_tilt(self, angle: int):
        """Set tilt servo (vertical)."""
        self.set_servo(1, angle)  # Servo 1 = tilt (vertical)
    
    def center_servos(self):
        """Center both servos."""
        self.set_pan(90)
        self.set_tilt(90)
    
    # ========== Ultrasonic Sensor ==========
    
    def read_distance(self) -> float:
        """Read distance from ultrasonic sensor. Returns cm or -1 on error."""
        if not GPIO_AVAILABLE or self._ultrasonic_trig is None:
            return 50.0  # Simulation
        
        try:
            self._ultrasonic_trig.off()
            time.sleep(0.000002)
            self._ultrasonic_trig.on()
            time.sleep(0.000015)
            self._ultrasonic_trig.off()
            
            t3 = time.time()
            while not self._ultrasonic_echo.is_active:
                if time.time() - t3 > 0.03:
                    return -1
            
            t1 = time.time()
            while self._ultrasonic_echo.is_active:
                if time.time() - t1 > 0.03:
                    return -1
            
            t2 = time.time()
            distance = ((t2 - t1) * 340 / 2) * 100
            return round(distance, 1)
            
        except Exception as e:
            print(f"Ultrasonic error: {e}")
            return -1
    
    def read_distance_filtered(self, samples: int = 5) -> float:
        """Read filtered distance (average of multiple samples)."""
        readings = []
        for _ in range(samples):
            d = self.read_distance()
            if 0 < d < 500:
                readings.append(d)
            time.sleep(0.01)
        
        if len(readings) >= 3:
            readings.sort()
            return sum(readings[1:-1]) / (len(readings) - 2)
        elif readings:
            return sum(readings) / len(readings)
        return -1
    
    # ========== Line Tracking Sensors ==========
    
    def read_line_tracker(self) -> LineTrackerReading:
        """Read all 4 line tracker sensors. True = line detected."""
        if not GPIO_AVAILABLE:
            return LineTrackerReading(False, False, False, False)
        
        return LineTrackerReading(
            left1=not self._tracker_l1.is_active if self._tracker_l1 else False,
            left2=not self._tracker_l2.is_active if self._tracker_l2 else False,
            right1=not self._tracker_r1.is_active if self._tracker_r1 else False,
            right2=not self._tracker_r2.is_active if self._tracker_r2 else False
        )
    
    def read_line_tracker_raw(self) -> Tuple[int, int, int, int]:
        """Read raw values from line trackers."""
        if not GPIO_AVAILABLE:
            return (1, 1, 1, 1)
        
        return (
            1 if self._tracker_l1 and self._tracker_l1.is_active else 0,
            1 if self._tracker_l2 and self._tracker_l2.is_active else 0,
            1 if self._tracker_r1 and self._tracker_r1.is_active else 0,
            1 if self._tracker_r2 and self._tracker_r2.is_active else 0
        )
    
    # ========== IR Obstacle Sensors ==========
    
    def read_ir_obstacle(self) -> Tuple[bool, bool]:
        """Read IR obstacle sensors. True = obstacle detected."""
        if not GPIO_AVAILABLE:
            return (False, False)
        
        return (
            not self._ir_left.is_active if self._ir_left else False,
            not self._ir_right.is_active if self._ir_right else False
        )
    
    def enable_ir_sensors(self, enable: bool = True):
        """Enable or disable IR obstacle sensors."""
        if self._ir_enable:
            if enable:
                self._ir_enable.on()
            else:
                self._ir_enable.off()
    
    # ========== Buzzer Control ==========
    
    def buzzer_on(self, frequency: int = 440):
        """Turn buzzer on at specified frequency."""
        if self._buzzer is None:
            return
        try:
            # Recreate PWM with new frequency for tone control
            self._buzzer.close()
            self._buzzer = PWMOutputDevice(self.PIN_BUZZER, frequency=frequency)
            self._buzzer.value = 0.5  # 50% duty cycle
        except Exception as e:
            print(f"Buzzer error: {e}")
    
    def buzzer_off(self):
        """Turn buzzer off."""
        if self._buzzer is None:
            return
        try:
            self._buzzer.value = 0
        except:
            pass
    
    def beep(self, duration: float = 0.1, frequency: int = 440):
        """Play a short beep."""
        if self._buzzer is None:
            return
        try:
            # Recreate with desired frequency
            self._buzzer.close()
            self._buzzer = PWMOutputDevice(self.PIN_BUZZER, frequency=frequency)
            self._buzzer.value = 0.5
            time.sleep(duration)
            self._buzzer.value = 0
        except Exception as e:
            print(f"Beep error: {e}")
    
    # ========== LED Control ==========
    
    def led_red(self, on: bool = True):
        """Control red LED."""
        if self._led_red is None:
            return
        if on:
            self._led_red.on()
        else:
            self._led_red.off()
    
    def led_blue(self, on: bool = True):
        """Control blue LED."""
        if self._led_blue is None:
            return
        if on:
            self._led_blue.on()
        else:
            self._led_blue.off()
    
    def leds_on(self):
        """Turn on both LEDs."""
        self.led_red(True)
        self.led_blue(True)
    
    def leds_off(self):
        """Turn off both LEDs."""
        self.led_red(False)
        self.led_blue(False)


def create_robot() -> Raspbot:
    """Create and initialize a Raspbot instance."""
    robot = Raspbot()
    robot.init()
    return robot


if __name__ == "__main__":
    print("Yahboom Raspbot V1 Control Library")
    print("===================================")
    print(f"GPIO available: {GPIO_AVAILABLE}")
    print(f"I2C available: {SMBUS_AVAILABLE}")
