from gpiozero import PWMOutputDevice
from time import sleep
import signal
import sys

# Initialize PWM for two motors on GPIO 12 and 13
yaw_motor = PWMOutputDevice(12, frequency=200, initial_value=0)  
pitch_motor = PWMOutputDevice(13, frequency=200, initial_value=0)

# Define motor positions
YAW_MIN = -1  
YAW_MAX = 1   
PITCH_MIN = 0.5
PITCH_MAX = 1

# Define the range for the duty cycle (ensure it's well-calibrated)
MIN_DUTY = 0.05  
MAX_DUTY = 0.11  

# Smooth movement parameters
STEPS = 300  
DELAY = 0.005  

def position_to_duty(position, min_pos, max_pos):
    """Convert position (-1 to 1 or custom range) to PWM duty cycle with clamping."""
    duty = ((position - min_pos) / (max_pos - min_pos)) * (MAX_DUTY - MIN_DUTY) + MIN_DUTY
    return max(MIN_DUTY, min(MAX_DUTY, duty))  # Clamping to prevent jumps

def move_slowly(motor, start, end, steps=STEPS, delay=DELAY):
    """Move motor smoothly between two positions using fine steps."""
    step_size = (end - start) / steps
    for i in range(steps + 1):
        position = start + i * step_size
        duty_cycle = position_to_duty(position, YAW_MIN if motor == yaw_motor else PITCH_MIN, 
                                      YAW_MAX if motor == yaw_motor else PITCH_MAX)
        motor.value = duty_cycle
        sleep(delay)
    sleep(0.1)  # Small pause to ensure no quick resets
    motor.off()  # Stop PWM after motion to prevent jitter

def cleanup(*args):
    """Ensure motors turn off before exiting."""
    print("\nStopping motors...")
    yaw_motor.off()
    pitch_motor.off()
    sys.exit(0)

# Catch exit signals (CTRL+C or script termination)
signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)

# Move motors smoothly between two positions
while True:
    move_slowly(yaw_motor, YAW_MIN, YAW_MAX)
    move_slowly(pitch_motor, PITCH_MIN, PITCH_MAX)
    sleep(1)

    move_slowly(yaw_motor, YAW_MAX, YAW_MIN)
    move_slowly(pitch_motor, PITCH_MAX, PITCH_MIN)
    sleep(1)
