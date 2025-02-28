from gpiozero import PWMOutputDevice
from time import sleep
import signal
import sys

# Initialize PWM for two motors on GPIO 12 and 13
yaw_motor = PWMOutputDevice(12, frequency=50, initial_value=0)
pitch_motor = PWMOutputDevice(13, frequency=50, initial_value=0)

# Define motor positions
YAW_MIN = -1  
YAW_MAX = 1   
PITCH_MIN = 0.7
PITCH_MAX = 0.8

# Define the range for the duty cycle
YAW_MIN_DUTY = 0.05  
YAW_MAX_DUTY = 0.11  
YAW_MIN_DUTY = 0.08  
YAW_MAX_DUTY = 0.11  

# Smooth movement parameters
STEPS = 100   # More steps = smoother movement
DELAY = 0.03  # Time between steps

def set_motor_position(motor, position, min_pos, max_pos):
    """Convert position to duty cycle and move motor."""
    duty_cycle = ((position - min_pos) / (max_pos - min_pos)) * (YAW_MAX_DUTY - YAW_MIN_DUTY) + YAW_MIN_DUTY
    motor.value = duty_cycle
    print(f"Motor on GPIO {motor.pin.number}: Position {position:.3f}, Duty {duty_cycle:.5f}")
    
def move_slowly(motor, start, end, steps=STEPS, delay=DELAY):
    """Move motor gradually from start to end position."""
    step_size = (end - start) / steps
    for i in range(steps + 1):
        set_motor_position(motor, start + i * step_size, YAW_MIN if motor == yaw_motor else PITCH_MIN, 
                           YAW_MAX if motor == yaw_motor else PITCH_MAX)
        sleep(delay)
    motor.off()  # Stop PWM to prevent jitter

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
