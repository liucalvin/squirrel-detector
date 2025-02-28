from gpiozero import PWMOutputDevice
from time import sleep

# Initialize PWM for two motors on GPIO 12 and 13
pitch_motor = PWMOutputDevice(13)
yaw_motor = PWMOutputDevice(12)

# Define motor positions (adjust these if needed)
YAW_MIN = -1  # One position (e.g., left)
YAW_MAX = 1   # Another position (e.g., right)
PITCH_MIN = 0.4
PITCH_MAX = 1.5

# Define the range for the duty cycle (adjust for your motor type)
MIN_DUTY = 0.05
MAX_DUTY = 0.11

def set_motor_position(motor, position):
    """Move motor smoothly to a given position (-1 to 1)."""
    duty_cycle = MIN_DUTY + (position + 1) * (MAX_DUTY - MIN_DUTY) / 2
    motor.value = duty_cycle
    print(f"Motor on GPIO {motor.pin.number}: Position {position:.3f}, Duty {duty_cycle:.5f}")

from time import sleep

def move_slowly(motor, start, end, step_size=0.05, delay=0.1):
    """Move motor gradually from start to end position using a fixed step size."""
    steps = max(1, int(abs(end - start) / step_size))  # Ensure at least one step
    direction = 1 if end > start else -1  # Determine movement direction
    
    for i in range(steps + 1):
        position = start + i * step_size * direction
        if (direction == 1 and position > end) or (direction == -1 and position < end):
            position = end  # Ensure we don't overshoot the target
        set_motor_position(motor, position)
        sleep(delay)
    
    set_motor_position(motor, end)  # Ensure final position is reached


while True:
    # Move motors from A to B
    move_slowly(pitch_motor, PITCH_MIN, PITCH_MAX)
    move_slowly(yaw_motor, YAW_MIN, YAW_MAX)
    sleep(0.2)  # Hold position briefly

    # Move motors from B to A
    move_slowly(pitch_motor, PITCH_MAX, PITCH_MIN)
    move_slowly(yaw_motor, YAW_MAX, YAW_MIN)
    sleep(0.2)  # Hold position briefly
