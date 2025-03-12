import lgpio
import time

# GPIO setup
servo_pin = 13  # GPIO pin connected to the servo

# Open the GPIO chip
gpio_chip = lgpio.gpiochip_open(0)
if gpio_chip < 0:
    raise RuntimeError("Failed to open GPIO chip")

# Claim the servo pin as an output
lgpio.gpio_claim_output(gpio_chip, servo_pin)

# Define the panning duration (5 seconds)
pan_duration = 5  # Time to move from one end to the other

# Function to set servo angle
def set_servo_angle(angle):
    # Clamp the angle to be within 0 to 180 degrees
    angle = max(0, min(180, angle))
    
    # Convert angle to duty cycle (500-2500 microseconds)
    duty_cycle = 500 + (angle / 180) * 2000
    duty_cycle = int(duty_cycle)
    
    # Ensure duty_cycle is within the acceptable range
    if duty_cycle < 500 or duty_cycle > 2500:
        raise ValueError(f"Invalid duty cycle: {duty_cycle}")
    
    print(f"{gpio_chip} {servo_pin} {duty_cycle}")
    lgpio.tx_pwm(gpio_chip, servo_pin, 50, duty_cycle)  # 50 Hz, duty_cycle in microseconds

try:
    while True:
        # Move the servo from 0 to 180 degrees
        print("Moving from 0 to 180 degrees...")
        start_time = time.time()
        while time.time() - start_time < pan_duration:
            # Calculate the current angle based on elapsed time
            elapsed_time = time.time() - start_time
            angle = 180 * (elapsed_time / pan_duration)
            set_servo_angle(angle)
            time.sleep(0.01)  # Small delay for smooth movement

        # Move the servo from 180 to 0 degrees
        print("Moving from 180 to 0 degrees...")
        start_time = time.time()
        while time.time() - start_time < pan_duration:
            # Calculate the current angle based on elapsed time
            elapsed_time = time.time() - start_time
            angle = 180 - (180 * (elapsed_time / pan_duration))
            set_servo_angle(angle)
            time.sleep(0.01)  # Small delay for smooth movement

except KeyboardInterrupt:
    # Cleanup on Ctrl+C
    print("\nExiting...")
    lgpio.tx_pwm(gpio_chip, servo_pin, 50, 0)  # Stop the PWM signal
    lgpio.gpiochip_close(gpio_chip)  # Close the GPIO chip
