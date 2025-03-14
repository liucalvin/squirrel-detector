from rpi_hardware_pwm import HardwarePWM
import time

# Set up hardware PWM on channel 0 (GPIO 18, which is pin 12 on the header)
# Channel 0 corresponds to PWM0, which is available on GPIO 18
pwm = HardwarePWM(pwm_channel=1, hz=50, chip=2)  # 50 Hz frequency for servos

# Start PWM with a duty cycle of 0%
pwm.start(0)

def set_servo_angle(angle):
    # Convert angle to duty cycle (0.5ms to 2.5ms pulse width for 0-180 degrees)
    # Duty cycle is calculated as (pulse_width / period) * 100
    # For 50 Hz, period = 20ms
    pulse_width = (angle / 180) * 2 + 0.5  # 0.5ms to 2.5ms
    duty_cycle = (pulse_width / 20) * 100  # Convert to percentage
    pwm.change_duty_cycle(duty_cycle)

try:
    while True:
        # Move the servo to 0 degrees
        set_servo_angle(0)
        time.sleep(1)

        # Move the servo to 90 degrees
        set_servo_angle(90)
        time.sleep(1)

        # Move the servo to 180 degrees
        set_servo_angle(180)
        time.sleep(1)

except KeyboardInterrupt:
    # Stop PWM and clean up on exit
    pwm.stop()
