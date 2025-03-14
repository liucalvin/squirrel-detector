from gpiozero import PWMOutputDevice

# Set up GPIO 12 for PWM (e.g., for motor control)
pwm_device = PWMOutputDevice(12, frequency=50)

# Set PWM duty cycle (0 to 1)
pwm_device.value = 0.5  # 50% duty cycle
