import RPi.GPIO as GPIO
from time import sleep

# GPIO setup
servo_pin = 13
GPIO.setmode(GPIO.BCM)
GPIO.setup(servo_pin, GPIO.OUT)

# PWM setup
pwm = GPIO.PWM(servo_pin, 50)  # 50 Hz (standard for servos)
pwm.start(0)

def set_angle(angle):
    duty = angle / 18 + 2
    GPIO.output(servo_pin, True)
    pwm.ChangeDutyCycle(duty)
    sleep(1)
    GPIO.output(servo_pin, False)
    pwm.Change
