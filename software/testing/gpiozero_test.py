# from gpiozero import AngularServo
# from gpiozero.pins.lgpio import LGPIOFactory
# import time

# # GPIO setup
# servo_pin = 12  # GPIO pin connected to the servo

# # Use lgpio as the backend for gpiozero
# factory = LGPIOFactory()

# # Initialize the servo
# servo = AngularServo(servo_pin, min_angle=0, max_angle=180, pin_factory=factory)

# # Define the panning duration (5 seconds)
# pan_duration = 5  # Time to move from one end to the other

# try:
#     while True:
#         # Move the servo from 0 to 180 degrees
#         print("Moving from 0 to 180 degrees...")
#         start_time = time.time()
#         while time.time() - start_time < pan_duration:
#             # Calculate the current angle based on elapsed time
#             elapsed_time = time.time() - start_time
#             angle = 180 * (elapsed_time / pan_duration)
#             servo.angle = angle
#             time.sleep(0.01)  # Small delay for smooth movement

#         # Move the servo from 180 to 0 degrees
#         print("Moving from 180 to 0 degrees...")
#         start_time = time.time()
#         while time.time() - start_time < pan_duration:
#             # Calculate the current angle based on elapsed time
#             elapsed_time = time.time() - start_time
#             angle = 180 - (180 * (elapsed_time / pan_duration))
#             servo.angle = angle
#             time.sleep(0.01)  # Small delay for smooth movement

# except KeyboardInterrupt:
#     # Cleanup on Ctrl+C
#     print("\nExiting...")
#     servo.angle = None  # Detach the servo

from gpiozero import Servo
from time import sleep
import numpy as np  # Import numpy for floating point range

servo = Servo(12)

servo.min()
sleep(3)
servo.max()
sleep(3)
servo.min()


