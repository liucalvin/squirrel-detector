from gpiozero import LED
from time import sleep

led = LED(17)  # Set GPIO 17 as an output

while True:
    led.on()
    print("GPIO 17 is ON")
    sleep(1)  # Wait for 1 second

    led.off()
    print("GPIO 17 is OFF")
    sleep(1)  # Wait for 1 second

