#!/usr/bin/python3
import os
import time

class pi5RC:

    def __init__(self, Pin):
        pins = [12, 13, 14, 15, 18, 19]
        afunc = ['a0', 'a0', 'a0', 'a0', 'a3', 'a3']
        self.pwmx = [0, 1, 2, 3, 2, 3]
        self.enableFlag = False

        if Pin in pins:
            self.pin = Pin
            self.pinIdx = pins.index(Pin)

            # Set pin function using pinctrl
            os.system(f"/usr/bin/pinctrl set {self.pin} {afunc[self.pinIdx]}")

            # Export the PWM channel
            pwm_path = f"/sys/class/pwm/pwmchip0/pwm{self.pwmx[self.pinIdx]}"
            if not os.path.exists(pwm_path):
                os.system(f"echo {self.pwmx[self.pinIdx]} > /sys/class/pwm/pwmchip0/export")
                time.sleep(0.2)  # Wait for the PWM channel to be exported

            # Set PWM period (20ms for servos)
            os.system(f"echo 20000000 > {pwm_path}/period")
            time.sleep(0.1)

            # Disable PWM initially
            self.enable(False)
        else:
            self.pin = None
            print("Error: Invalid Pin")

    def enable(self, flag):
        self.enableFlag = flag
        pwm_path = f"/sys/class/pwm/pwmchip0/pwm{self.pwmx[self.pinIdx]}"
        os.system(f"echo {int(self.enableFlag)} > {pwm_path}/enable")

    def set(self, onTime_us):
        if not self.enableFlag:
            self.enable(True)
        self.onTime_ns = onTime_us * 1000
        pwm_path = f"/sys/class/pwm/pwmchip0/pwm{self.pwmx[self.pinIdx]}"
        os.system(f"echo {self.onTime_ns} > {pwm_path}/duty_cycle")

    def __del__(self):
        if self.pin is not None:
            # Unexport the PWM channel
            os.system(f"echo {self.pwmx[self.pinIdx]} > /sys/class/pwm/pwmchip0/unexport")
            # Reset the pin function
            os.system(f"/usr/bin/pinctrl set {self.pin} ip")


# Example usage
if __name__ == "__main__":
    servo = pi5RC(18)  # Use GPIO 18 (hardware PWM pin)
    try:
        while True:
            # Move servo to 0 degrees (500us pulse width)
            servo.set(500)
            time.sleep(1)

            # Move servo to 90 degrees (1500us pulse width)
            servo.set(1500)
            time.sleep(1)

            # Move servo to 180 degrees (2500us pulse width)
            servo.set(2500)
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nExiting...")
