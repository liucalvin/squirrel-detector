import serial
from time import sleep

class MotorController:
    def __init__(self):
        # Initialize motor angles and water state
        self.motor_angle_1 = 90  # Neutral position (90 degrees)
        self.motor_angle_2 = 90  # Neutral position (90 degrees)
        self.water_on = 0  # Water control (0 = off, 1 = on)

        # Serial communication setup
        self.ser = serial.Serial('/dev/ttyAMA0', 115200, timeout=1)  # Baud rate set to 115200
        print("Serial port opened.")  # Debug print

    def move_left(self, step_size=10):
        self.motor_angle_1 = min(180, self.motor_angle_1 + step_size)  # Limit to 180 degrees
        self.send_serial_command()
        return f"Moved Motor 1 Left to {self.motor_angle_1} degrees"

    def move_right(self, step_size=10):
        self.motor_angle_1 = max(0, self.motor_angle_1 - step_size)  # Limit to 0 degrees
        self.send_serial_command()
        return f"Moved Motor 1 Right to {self.motor_angle_1} degrees"

    def move_up(self, step_size=10):
        self.motor_angle_2 = min(180, self.motor_angle_2 + step_size)  # Limit to 180 degrees
        self.send_serial_command()
        return f"Moved Motor 2 Up to {self.motor_angle_2} degrees"

    def move_down(self, step_size=10):
        self.motor_angle_2 = max(0, self.motor_angle_2 - step_size)  # Limit to 0 degrees
        self.send_serial_command()
        return f"Moved Motor 2 Down to {self.motor_angle_2} degrees"

    def turn_water_on(self):
        self.water_on = 1  # Turn water on
        self.send_serial_command()
        return "Water turned ON"

    def turn_water_off(self):
        self.water_on = 0  # Turn water off
        self.send_serial_command()
        return "Water turned OFF"

    def toggle_water(self):
        if self.water_on == 0:
            return self.turn_water_on()
        else:
            return self.turn_water_off()

    def send_serial_command(self):
        # Send the serial command in the format: motor_angle_1,motor_angle_2,water_on
        command = f"{self.motor_angle_1},{self.motor_angle_2},{self.water_on}\n"
        print(f"Sending serial command: {command.strip()}")  # Debug print
        self.ser.write(command.encode())
        sleep(0.1)  # Add a small delay
