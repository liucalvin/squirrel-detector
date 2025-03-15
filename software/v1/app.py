from flask import Flask, render_template, Response, request
from camera_controller import CameraController
from motor_controller import MotorController

app = Flask(__name__)

# Create an instance of the MotorController
controller = MotorController()

# Create an instance of the CameraController
camera_controller = CameraController(controller)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed_rgb')
def video_feed_rgb():
    return Response(camera_controller.generate_rgb_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_depth')
def video_feed_depth():
    return Response(camera_controller.generate_depth_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Motor and water control routes
@app.route('/move_left')
def move_left():
    return controller.move_left()

@app.route('/move_right')
def move_right():
    return controller.move_right()

@app.route('/move_up')
def move_up():
    return controller.move_up()

@app.route('/move_down')
def move_down():
    return controller.move_down()

@app.route('/move_left_big')
def move_left_big():
    return controller.move_left(step_size=10)

@app.route('/move_right_big')
def move_right_big():
    return controller.move_right(step_size=10)

@app.route('/move_up_big')
def move_up_big():
    return controller.move_up(step_size=10)

@app.route('/move_down_big')
def move_down_big():
    return controller.move_down(step_size=10)

@app.route('/water_on')
def water_on():
    return controller.turn_water_on()

@app.route('/water_off')
def water_off():
    return controller.turn_water_off()
  
@app.route('/water_toggle')
def water_toggle():
    if controller.water_on == 0:
        return controller.turn_water_on()
    else:
        return controller.turn_water_off()

# Custom serial command route
@app.route('/send_command', methods=['POST'])
def send_command():
    command = request.form.get('command')  # Get the command from the form
    if command:
        controller.send_custom_command(command)
        return f"Sent custom command: {command}"
    return "No command provided."

@app.route('/toggle_motor_tracking', methods=['POST'])
def toggle_motor_tracking():
    motor_tracking = request.form.get('motor_tracking')  # Get the value from the form
    if motor_tracking is not None:
        motor_tracking = motor_tracking.lower() == 'true'
        camera_controller.set_motor_tracking(motor_tracking)
        return f"Motor tracking set to {motor_tracking}"
    return "No value provided for motor_tracking."


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)

