# app.py
from flask import Flask, render_template, Response
from camera import Camera  # Import the Camera class
from motor_controller import MotorController  # Import the MotorWaterController class

app = Flask(__name__)

# Create instances of Camera and MotorWaterController
camera = Camera()
controller = MotorController()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed_rgb')
def video_feed_rgb():
    return Response(camera.generate_rgb_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_depth')
def video_feed_depth():
    return Response(camera.generate_depth_frames(),
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

@app.route('/water_on')
def water_on():
    return controller.turn_water_on()

@app.route('/water_off')
def water_off():
    return controller.turn_water_off()

@app.route('/water_toggle')
def water_toggle():
    return controller.toggle_water()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
