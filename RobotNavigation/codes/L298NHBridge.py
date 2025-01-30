import RPi.GPIO as io
import time
import math

# Setup GPIO mode
io.setmode(io.BCM)
io.setwarnings(False)

# Pin definitions
DC_MAX = 100
IN1, IN2, IN3, IN4 = 21, 20, 16, 26
ENA, ENB = 27, 22

# Motor control pins
rightmotor_in1_pin = IN1
rightmotor_in2_pin = IN2
leftmotor_in1_pin = IN3
leftmotor_in2_pin = IN4

# Initialize GPIO pins for motors
io.setup([rightmotor_in1_pin, rightmotor_in2_pin, leftmotor_in1_pin, leftmotor_in2_pin], io.OUT)

# Initialize motor PWM pins
rightmotorpwm_pin = ENA
leftmotorpwm_pin = ENB
io.setup([rightmotorpwm_pin, leftmotorpwm_pin], io.OUT)

rightmotorpwm = io.PWM(rightmotorpwm_pin, 100)
leftmotorpwm = io.PWM(leftmotorpwm_pin, 100)

# Initialize motors to stop
rightmotorpwm.start(0)
leftmotorpwm.start(0)
leftmotorpwm.ChangeDutyCycle(0)
rightmotorpwm.ChangeDutyCycle(0)

# Light Sensor pins
io.setup(6, io.IN)
io.setup(5, io.IN)

# Variables
rpm = 0
rightgap_counter = 0
gap_counter = 0
previous_time = 0
velocity = 0
global distance
distance = 0

# Callback functions for wheel rotation detection
def rightwheel(channel):
    global rightgap_counter, rpm, previous_time
    rightgap_counter += 1
    if rightgap_counter >= 20:
        timetaken = int(round(time.time() * 1000)) - previous_time
        rpm = (1000 / timetaken) * 60  # Calculate RPM
        previous_time = int(round(time.time() * 1000))
        rightgap_counter = 0

def leftwheel(channel):
    global gap_counter
    gap_counter += 1

# Gap detection function
def gapdetection():
   io.add_event_detect(6, io.RISING, callback=rightwheel)
   io.add_event_detect(5, io.RISING, callback=leftwheel)

# Calculate velocity based on wheel RPM
def speed():
    global velocity
    # Velocity = 2π × RPS × radius of wheel
    velocity = 0.033 * rpm * 0.104  # Adjust the constants as per your wheel
    return velocity

# Calculate distance based on wheel rotations
def distance():
    # Distance = Circumference × Number of rotations
    distance = (2 * math.pi * 0.033) * (gap_counter / 20)
    return distance

# Set motor mode (forward, reverse, stop)
def setMotorMode(motor, mode):
    if motor == "leftmotor":
        if mode == "reverse":
            io.output(leftmotor_in1_pin, False)
            io.output(leftmotor_in2_pin, True)
        elif mode == "forward":
            io.output(leftmotor_in1_pin, True)
            io.output(leftmotor_in2_pin, False)
        else:
            io.output(leftmotor_in1_pin, False)
            io.output(leftmotor_in2_pin, False)

    elif motor == "rightmotor":
        if mode == "reverse":
            io.output(rightmotor_in1_pin, True)
            io.output(rightmotor_in2_pin, False)
        elif mode == "forward":
            io.output(rightmotor_in1_pin, False)
            io.output(rightmotor_in2_pin, True)
        else:
            io.output(rightmotor_in1_pin, False)
            io.output(rightmotor_in2_pin, False)
    else:
        # Turn off both motors
        io.output(leftmotor_in1_pin, False)
        io.output(leftmotor_in2_pin, False)
        io.output(rightmotor_in1_pin, False)
        io.output(rightmotor_in2_pin, False)

# Set motor power (forward/reverse)
def setMotorLeft(power):
    power = int(power)
    if power < 0:
        setMotorMode("leftmotor", "reverse")
        pwm = -int(DC_MAX * power)
        pwm = min(pwm, DC_MAX)
    elif power > 0:
        setMotorMode("leftmotor", "forward")
        pwm = int(DC_MAX * power)
        pwm = min(pwm, DC_MAX)
    else:
        setMotorMode("leftmotor", "stopp")
        pwm = 0
    leftmotorpwm.ChangeDutyCycle(pwm)

def setMotorRight(power):
    power = int(power)
    if power < 0:
        setMotorMode("rightmotor", "reverse")
        pwm = -int(DC_MAX * power)
        pwm = min(pwm, DC_MAX)
    elif power > 0:
        setMotorMode("rightmotor", "forward")
        pwm = int(DC_MAX * power)
        pwm = min(pwm, DC_MAX)
    else:
        setMotorMode("rightmotor", "stopp")
        pwm = 0
    rightmotorpwm.ChangeDutyCycle(pwm)

# Cleanup function
def exit():
    io.output(leftmotor_in1_pin, False)
    io.output(leftmotor_in2_pin, False)
    io.output(rightmotor_in1_pin, False)
    io.output(rightmotor_in2_pin, False)
    io.cleanup()
