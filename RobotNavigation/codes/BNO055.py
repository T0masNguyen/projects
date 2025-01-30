import adafruit_bno055
import board
import math
import L298NHBridge as B
import time

# Initializing the sensor
i2c = board.I2C()
sensor = adafruit_bno055.BNO055_I2C(i2c)

def get_mag():
    while True:
        if sensor.magnetic is not None:
            data_tuple = sensor.magnetic
            calibrated_angle = math.atan2(data_tuple[0], data_tuple[1]) * (180 / math.pi)
            
            if calibrated_angle > 0:
                return 180 - calibrated_angle
            elif calibrated_angle <= 0:
                return 360 + (-180 - calibrated_angle)

def yaw():
    while True:
        yaw_angle = sensor.euler[0]
        if yaw_angle is not None:
            return yaw_angle

def spin(target_angle):
    restore_calibration()

    car_angle = get_mag()
    time.sleep(2)

    print(f"Current car angle to the north: {car_angle}")
    time.sleep(2)

    turn_angle = abs(target_angle - car_angle)
    time.sleep(0.5)

    dc = 1  # Motor power

    if turn_angle > 180:
        if target_angle > car_angle:
            B.setMotorRight(dc)
            B.setMotorLeft(-dc)
            turn_angle = 360 - turn_angle
            print(f"Turn left by {turn_angle} degrees!")
            stop_car(turn_angle)
        else:
            B.setMotorRight(-dc)
            B.setMotorLeft(dc)
            turn_angle = 360 - turn_angle
            print(f"Turn right by {turn_angle} degrees!")
            stop_car(turn_angle)
    else:
        if target_angle > car_angle:
            B.setMotorRight(-dc)
            B.setMotorLeft(dc)
            print(f"Turn right by {turn_angle} degrees!")
            stop_car(turn_angle)
        else:
            B.setMotorRight(dc)
            B.setMotorLeft(-dc)
            print(f"Turn left by {turn_angle} degrees!")
            stop_car(turn_angle)

def stop_car(target_angle):
    start_yaw = yaw()
    print(f"Starting yaw: {start_yaw}")

    while abs(yaw() - start_yaw) < target_angle:
        print(f"Turning: {abs(yaw() - start_yaw)} degrees, Magnetic: {get_mag()}")

    B.setMotorRight(0)
    B.setMotorLeft(0)
    time.sleep(0.5)
    print('Target angle reached.')

def restore_calibration():
    sensor.mode = adafruit_bno055.CONFIG_MODE
    time.sleep(0.7)
    
    # Pre-defined calibration offsets
    acceleration = (-35, -16, -31)
    magnetometer = (-491, -725, -310)
    gyro = (-1, -3, 0)
    r_accelerometer = 1000
    r_magnetometer = 668

    sensor.offsets_accelerometer = acceleration
    sensor.offsets_magnetometer = magnetometer
    sensor.offsets_gyroscope = gyro
    sensor.radius_accelerometer = r_accelerometer
    sensor.radius_magnetometer = r_magnetometer

    sensor.mode = adafruit_bno055.NDOF_MODE
    time.sleep(0.7)

if __name__ == "__main__":
    # Example of usage: Spin car to 90 degrees
    spin(90)
