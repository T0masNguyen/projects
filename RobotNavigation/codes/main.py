import time
import L298NHBridge as Bridge
import KalmanFilter as KF
import coordinateconverter as converter
import gpsd
import gps_calculation as gps
from savinggpsdata import save
import BNO055
import numpy as np
import os, readchar
from threading import Thread

# Interface
def getch():
    return readchar.readchar()

# GPS Start Position Thread
def start_pos(stop_event):
    global lat0, lon0, X
    while not stop_event.is_set():
        packet = gpsd.get_current()
        if packet.lat != 0:
            x, y = converter.toUTM(packet.lat, packet.lon)
            angle = BNO055.get_mag("north")
            speed = Bridge.speed()

            # Initial state vector
            X = np.array([[x], [y], [angle], [speed]])

            z = np.array([[x], [y], [angle], [speed]])
            deltaT = time.time() - t
            t = time.time()
            lat0, lon0, pre_lat, pre_lon, X = KF.filter(X, z, packet.error, deltaT)

        if stop_event.is_set():
            distance, turn_angle = gps.calc(lat0, lon0, targetLat, targetLon)
            print(f"Start position: {lat0}, {lon0} | Distance: {distance} meters")
            break

# Main Execution Loop
def main():
    global lat0, lon0, targetLat, targetLon, stop_event

    # Connect to GPSd
    gpsd.connect()

    # Initialize variables
    t = time.time()
    X = np.array([[0], [0], [0], [0]])
    stop_event = Thread.Event()  # Event for stopping the GPS position thread

    # Start the GPS position thread
    t1 = Thread(target=start_pos, args=(stop_event,))
    t1.start()

    # Give the filter time to approximate the start position
    time.sleep(3)
    stop_event.set()  # Signal the thread to stop

    # Main user interaction loop
    while True:
        print("Press 's' to start program!")
        print("Press 'x' to end program!")
        char = getch()

        if char == 's':
            os.system('clear')

            # Get current GPS data and calculate target angle
            packet = gpsd.get_current()
            distance, turn_angle = gps.calc(lat0, lon0, targetLat, targetLon)
            print(f"Target angle: {turn_angle}")
            BNO055.spin(turn_angle)

            # Start motor speed and distance calculations
            Bridge.gapdetection()

            while True:
                packet = gpsd.get_current()

                # If no valid GPS data, stop motors
                if packet.lon == 0:
                    Bridge.setMotorLeft(0)
                    Bridge.setMotorRight(0)
                else:
                    Bridge.setMotorLeft(0.5)
                    Bridge.setMotorRight(0.5)

                    # Convert lat/lon to UTM
                    x, y = converter.toUTM(packet.lat, packet.lon)

                    # Measurement vector
                    angle = BNO055.get_mag("north")
                    speed = Bridge.speed()
                    z = np.array([[x], [y], [angle], [speed]])

                    # Time difference
                    deltaT = time.time() - t
                    t = time.time()

                    # Apply Kalman filter
                    K_lat, K_lon, pre_lat, pre_lon, X = KF.filter(X, z, packet.error, deltaT)

                    # Save GPS data
                    save(packet.lat, packet.lon, K_lat, K_lon, speed, angle, packet.error, deltaT)

                    # Calculate distance using various methods
                    distanceKF, turn_angle = gps.calc(K_lat, K_lon, targetLat, targetLon)
                    distanceGPS, turn_angle = gps.calc(packet.lat, packet.lon, targetLat, targetLon)
                    distanceEstimation, turn_angle = gps.calc(pre_lat, pre_lon, targetLat, targetLon)

                    print(f"KF Distance: {distanceKF} | GPS Distance: {distanceGPS} | Predicted Distance: {distanceEstimation}")

                    if distanceKF < 4:
                        print("Reached the target!")
                        Bridge.setMotorLeft(0)
                        Bridge.setMotorRight(0)
                        Bridge.exit()
                        break

                if char == 'x':
                    os.system('clear')
                    Bridge.setMotorLeft(0)
                    Bridge.setMotorRight(0)
                    Bridge.exit()
                    print("Program Ended!")
                    break

        if char == 'x':
            os.system('clear')
            Bridge.setMotorLeft(0)
            Bridge.setMotorRight(0)
            Bridge.exit()
            print("Program Ended!")
            break
