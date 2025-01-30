import csv
from datetime import datetime
import BNO055 as B
import time

def get_current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def save_data(A, M, G, RA, RM, filename):
    with open(filename, 'a', newline='') as f:
        header = ['A', 'M', 'G', 'RA', 'RM']
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writerow({'A': A, 'M': M, 'G': G, 'RA': RA, 'RM': RM})

def initialize_sensor():
    while not B.sensor.calibrated:
        print(B.sensor.calibration_status, B.sensor.calibrated)
        time.sleep(0.5)

def main():
    timestamp = get_current_time()
    filename = f'/home/pi/Sensor/offsets_{timestamp}.csv'
    
    with open(filename, 'a', newline='') as f:
        header = ['A', 'M', 'G', 'RA', 'RM']
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()

    initialize_sensor()
    
    A = B.sensor.offsets_accelerometer
    M = B.sensor.offsets_magnetometer
    G = B.sensor.offsets_gyroscope
    RA = B.sensor.radius_accelerometer
    RM = B.sensor.radius_magnetometer
    
    print(A, M, G, RA, RM)
    
    save_data(A, M, G, RA, RM, filename)

if __name__ == "__main__":
    main()
