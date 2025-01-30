from Tools.camera_thread import CameraThreadHandler
from Tools.scatter_plot import scatter_plot
import sys


# ----------------------------------------------------------------------------------------------------------------------------------- general settings
cam_type_whitelist = ['Depstech webcam', 'GENERAL WEBCAM']
# cam_type_whitelist.append('Integrated Webcam')
# cam_type_whitelist.append('Integrated Camera')
resolution = (1280, 720)          # resolution:  (320, 180), (640, 360), (1280, 720), (1920, 1080), (2560, 1440)

# ====================================================================================================================================================
if __name__ == '__main__':

    print('--------------------------------------------------------------------------------------------- start process')
    threads = CameraThreadHandler(cam_type_whitelist, resolution, manual_inp_control=False,
                                  system_calibration_file="Data\system_individual_calibration.json",
                                  measure_facial_landmarks=True,
                                  measure_aruco=True,
                                  measure_precision_targets=True,
                                  measure_pose=False,
                                  measure_hands=False,
                                  measure_chessboard=False)

    threads.aruco_dict = 'DICT_4X4_250'
    threads.startup_threads()

    # TODO: create scatter plot -> https://stackoverflow.com/questions/42722691/python-matplotlib-update-scatter-plot-from-a-function
    i = 0
    try:
        while True:
            threads.calculation_barrier.wait()
            threads.thread_lock.acquire()
            # -> function collecting the data out of threads
            threads.thread_lock.release()
            coords = []
            aruco = []
            for thread in threads.thread_list:
                coords.append(thread.results_facial_landmarks)
                aruco.append(thread.results_aruco)
            print("Calculation", i)
            print('---------------------------------------> facial landmarks')
            print(coords)
            print('---------------------------------------> aruco')
            print(aruco)
            #testplot -> be careful: stops in first iteration, outside of the loop?
            # TODO: update scatter plot
            # scatter_plot(coords, aruco)


            print('catch aruco -> check relative orientation of the camera system')
            #print(thread[0].results_aruco)
            # print(thread_list[1].results_aruco)
            print('calculate facial landmarks in the world system')

            i += 1
    except KeyboardInterrupt:
        sys.exit(0)

    print("Exit\n")
