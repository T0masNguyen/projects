import mediapipe as mp
import cv2 as cv
import time
import numpy as np
from Tools.system_analyzer import get_cam_names_pygrabber
from Measurement_operators.high_precision_targets import findHighPrecisionTargets
from Measurement_operators.aruco_detection import ArucoDetection


# ======================================================================================================================
"""
 test unit for a single camera
 -> running without the camera thread
"""


def _display_fps(start_time: time, end_time: time, old_avg_fps: float, cnt: int, img: np.ndarray, font):
    """Cloned function CamThread._display_fps()"""
    curr_fps = 1.0 / (end_time - start_time)
    curr_fps_string = "FPS: %4.1f" % curr_fps
    avg_fps = (old_avg_fps * (cnt - 1) + curr_fps) / cnt
    avg_fps_string = "AVG: %4.1f" % avg_fps
    cv.putText(img, curr_fps_string, (10, 25), font, 0.75, (0, 255, 0), 2, cv.LINE_AA)
    cv.putText(img, avg_fps_string, (160, 25), font, 0.75, (0, 128, 0), 2, cv.LINE_AA)

    return avg_fps


if __name__ == "__main__":
    aruco_dict_name = 'DICT_4X4_250'
    ad = ArucoDetection(aruco_dict_name)
    cam_names = get_cam_names_pygrabber()

    streamid = 1
    print('Set-Stream: ', streamid)
    cap = cv.VideoCapture(streamid, cv.CAP_DSHOW) #,cv.CAP_ANY) #; , cv.CAP_MSMF) # cv.CAP_DSHOW)

    print('   ---> done')
    print('Set-Resolution:')
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(3, 1280)
    # cap.set(3, 1920)
    # cap.set(3,2560)
    print('   ---> ', cap.get(3), ' X ', cap.get(4), 'Pixel')

    font = cv.FONT_HERSHEY_SIMPLEX

    print('init mediapipe:')
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_face_mesh = mp.solutions.face_mesh
    drawing_spec = mp_drawing.DrawingSpec(thickness=5, circle_radius=1)
    print('   ---> done')

    cnt = 1
    avg_fps = 0
    while True:
        start_time = time.time()
        success, img = cap.read()
        # --------------------------------------------------------------------------------- operators on grayscale image
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        #marker_dict = ad.detect_markers(gray_img, True, True)
        #aruco_ids, aruco_points = ad.marker_dict_to_numpy_arrays(marker_dict)
        findHighPrecisionTargets(img, gray)
        #findChessboardCorner(gray_img, gray)  # Chessbord
        # ----------------------------------------------------------------------------- operators on grayscale rgb image
        rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        # findFacialLandmarks(mp_face_mesh, mp_drawing_styles, mp_drawing, gray_img, rgb, draw=True)
        # ------------------------------------------------------------------------------------------------------ Anzeige
        end_time = time.time()
        avg_fps = _display_fps(start_time, end_time, avg_fps, cnt, img, font)
        cnt += 1

        cv.imshow('gray_img', img)
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break
    cap.release()
    cv.destroyAllWindows()
