"""This scripts uses OpenCV to capture webcam output, applies a filter, and sends it to the virtual camera.

It also shows how to use BGR as pixel format.
=> https://github.com/letmaik/pyvirtualcam
=> install OBS Studio -> https://obsproject.com/
"""

import mediapipe as mp
import cv2 as cv
import argparse
import pyvirtualcam
from pyvirtualcam import PixelFormat
from Measurement_operators.aruco_detection import ArucoDetection
from Measurement_operators.mediapipe_facial_landmarks import findFacialLandmarks

aruco_dict_name = 'DICT_4X4_250'
ad = ArucoDetection(aruco_dict_name)

parser = argparse.ArgumentParser()
parser.add_argument("--camera", type=int, default=0, help="ID of webcam device (default: 0)")
parser.add_argument("--fps", action="store_true", help="output fps every second")
parser.add_argument("--filter", choices=["shake", "none"], default="shake")
args = parser.parse_args()

vc = cv.VideoCapture(2, cv.CAP_DSHOW)

if not vc.isOpened():
    raise RuntimeError('Could not open video source')

pref_width = 1280
pref_height = 720
pref_fps_in = 20
vc.set(cv.CAP_PROP_FRAME_WIDTH, pref_width)
vc.set(cv.CAP_PROP_FRAME_HEIGHT, pref_height)
vc.set(cv.CAP_PROP_FPS, pref_fps_in)

print('init mediapipe:')
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=5, circle_radius=1)
print('   ---> done')



# Query final capture device values (may be different from preferred settings).
width = int(vc.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(vc.get(cv.CAP_PROP_FRAME_HEIGHT))
fps_in = vc.get(cv.CAP_PROP_FPS)
print(f'Webcam capture started ({width}x{height} @ {fps_in}fps)')

fps_out = 20

with pyvirtualcam.Camera(width, height, fps_out, fmt=PixelFormat.BGR, print_fps=args.fps) as cam:
    print(f'Virtual cam started: {cam.device} ({cam.width}x{cam.height} @ {cam.fps}fps)')

    # Shake two channels horizontally each frame.
    channels = [[0, 1], [0, 2], [1, 2]]

    while True:
        # Read frame from webcam.
        ret, frame = vc.read()
        success, img = vc.read()
        # --------------------------------------------------------------------------------- operators on grayscale image
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        marker_dict = ad.detect_markers(img, True, True)
        aruco_ids, aruco_points = ad.marker_dict_to_numpy_arrays(marker_dict)
        #findORB(gray_img, gray) # for edges
        #findSift(gray_img, gray)  # for inner contour-points
        #findChessboardCorner(gray_img, gray)  # Chessbord
        #findSurf(gray)  # algorithm are patented -> not free available
        # ----------------------------------------------------------------------------- operators on grayscale rgb image
        rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        findFacialLandmarks(mp_face_mesh, mp_drawing_styles, mp_drawing, img, rgb, draw=True)


        if not ret:
            raise RuntimeError('Error fetching frame')

        if args.filter == "shake":
            dx = 15 - cam.frames_sent % 5
            c1, c2 = channels[cam.frames_sent % 3]
            frame[:,:-dx,c1] = frame[:,dx:,c1]
            frame[:,dx:,c2] = frame[:,:-dx,c2]

        # Send to virtual cam.
        cam.send(img)

        # Wait until it's time for the next frame.
        cam.sleep_until_next_frame()
