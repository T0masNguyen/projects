import cv2 as cv
import numpy as np


def findChessboardCorner(img, gray, draw=True):
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    axis = np.float32([[10, 0, 0], [0, 10, 0], [0, 0, -10]]).reshape(-1, 3)

    target_points = np.zeros((11 * 8, 3), np.float32)  # erstellt Matrix mit nullen in der größe 6*9, 3
    target_points[:, :2] = 200 / 10 * np.mgrid[0:11, 0:8].T.reshape(-1, 2)  # wenn aruco -> dann mit ID ...

    # TODO: use CamDict instead od hard coded parameters
    fx = 1280
    fy = 720

    cameraMatrix = np.array([[1850 / 2, 0, fx / 2],
                             [0, 1850 / 2, fy / 2],
                             [0, 0, 1]])

    distCoeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    ret, corners = cv.findChessboardCorners(gray, (11, 8), None)  # https://docs.opencv.org/master/d7/d53/tutorial_py_pose.html

    if ret == True:
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        ret, rvecs, tvecs = cv.solvePnP(target_points,  # points_2D,
                                         corners2,
                                         cameraMatrix,
                                         distCoeffs)
        # ,None, None, False, cv.SOLVEPNP_UPNP)

        # project 3D points to image plane
        imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, cameraMatrix, distCoeffs)

        if ret == True:
            print("Position: %4.2f" % tvecs[0], " %4.2f" % tvecs[1], " %4.2f" % tvecs[2], '  ---  ',
                  "Rotation: %4.2f" % rvecs[0], " %4.2f" % rvecs[1], " %4.2f" % rvecs[2], )
        # print(R_exp)
        else:
            print('solvePnP -> error')

        if draw:
            cv.drawChessboardCorners(img, (11, 8), corners2, ret)
            #
            corner = tuple(corners2[0].ravel())
            cv.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 5)
            cv.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 5)
            cv.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 5)


if __name__ == '__main__':
    print("unit for chessboard detection \n \t -> run facial_landmarks_pos_detection.py  \n \t    with option measure_chessboard=True")

