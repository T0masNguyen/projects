import paths
from Tools.camera_utils import CameraResolution, CameraDict, CameraParameters, InnerCoeff, read_cam_calib_json
from Tools.point_pattern_parser import PointPatternParser
from Tools.system_analyzer import get_cam_names_pygrabber
from Tools.trafo_module import PoseTrafo

from Measurement_operators.aruco_detection import ArucoDetection
import math
import uuid
import cv2 as cv
import numpy as np
import os
import time


def undistort_image(rgb_img, mtx, dist, estimate_new_mtx: bool = False):
    """Calculates new camera matrix (after cropping or for undistortion).

    :param rgb_img: rgb image (distorted, undistorted)
    :param mtx: camera matrix from json-file
    :param dist: distortion_coefficients from json file
    :return: newcameraMatrix
    """
    # -> https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html#ga7dfb72c9cf9780a347fbe3d1c47e5d5a
    # Refining the camera matrix using parameters obtained by calibration
    # mtx = np.array([[innerPara[0], 0, innerPara[2]], [0, innerPara[1], innerPara[3]], [0, 0, 1]]) --> wenn aus json Datei gelesen wird
    # mtx = np.array([[innerPara[0][0], 0, innerPara[0][2]], [0, innerPara[0][1], innerPara[0][3]], [0, 0, 1]])

    # -------------------------------------------------------------------------------------image size
    h, w = rgb_img.shape[:2]

    # -----------------------------------------------------------------Query whether newcameramatrix should be determined
    if estimate_new_mtx:
        newcameraMatrix, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    else:
        roi = 0, 0, w, h
        newcameraMatrix = mtx

    # ------------------------------------------------------------------------------------------- Method to undistort the image
    mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameraMatrix, (w, h), 5)   #  5=CV_32FC1, 11=CV_16FC2, 13=CV_32FC2
    undistored_img = cv.remap(rgb_img, mapx, mapy, cv.INTER_LINEAR)

    # --------------------------------------------------------------------------------------------------------------------------------- crop the image
    x, y, w, h = roi
    cropped_img = undistored_img[y:y + h, x:x + w]

    # ------------------------------------------------------------------------------------------new mtx after cropping
    if estimate_new_mtx:
        dist2 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], float)
        newcameraMatrix = new_camera_matrix(cropped_img, newcameraMatrix, dist2)

    #cropped_mtx, roi = cv.getOptimalNewCameraMatrix(newcameraMatrix, dst, (w, h), 1, (w, h))

    return cropped_img, newcameraMatrix


def new_camera_matrix(rgb_img, mtx, dist):
    """Calculates new camera matrix (after cropping or for undistortion).

    :param rgb_img: rgb image (distorted, undistorted)
    :param mtx: camera matrix from json-file
    :param dist: distortion_coefficients from json file
    :return: newcameraMatrix
    """

    h, w = rgb_img.shape[:2]
    newcameraMatrix, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    return newcameraMatrix


def get_default_calib_parameters(filtered_cam_dict: CameraDict, calib_file: str, read_extrinsic: bool = False) -> CameraDict:
    """Assignment of cameras to inner and extrinsic parameters from default type calib json-file.

    :param filtered_cam_dict: filtered_cam_dict with relevant cameras
    :param calib_file: full PATH to cam type calibration file cam_type_calibration.json
    :return: CalibParameter_CameraDict
    """

    print('\n--- Load default camera type calibration parameters ---')
    cam_dict = filtered_cam_dict
    for cam in cam_dict.devices.values():
        found, new_cam = read_cam_calib_json(cam.camera_type_name, calib_file, True, read_extrinsic)
        cam.type_calibration = found
        cam.calib_resolution = new_cam.resolution
        cam.dist_coeff = new_cam.dist_coeff
        cam.extrinsic = new_cam.extrinsic
        if found:
            cam.inner_coeff = new_cam.inner_coeff
        else:
            inner_coeff: InnerCoeff = InnerCoeff()
            inner_coeff.cx = int(cam.resolution.x / 2)
            inner_coeff.cy = int(cam.resolution.y / 2)
            inner_coeff.fx = int(cam.resolution.x)
            inner_coeff.fy = int(cam.resolution.y)
            cam.inner_coeff = inner_coeff

        # --------------------------------------------------------------------------------------------------------------------------------------------
        print(f'For stream id: {cam.stream_id}', f'camera type: "{cam.camera_type_name}"', sep='\t', end=':\n')
        print('\tfound camera type calibration: ' + str(cam.type_calibration))
        print('\t' + str(cam.inner_coeff))
        print('\t' + str(cam.dist_coeff))

    return cam_dict


def get_indiv_calib_parameters(filtered_cam_dict: CameraDict, calib_file: str, read_extrinsic: bool = False) -> CameraDict:
    """Assignment of inner and extrinsic parameters from individual calib json-file to connected camera uuids.

    :param filtered_cam_dict: filtered_cam_list with relevant cameras
    :param calib_file: name and path to cam type calibration file cam_type_calibration.json
    :return: CalibParameter_CameraList
    """
    # TODO: check the cameras before reading the values!!!!
    print('\n--- Load individual camera calibration parameters ---')
    cam_dict = filtered_cam_dict
    i = 1  # new Anne
    for cam in cam_dict.devices.values():
        cam_identifier = cam.camera_type_name + " " + str(i)  # neu Anne
        found, new_cam = read_cam_calib_json(cam_identifier, calib_file, False, read_extrinsic)
        cam.indiv_calibration = found
        i = i+1
        if found:
            cam.calib_resolution = new_cam.resolution
            cam.dist_coeff = new_cam.dist_coeff
            cam.extrinsic = new_cam.extrinsic
            cam.inner_coeff = new_cam.inner_coeff

        # --------------------------------------------------------------------------------------------------------------------------------------------
        print(f'For stream id: {cam.stream_id}', f'camera type: "{cam.camera_type_name}"', f'uuid: {cam.u_id}', sep='\t', end=':')
        print('\tfound individual camera calibration: ' + str(cam.indiv_calibration))
        print('\t' + str(cam.inner_coeff))
        print('\t' + str(cam.dist_coeff))
        print('\t' + str(cam.extrinsic))

    return cam_dict


def get_orientation_assignment(cam_dict: CameraDict):
    """
    File muss geladen werden:
    -  äußere und innere Orientierung
    Vergleich tatsächliche und gespeicherte Äußere Orientierung
    -> Zuweisung der Inneren Orientierung zur Stream ID
    Zuordnung -> StreamID zu Kalibrierparameter

    :return:
    """

    # ------------------------------------------------------------------------------------------------------ load object points from aruco target file
    ppp = PointPatternParser(paths.ARUCO_TARGET_FILE.path) # TODO: from system calibration file!!!
    ppp.parse_file()
    aruco_target_ids = ppp.ar_id_array
    aruco_target_points = ppp.ar_point_array
    aruco_dict_name = ppp.processed_data['settings']['aruco_dict']
    print('\n--- Aruco target ----------------------------------------------------------------------')
    print('\tAruco-IDs: ' + str(aruco_target_ids))
    # print('\tAruco-coordinates: \n' + str(aruco_target_points))

    ad = ArucoDetection(aruco_dict_name)
    ext_orientations = []

    # ----------------------------------------------------------------------------------------------------------------- take initial images per webcam
    for cam in cam_dict.devices.values():
        # ----------------------------------------------------------------------------------------------------------- camera type specific information
        cam_mtx = cam.inner_coeff.get_camera_matrix()
        cam_dist = cam.dist_coeff.get_dist_coeff_values()
        # ------------------------------------------------------------------------------------------------------------------- grab and undistort image
        print('Grab image of camera: ' + str(cam.camera_type_name) + '\t stream_id: ' + str(cam.stream_id))
        cap = cv.VideoCapture(int(cam.stream_id), cv.CAP_DSHOW)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, cam.resolution.x)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, cam.resolution.y)
        return_value, rgb_image = cap.read()
        cap.release()

        undist_rgb_img, cropped_mtx = undistort_image(rgb_image, cam_mtx, cam_dist, True)

        # ----------------------------------------------------------------------------------------------------------------- subpixel image coordinates
        detect_res = ad.detect_markers(undist_rgb_img, undist_rgb_img, do_subpix=True, draw_markers=True)

        # ---------------------------------------------------------------------------------------------------------- save initial image of each camera
        # TODO: Pfad prüfen, wenn nicht vorhanden dann anlegen ...
        print('\t-> write to path: Data/cam_sample_calibration/_image_sets/correction_orientation')

        # --------------------------------------------------------------------------------------------check path and build new folder if not existent
        image_path = 'Data/cam_sample_calibration/_image_sets/correction_orientation/'
        if os.path.exists(image_path) == False:
            os.mkdir(image_path)

        #Anne
        #now = time.time()
        #test = image_path + str(now) + '_distorted.jpg'
        #print(test)
        #Ende

        cv.imwrite(image_path +
                    str(cam.camera_type_name) + '_' + str(cam.stream_id) + '_distorted.jpg', rgb_image)
        #cv.imwrite(test, rgb_image)
        cv.imwrite(image_path +
                    str(cam.camera_type_name) + '_' + str(cam.stream_id) + '_undistorted.jpg', undist_rgb_img)

        if detect_res['data']:
            # ----------------------------------------------------------------------------------------------- sync content of target and detected data
            target_ids = list(aruco_target_ids)
            
            surplus_target = [idx for idx, ar_id in enumerate(target_ids) if str(ar_id) not in detect_res['data'].keys()]
            aruco_target_ids = np.delete(aruco_target_ids, surplus_target)
            aruco_target_points = np.delete(aruco_target_points, surplus_target, axis=0)

            # ------------------------------------------------------------------------------------------- build image point and id arrays from results
            aruco_image_ids, aruco_image_points = ad.marker_dict_to_numpy_arrays(detect_res, list(aruco_target_ids))

            # ------------------------------------------------------------------------------------------------- sync order of target and detected data
            meas_sorted_idxs = np.argsort(aruco_image_ids)          # -> index sorting template by aruco id order
            aruco_image_ids = aruco_image_ids[meas_sorted_idxs]
            aruco_image_points = aruco_image_points[meas_sorted_idxs]

            target_sorted_idxs = np.argsort(aruco_target_ids)
            #target_sorted_idxs = np.argsort(aruco_image_ids)
            aruco_target_ids = aruco_target_ids[target_sorted_idxs]
            aruco_target_points = aruco_target_points[target_sorted_idxs]
            print(aruco_target_points)

            print('\t-> Aruco image measurement results restricted to target markers:')
            print('\t\tImage aruco ids:', aruco_image_ids, sep='\t')
            print('\t\tImage points:', *aruco_image_points, sep='\n\t\t\t')
            # ------------------------------------------------------------------------------------------------------------------------------- SolvePNP
            print('\t-> solvePnP')

            try:
                flags = (cv.SOLVEPNP_ITERATIVE)
                if len(aruco_target_points) < 6:
                
                    raise ValueError(
                        f'To few aruco markers were detected for processing the exterior orientation. Found {len(aruco_target_points[0])}, need 6.')
                else:
                    print('\t-> solvePnP')

                    cam_dist2  = np.zeros((14,1)) # Assuming no lens distortion
                    ret, rvecs, tvecs = cv.solvePnP(aruco_target_points, aruco_image_points, cropped_mtx, cam_dist2, flags) # cropped_mtx and cam_dist2 (no distortion coefficients because of undistorted image
                    trafo = PoseTrafo('rodrigues', 'rad', rvecs, tvecs)
                    #trafo = PoseTrafo('rodrigues', 'deg', rvecs, tvecs)

                    #alternative approach --------------------#
                    rmat = cv.Rodrigues(rvecs)[0]
                    camera_position = -np.matrix(rmat).T * np.matrix(tvecs)
                    print('camera_position')
                    print(camera_position)
                    #----------------------------------------#

                    # ext_trafos.append(trafo)
                    t_rvec = trafo.get_rvec('eulerxyz', 'rad')#deg
                    t_tvec = trafo.tvec
                    image_ext_orient = np.vstack((t_tvec, t_rvec)).reshape(1, 6)
                    #Be carefull! false calculation of image_ext_orient. Check in progress
                    #instead: image_ext_orient = camera_position, but false angles
                    image_ext_orient[0][0] = camera_position[0]
                    image_ext_orient[0][1] = camera_position[1]
                    image_ext_orient[0][2] = camera_position[2]
                    #----------------------------------------

                    if len(ext_orientations) > 0:
                        ext_orientations = np.concatenate((ext_orientations, image_ext_orient), axis=0)
                    else:
                        #ext_orientations = np.concatenate(image_ext_orient, axis=0)
                        ext_orientations = image_ext_orient

                    print('ext_orientations')
                    print(ext_orientations)
                    #aruco_image_points2, _ = cv.projectPoints(aruco_target_points, rvecs, tvecs, cam_mtx, cam_dist) #cropped_mtx?
                    aruco_image_points2, _ = cv.projectPoints(aruco_target_points, rvecs, tvecs, cropped_mtx, cam_dist2)  # cropped_mtx and cam_dist2 (no distortion coefficients because of undistorted image???
                    aruco_image_points2 = aruco_image_points2.reshape(-1, 2)

                    # ------------------------------------------------------------------------------------------------------------ draw reprojection error
                    draw_errors_in_image(aruco_image_points, aruco_image_points2, aruco_target_points, undist_rgb_img, cam.stream_id, image_path)
            except:
                image_ext_orient = [0, 0, 0, 0, 0, 0]
                ext_orientations = image_ext_orient.copy()

    # ----------------------------------------------------------------------------------------------------------------- save corrected calib-json file
    # #save in file or delivery to class 'CameraElement'?
    # calibfile_corrected = specific_calibfile.replace('.json', '_cor.json')
    # write_jsonFile_CamCalibList(calibfile_corrected, CalibParameter_CameraList)

    return ext_orientations  # , ext_trafos


def filter_connected_cameras(cam_type_whitelist, resolution: CameraResolution) -> CameraDict:
    """
    considers only relevant cameras
    :param cam_type_whitelist: list of relevant cameras types
    :param resolution: apply resolution tu the camera description
    :return: filtered camera dictionary with cameras uuid as key and its CameraParameters as value
    """

    graph = get_cam_names_pygrabber()
    cam_dict = CameraDict()
    for idx, name in enumerate(graph):
        if name in cam_type_whitelist:
            cam = CameraParameters()
            cam.camera_type_name = name
            cam.stream_id = idx
            cam.u_id = uuid.uuid4()
            cam.resolution = resolution
            cam_dict.devices[cam.u_id] = cam

    return cam_dict


def set_ext_orientation(cam_dict_indiv):
    '''

       :param cam_dict_indiv:
       :return ext_orientation:
    '''
    ext_orientation=[]

    for cam in cam_dict_indiv.devices.values():
        ext_orientation.append(cam.extrinsic.vector)

    return ext_orientation


def assignment_outer_orientation_to_cam_id(cam_list, outer_orientations, ext_orientations):
    '''
    considers only relevant cameras
    :param outer_orientations:
    :param calib_para:
    :return:
    '''

    # ---------------------------------------------------------------------------------DistancePattern for the 4 cameras
    
    dis_pattern_real = distancePattern(outer_orientations)
    
    dis_pattern_post = distancePattern(ext_orientations)
    

    dis_pattern_pre = distancePattern(ext_orientations)
    sort_list = compare_distancePattern(dis_pattern_real, dis_pattern_pre)
    # ---------------------------------------------------------------------------------------sort new cam list
    new_oriented_cam_list = sorting_cam_list(cam_list, sort_list)

    return new_oriented_cam_list


def sorting_cam_list(cam_list, sort_list):
    cam_dict_new = CameraDict()
    new_oriented_cam_list = []

            #cam_dict.devices[cam.u_id] = cam

    for iter in sort_list:
        print(iter)
        i = 0
        for cam in cam_list.devices.values():
            if i == iter:
                cam_dict_new.devices[cam.u_id] = cam
            i=1+i

    return cam_dict_new


def distancePattern(orientation):
    '''
    calculates the angle orientation of each camera
    dis_pattern_pre = distancePattern(ext_orientations)
    :return: matrix_dis
    '''
    '''SPX = 0
    SPY = 0
    SPZ = 0
    X = 0
    Y = 0
    Z = 0
    anz = len(orientation)
    for ori1 in orientation:
        X += ori1[0]
        Y += ori1[1]
        Z += ori1[2]
    SPX = X / anz
    SPY = Y / anz
    SPZ = Z / anz'''

    '''matrix_dis = np.zeros((4, 4), dtype=float)
    i = 0
    for ori1 in orientation:
        j = 0
        for ori2 in orientation:
            if any(ori2 == ori1):
                dis_sqrt = ((ori1[0] - SPX) * (ori1[0] - SPX)) + ((ori1[1] - SPY) * (ori1[1] - SPY)) + ((ori1[2] - SPZ) * (ori1[2] - SPZ))
            else:
                dis_sqrt = ((ori1[0] - ori2[0]) * (ori1[0] - ori2[0])) + ((ori1[1] - ori2[1]) * (ori1[1] - ori2[1])) + ((ori1[2] - ori2[2]) * (ori1[2] - ori2[2]))

            dis = math.sqrt(dis_sqrt)
            matrix_dis[i][j] = dis
            j = j + 1
        i = i + 1

    print(matrix_dis)'''

    matrix_dis = np.zeros((4, 4), dtype=float)
    i = 0
    for ori1 in orientation:
        j = 0
        for ori2 in orientation:
            if any(ori2 == ori1):
                dis_sqrt = ((ori1[3]) * (ori1[3])) + ((ori1[4]) * (ori1[4])) + (
                            (ori1[5]) * (ori1[5]))
            #else:
            #    dis_sqrt = ((ori1[3] - ori2[3]) * (ori1[3] - ori2[3])) + ((ori1[4] - ori2[4]) * (ori1[4] - ori2[4])) + (
            #                (ori1[5] - ori2[5]) * (ori1[5] - ori2[5]))

            dis = math.sqrt(dis_sqrt)
            matrix_dis[i][j] = dis
            j = j + 1
        i = i + 1

    print(matrix_dis)
    return matrix_dis


def compare_distancePattern(dis_pattern_real, dis_pattern_pre):
    '''
        compares the angle of each camera
        :param dis_pattern_real
        :param dis_pattern_pre:
        :return:
        '''
    sum_pattern_real = []
    i=0
    for idx in dis_pattern_real:
        sum_real = 0
        sum_real += idx[i]
        i = i + 1
        
        sum_pattern_real.append(sum_real)
    print(sum_pattern_real)

    i=0
    sum_pattern_pre = []
    for idx in dis_pattern_pre:
        sum_pre = 0
        sum_pre += idx[i]
        i = i+1
       
        sum_pattern_pre.append(sum_pre)
    print(sum_pattern_pre)

    number_list = []
    for idi in sum_pattern_real:
        i = 0
        dis_min = 100
        for idj in sum_pattern_pre:
            dis = math.fabs(idi - idj)
            if dis < dis_min:
                idj_nr = i
                dis_min = dis
            i = i + 1
        number_list.append(idj_nr)

    return number_list
        


def get_camera_assignment(cam_dict: CameraDict, default_calibfile: str, specific_calibfile: str) -> CameraDict:
    """
    :param camlist: list with all relevant cameras
    :param default_calibfile: file PATH of the calibration file (*.json) for all 4 cameras of the camera system
    :param aruco_target_file: file PATH for aruco targets
    :return:
    get_filter_cam_list: filtered camera list only with 'GENERAL WEBCAMs' and without internal webcam
    get_final_CalibParameter: finale camera calibration parameter
    outer_orientations: final outer orientation corrected with solvePnP
    """
    # ---------------------------------------------------setting of pre-orientation and calib parameters for each camera
    default_calib_cam_list = get_default_calib_parameters(cam_dict, default_calibfile)
    # -------------------------------------------------------------------final outer orientation corrected with solvePnP
    outer_oriPNP = get_orientation_assignment(default_calib_cam_list)
    # ------------------------------- setting extern orientation from calib-file
    
    # TODO: result outer_oriPNP -> same length as default_calib_cam_list

    # cam_dict_indiv = get_indiv_calib_parameters(cam_dict, specific_calibfile, True)
    # ext_orientations = set_ext_orientation(cam_dict_indiv)

    # new_oriented_cam_list = assignment_outer_orientation_to_cam_id(default_calib_cam_list, outer_oriPNP, ext_orientations)
    new_oriented_cam_list = default_calib_cam_list
    # -------------------------------------------------------------------------------------------------------------------
    return new_oriented_cam_list
    


# ======================================================================================================================
def draw_errors_in_image(aruco_image_points, aruco_image_points_post, aruco_object_points, undist_rgb_image, cam_stream_id: int, image_path: str):
    '''
    draws point errors after Solve-pnp as ellipses in the image
    :param aruco_image_points: list with aruco image points before solve pnp (x,y)
    :param aruco_image_points_post: list with aruco image points after solve pnp (x,y)
    :param aruco_object_points: list with aruco object points (x,y)
    :param cam_dict: camera dictionary
    :param undist_rgb_image: undistorted rgb image
    :param image_path: file PATH for saving image with ellipses
    '''

    sum_error = 0
    for i in range(len(aruco_image_points)):
        error = cv.norm(aruco_image_points[i], aruco_image_points_post[i], cv.NORM_L2)  # immer andere Werte?
        sum_error += error

        # scale factor 1.5 for ellipse drawing
        radius1 = abs(aruco_image_points[i][0] - aruco_image_points_post[i][0]) * 1.5
        radius2 = abs(aruco_image_points[i][1] - aruco_image_points_post[i][1]) * 1.5
        axes = (int(radius1), int(radius2))
        color = (0, 0, 255)
        center = (int(aruco_image_points_post[i][0]), int(aruco_image_points_post[i][1]))

        center = (int(aruco_image_points[i][0]), int(aruco_image_points[i][1]))

        angle_rad = math.atan((aruco_image_points[i][0] - aruco_image_points_post[i][0]) / (aruco_image_points[i][1] - aruco_image_points_post[i][1]))
        angle_deg = angle_rad * 360 / (2 * math.pi)

        thickness = 1
        start_angle = 0
        end_angle = 360

        undist_rgb_image_ell = cv.ellipse(undist_rgb_image, center, axes, angle_deg, start_angle, end_angle, color, thickness)

        cv.imwrite(image_path + str(cam_stream_id) + '_errorEllipses.jpg', undist_rgb_image_ell)

    mean_error = round((sum_error / len(aruco_object_points)), 5)
    print("total mean error: {}".format(mean_error))

    cv.putText(undist_rgb_image, text=str(mean_error), org=(25, 25), fontFace=cv.FONT_HERSHEY_TRIPLEX, fontScale=1,
               color=(0, 255, 255), thickness=1)
    cv.imwrite(image_path + str(cam_stream_id) + '_errorEllipses.jpg', undist_rgb_image_ell)


