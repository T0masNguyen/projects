import cv2 as cv
import numpy as np
from typing import Union, List
import os

"""
base class for pose transformation
-> refactor
-> use as submodule!
"""


def get_ext_orientation(data_path: str, aruco_local_points: np.ndarray, target_ar_ids: list, cam_mtx: np.ndarray, dist_coeff: np.ndarray):
    """Gets the exterior orientation as in the perspective of a aruco pattern in a camera system in form of a pose and the PoseTrafo object.

     Uses image aruco detection, target aruco markers and camera calibration data.

    :param data_path: str   -   data path to image directory
    :param aruco_local_points: np.array( [[ar1x, ar1y, ar1z], [ar2x, ar2y, ar2z], ...] )  - object points of the arucos to be used: float, shape(n,3)
    :param target_ar_ids: list  - list of int or number strings with the corresponding aruco marker ids, ordered as the object points
    :param cam_mtx: np.array( [[fx, 0 , cx], [0, fy, cy], [0, 0, 1]] )  - camera matrix
    :param dist_coeff: np.array - distortion coefficients
    :return: Tuple(np.ndarray, PoseTrafo)     - exterior orientations of all states row by row
    """
    data_path = os.path.normpath(data_path)
    # check if object points and aruco ids given have the same length
    if not len(aruco_local_points) == len(target_ar_ids):
        raise ValueError(f'Matching of aruco points and ids failed. Size disparity ({len(target_ar_ids)} ids for {len(aruco_local_points)} points).')

    # find all images in given directory
    file_ext = ['.png', '.jpg']
    file_names = []
    for file in os.listdir(data_path):
        if file[file.rfind('.'):] in file_ext:
            file_names.append(file)

    # for every state/image of a camera detect & match aruco markers in image, prepare input for solvePnP and retrieve exterior orientation per state
    target_ar_ids = list(map(str, target_ar_ids))
    ext_orientations: Union[np.ndarray, None] = None
    ext_trafos: List[PoseTrafo] = []
    for file in file_names:
        # load correct aruco library for reference
        img = cv.imread(os.path.join(data_path, file))
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        aruco_lib = getattr(cv.aruco, f'DICT_{4}X{4}_{250}')
        arucoDict = cv.aruco.Dictionary_get(aruco_lib)
        arucoParam = cv.aruco.DetectorParameters_create()

        # detect aruco markers in the image
        bboxs, detected_ids, rejected = cv.aruco.detectMarkers(gray, arucoDict, parameters=arucoParam)
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        image_points = []
        object_points = []
        if detected_ids is not None:
            detected_ids = detected_ids.squeeze()
            # get image points which appear in target aruco ids
            for target_index, target_id in enumerate(target_ar_ids):
                for det_index, det_id in enumerate(detected_ids):
                    if str(det_id) == target_id:
                        bbox_subpix = cv.cornerSubPix(gray, bboxs[det_index].copy(), (11, 11), (-1, -1), criteria)

                        px = (bbox_subpix[0][0][0] + bbox_subpix[0][1][0] + bbox_subpix[0][2][0] + bbox_subpix[0][3][0]) / 4
                        py = (bbox_subpix[0][0][1] + bbox_subpix[0][1][1] + bbox_subpix[0][2][1] + bbox_subpix[0][3][1]) / 4
                        image_point = [px, py]

                        image_points.append(image_point)
                        object_points.append(aruco_local_points[target_index])
                        break
        aruco_image_points = np.array(image_points, float).reshape(len(image_points), 2)
        aruco_image_points = np.expand_dims(aruco_image_points, axis=0)
        aruco_object_points = np.array(object_points, float).reshape(len(object_points), 3)
        aruco_object_points = np.expand_dims(aruco_object_points, axis=0)

        flags = cv.SOLVEPNP_ITERATIVE
        if len(aruco_object_points[0]) < 6:
            raise ValueError(f'To few aruco markers were detected for processing the exterior orientation. Found {len(aruco_object_points[0])}, need 6.')

        ret, rvec, tvec = cv.solvePnP(aruco_object_points, aruco_image_points, cam_mtx, dist_coeff, flags)

        trafo = PoseTrafo('rodrigues', 'rad', rvec, tvec)
        ext_trafos.append(trafo)
        t_rvec = trafo.get_rvec('eulerxyz', 'deg')
        t_tvec = trafo.tvec
        image_ext_orient = np.vstack((t_tvec, t_rvec)).reshape(1, 6)

        if ext_orientations is None:
            ext_orientations = image_ext_orient.copy()
        else:
            ext_orientations = np.concatenate((ext_orientations, image_ext_orient), axis=0)
    return ext_orientations, ext_trafos


def get_relative_orientation(ext_orientation_K_base: np.ndarray, ext_orientation_Kj: np.ndarray):
    rel_orientation: Union[np.ndarray, None] = None
    # calculate the homogeneous Pose from ext_orientation
    for i in range(len(ext_orientation_Kj)):
        tvecs_k_base = ext_orientation_K_base[i, 0:3]
        rvecs_k_base = ext_orientation_K_base[i, 3:6]
        T_44_k_base = PoseTrafo('eulerxyz', 'deg', rvecs_k_base, tvecs_k_base)


        rvecs_kj = ext_orientation_Kj[i, 0:3]
        tvecs_kj = ext_orientation_Kj[i, 3:6]
        T_44_kj = PoseTrafo('eulerxyz', 'deg', rvecs_kj, tvecs_kj)

        k_j =T_44_kj.T4x4_inv
        k_base = T_44_k_base.T4x4
        relative_Pose =  k_base @ k_j

        relative_orientation = PoseTrafo('t4x4', '', None, None, relative_Pose)

        t_rvec = relative_orientation.get_rvec('eulerxyz', 'deg')
        t_tvec = relative_orientation.tvec
        image_rel_orient = np.vstack((t_tvec, t_rvec)).reshape(1, 6)

        if rel_orientation is None:
            rel_orientation = image_rel_orient.copy()
        else:
            rel_orientation = np.concatenate((rel_orientation, image_rel_orient), axis=0)
    return rel_orientation


class PoseTrafo:
    def __init__(self, type: Union[str, None], rot_unit: str = '', rvec = None, tvec = None, t4x4: np.ndarray = None):
        """Class providing homogeneous 4x4 transformation and decomposing into various descriptive forms.

        Parameter:\n
        - type: str   - form of rvec passed: 'rodrigues', 'eulerxyz', 'eulerzyx' - pass 't4x4' if t4x4 is passed instead
        - rot_unit: str - unit of the passed rvec (euler as well as rodrigues): 'deg' or 'rad'; tvec has to be mm
        - rvec:
        - tvec:
        """

        self.__deg_repr_list = ['°', 'deg', 'degree', 'degrees']
        self.__rad_repr_list = ['rad', 'rads', 'radiant']
        if type is not None:
            if type.lower().strip()[:8] == 'rodrigue':
                self.__type = 0
                if rvec is None or tvec is None:
                    raise Exception('rvec and tvec need to be passed for a rodrigues transformation.')
                if rot_unit.lower().strip() in self.__deg_repr_list:
                    for i, _ in enumerate(rvec):
                        rvec[i] *= np.square(np.pi / 180)                # deg -> rad for under sqrt
                elif rot_unit.lower().strip() not in self.__rad_repr_list:
                    raise ValueError(f'{rot_unit} is not a valid unit for rotations. Use an rodrigues rvec scaled for rad or deg.')
                self.__rvec_rodr = rvec.reshape(3, )
                self.tvec = tvec
                self.__T4x4 = self.build_pose()

            elif type.lower().strip() == 'eulerxyz':
                self.__type = 1
                if rvec is None or tvec is None:
                    raise Exception('rvec and tvec need to be passed for a euler transformation.')
                if rot_unit.lower().strip() in self.__deg_repr_list:
                    for i, _ in enumerate(rvec):
                        rvec[i] *= np.pi / 180                # deg -> rad
                elif rot_unit.lower().strip() not in self.__rad_repr_list:
                    raise ValueError(f'{rot_unit} is not a valid unit for rotations. Use rad or deg.')
                self.__rvec_euler_xyz = rvec.reshape(3, )
                self.tvec = tvec
                self.__T4x4 = self.build_pose()

            elif type.lower().strip() == 'eulerzyx':
                self.__type = 2
                if rvec is None or tvec is None:
                    raise Exception('rvec and tvec need to be passed for a euler transformation.')
                if rot_unit.lower().strip() in self.__deg_repr_list:
                    for i, _ in enumerate(rvec):
                        rvec[i] *= np.pi / 180                # deg -> rad
                elif rot_unit.lower().strip() not in self.__rad_repr_list:
                    raise ValueError(f'{rot_unit} is not a valid unit for rotations. Use rad or deg.')
                self.__rvec_euler_zyx = rvec.reshape(3, )
                self.tvec = tvec
                self.__T4x4 = self.build_pose()

            elif type.lower().strip() == 't4x4':
                self.__type = 3
                if t4x4 is None:
                    raise Exception('T4x4 needs to be passed for to convert it to a PoseTrafo.')
                self.__T4x4 = t4x4

            else:
                raise ValueError(f'{type} is not a valid type for pose transformations.')
            self.decompose_pose()
            self.T4x4_inv = self.inv_pose()

    def __repr__(self):
        return 'PoseTrafo'

    @property
    def T4x4(self):
        """Getter for T4x4 property.

        :return: T4x4   - Returns the T4x4 matrix stored
        """
        return self.__T4x4

    @T4x4.setter
    def T4x4(self, set_t4x4: np.ndarray):
        """Wrapped setter for T4x4 property to ensure consistency of rvec, tvec and T4x4_inv propierties in case of an overwritten T4x4.

        :param set_t4x4:    Argument to set a new T4x4
        """
        if isinstance(set_t4x4, np.ndarray):
            if set_t4x4.shape == (4, 4):
                self.__T4x4 = set_t4x4
                self.decompose_pose()
                self.inv_pose()
        # setter explanation: https://stackoverflow.com/questions/6618002/using-property-versus-getters-and-setters

    def get_rvec(self, type: str, rot_unit: str) -> np.ndarray:
        """Retrieve rodrigues or euler rvec of the transformation in specified unit.

        :param type: str    - specify type of rvec to be retrieved: rodrigues, eulerxyz, eulerzyx
        :param rot_unit: str    - specify unit the rvec is converted to: deg, rad
        """
        if type.lower().strip()[:8] == 'rodrigue':
            rvec = self.__rvec_rodr.copy()
            if rot_unit.lower().strip() in self.__deg_repr_list:
                for i, _ in enumerate(rvec):
                    rvec[i] *= np.square(180 / np.pi)  # rad -> deg for under sqrt
            elif rot_unit.lower().strip() not in self.__rad_repr_list:
                raise ValueError(f'{rot_unit} is not a valid unit for rotations. Can only get rad or deg.')
        elif type.lower().strip() == 'eulerxyz':
            rvec = self.__rvec_euler_xyz.copy()
            if rot_unit.lower().strip() in self.__deg_repr_list:
                for i, _ in enumerate(rvec):
                    rvec[i] *= 180 / np.pi                # rad -> deg
            elif rot_unit.lower().strip() not in self.__rad_repr_list:
                raise ValueError(f'{rot_unit} is not a valid unit for rotations. Use rad or deg.')
        elif type.lower().strip() == 'eulerzyx':
            rvec = self.__rvec_euler_zyx.copy()
            if rot_unit.lower().strip() in self.__deg_repr_list:
                for i, _ in enumerate(rvec):
                    rvec[i] *= np.pi / 180                # rad -> deg
            elif rot_unit.lower().strip() not in self.__rad_repr_list:
                raise ValueError(f'{rot_unit} is not a valid unit for rotations. Use rad or deg.')
        else:
            raise ValueError(f'{type} is not a valid type for rotations.')
        return rvec

    def build_pose(self):
        """build the pose out of the translation and rotation vector"""
        if self.__type == 0:                            # Rodrigues
            rot_matrix, _ = cv.Rodrigues(self.__rvec_rodr)  # the rotation vector must transform to rodrigues matrix
        elif self.__type == 1:                          # Euler XYZ
            rot_matrix = self.rotate_xyz(self.__rvec_euler_xyz)
        elif self.__type == 2:                          # Euler ZYX
            rot_matrix = self.rotate_zyx(self.__rvec_euler_zyx)
        else:
            raise ValueError('Wrong pose transformation type for build_pose().')

        flat_tvec = self.tvec.reshape(3,)
        T4x4 = np.array([[rot_matrix[0][0], rot_matrix[0][1], rot_matrix[0][2], flat_tvec[0]],
                         [rot_matrix[1][0], rot_matrix[1][1], rot_matrix[1][2], flat_tvec[1]],
                         [rot_matrix[2][0], rot_matrix[2][1], rot_matrix[2][2], flat_tvec[2]],
                         [0.0, 0.0, 0.0, 1.0]])
        return T4x4

    def decompose_pose(self):
        """Decompose the pose into the rotation and translation vector"""
        rot_matrix = self.__T4x4[0:3, 0:3]

        rvec_rodr, _ = cv.Rodrigues(rot_matrix)  # transform the rodrigues matrix to rotation vector
        self.__rvec_rodr = rvec_rodr

        rx, ry, rz = self.get_t4x4_angles(self.__T4x4)
        self.__rvec_euler_xyz = np.array([[rx], [ry], [rz]], float)
        self.__rvec_euler_zyx = np.array([[rz], [ry], [rx]], float)

        tx = self.__T4x4[0, 3]
        ty = self.__T4x4[1, 3]
        tz = self.__T4x4[2, 3]

        self.tvec = np.array([[tx], [ty], [tz]], float)

    def inv_pose(self):
        rot_matrix = self.__T4x4[0:3, 0:3]
        rot_matrix_transp = np.transpose(rot_matrix)

        Xi = rot_matrix_transp.dot(self.tvec)
        pose_inv = np.array([[rot_matrix_transp[0, 0], rot_matrix_transp[0, 1], rot_matrix_transp[0, 2], -1 * float(Xi[0])],
                             [rot_matrix_transp[1, 0], rot_matrix_transp[1, 1], rot_matrix_transp[1, 2], -1 * float(Xi[1])],
                             [rot_matrix_transp[2, 0], rot_matrix_transp[2, 1], rot_matrix_transp[2, 2], -1 * float(Xi[2])],
                             [0, 0, 0, 1]])

        self.T4x4_inv = pose_inv
        return self.T4x4_inv

    # helper functions -------------------------------------------------------------------------------------------------------------------------------
    def get_x_rotation_matrix(self, angle_rad: float) -> np.ndarray:
        """Returns the homogenous rotation matrix by x with a"""
        sa = np.sin(angle_rad)
        ca = np.cos(angle_rad)
        return np.array(
            [[1.0, 0.0, 0.0, 0.0],
             [0.0, ca, -sa, 0.0],
             [0.0, sa, ca, 0.0],
             [0.0, 0.0, 0.0, 1.0]])

    def get_y_rotation_matrix(self, angle_rad: float) -> np.ndarray:
        """Returns the homogenous rotation matrix by y with b"""
        sb = np.sin(angle_rad)
        cb = np.cos(angle_rad)
        return np.array(
            [[cb, 0.0, sb, 0.0],
             [0.0, 1.0, 0.0, 0.0],
             [-sb, 0.0, cb, 0.0],
             [0.0, 0.0, 0.0, 1.0]])

    def get_z_rotation_matrix(self, angle_rad: float) -> np.ndarray:
        """Returns the homogenous rotation matrix by z with c"""
        sc = np.sin(angle_rad)
        cc = np.cos(angle_rad)
        return np.array(
            [[cc, -sc, 0.0, 0.0],
             [sc, cc, 0.0, 0.0],
             [0.0, 0.0, 1.0, 0.0],
             [0.0, 0.0, 0.0, 1.0]])

    def rotate_zyx(self, rvec_euler_zyx: np.ndarray) -> np.ndarray:
        """Returns the homogenous rotation matrix by z, y, x"""
        angle_z_rad: float = rvec_euler_zyx[0]
        angle_y_rad: float = rvec_euler_zyx[1]
        angle_x_rad: float = rvec_euler_zyx[2]
        rot_matrix = self.get_x_rotation_matrix(angle_x_rad) @ self.get_y_rotation_matrix(angle_y_rad) @ self.get_z_rotation_matrix(angle_z_rad)
        return rot_matrix

    def rotate_xyz(self, rvec_euler_xyz: np.ndarray) -> np.ndarray:
        """Returns the homogenous rotation matrix by x, y, z"""
        angle_x_rad: float = rvec_euler_xyz[0]
        angle_y_rad: float = rvec_euler_xyz[1]
        angle_z_rad: float = rvec_euler_xyz[2]
        rx = self.get_x_rotation_matrix(angle_x_rad)
        ry = self.get_y_rotation_matrix(angle_y_rad)
        rz = self.get_z_rotation_matrix(angle_z_rad)
        rot_matrix = rz @ ry @ rx
        return rot_matrix

    def get_t4x4_angles(self, t4x4):
        """Returns the angles A, B and C by the consecutive axis x, y and z for the frame T

        :param t4x4: frame
        :type t4x4: np.ndarray[4,4]
        :return: angles A, B and C
        :rtype: numbers
        """
        if np.abs(t4x4[2, 0]) != 1:
            ry = -np.arcsin(t4x4[2, 0])
            rx = np.arctan2(t4x4[2, 1] / np.cos(ry), t4x4[2, 2] / np.cos(ry))
            rz = np.arctan2(t4x4[1, 0] / np.cos(ry), t4x4[0, 0] / np.cos(ry))
        else:
            rz = 0
            if t4x4[2, 0] == -1:
                ry = np.pi / 2
                rx = rz + np.arctan2(t4x4[0, 1], t4x4[0, 2])
            else:
                ry = -np.pi / 2
                rx = -rz + np.arctan2(-t4x4[0, 1], -t4x4[0, 2])
        return rx, ry, rz


# ==========================================================================================================================================
if __name__ == '__main__':
    test_rvec = np.array([30,40,50], float)
    test_tvec = np.array([10,10,10], float)
    trafo = PoseTrafo('eulerxyz', 'deg', test_rvec, test_tvec)
    deg = trafo.get_rvec('eulerxyz', 'deg')
    rad = trafo.get_rvec('eulerxyz', 'rad')
    pass