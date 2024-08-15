import warnings
import numpy as np
import json
import uuid
from typing import List, Tuple, Union, Dict

"""
.as .... 
"""


class CameraResolution:
    def __init__(self, x: int = 1280, y: int = 720):
        self.x: int = x
        self.y: int = y


class ExtrinsicPara:
    """
    extrinsic orientation of a camera
    """
    def __init__(self):
        self.x: float = 0.0
        self.y: float = 0.0
        self.z: float = 0.0
        self.rx: float = 0.0
        self.ry: float = 0.0
        self.rz: float = 0.0

    def __str__(self):
        extrinsic_string = 'extrinsic parameter:\t' + \
                       'x: ' + str(self.x) + ' |\t' + \
                       'y: ' + str(self.y) + ' |\t' + \
                       'z: ' + str(self.z) + ' |\t' + \
                       'rx: ' + str(self.rx) + ' |\t' + \
                       'ry: ' + str(self.ry) + ' |\t' + \
                       'rz: ' + str(self.rz)
        return extrinsic_string

    @property
    def vector(self):
        return np.array([self.x, self.y, self.z, self.rx, self.ry, self.rz], float)

    @vector.setter
    def vector(self, new_vector: Union[List[float], np.ndarray]):
        array_condition = isinstance(new_vector, np.ndarray) and new_vector.shape == (6,)
        list_condition = all(map(lambda e: isinstance(e, float), new_vector))
        if array_condition or list_condition:
            self.x = new_vector[0]
            self.y = new_vector[1]
            self.z = new_vector[2]
            self.rx = new_vector[3]
            self.ry = new_vector[4]
            self.rz = new_vector[5]


class InnerCoeff:
    def __init__(self):
        self.type_calibration: bool = False
        self.indiv_calibration: bool = False

        self.fx: float = 0.0
        self.fy: float = 0.0
        self.cx: float = 0.0
        self.cy: float = 0.0

    # Inner orientation methods -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def get_camera_matrix(self):
        """Gets the CV camera matrix as an np.ndarray.

        :return: cam_matrix = ndarray[ [fx, 0, cx], [0, fy, cy], [0, 0, 1] ]
        """
        cam_matrix = np.array([[self.fx, 0, self.cx],
                               [0, self.fy, self.cy],
                               [0, 0, 1]], float)
        return cam_matrix

    def __str__(self):
        coeff_string = 'inner camera coefficients:\t' + \
                       'fx: ' + str(self.fy) + ' |\t' + \
                       'fy: ' + str(self.fy) + ' |\t' + \
                       'cx: ' + str(self.cx) + ' |\t' + \
                       'cy: ' + str(self.cy)
        return coeff_string


class DistCoeff:
    def __init__(self):
        self.type_calibration: bool = False
        self.indiv_calibration: bool = False

        self.k1: float = 0.0
        self.k2: float = 0.0
        self.p1: float = 0.0
        self.p2: float = 0.0
        self.k3: float = 0.0
        self.k4: float = 0.0
        self.k5: float = 0.0
        self.k6: float = 0.0
        self.s1: float = 0.0
        self.s2: float = 0.0
        self.s3: float = 0.0
        self.s4: float = 0.0
        self.t1: float = 0.0
        self.t2: float = 0.0

    # ------------------------------------------------------------------------------------------------------------------
    def get_dist_coeff_values(self) -> np.ndarray:
        """Returns the values of the distortion coefficient as an np.ndarray.

        :return: coeff_values = ndarray[k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4, t1, t2]
        """
        coeff_values = np.array([self.k1, self.k2,
                                 self.p1, self.p2,
                                 self.k3, self.k4, self.k5, self.k6,
                                 self.s1, self.s2, self.s3, self.s4,
                                 self.t1, self.t2], float)

        return coeff_values

    def __str__(self):
        coeff_string = 'distortion coefficients:\t' + \
                       'k1: ' + str(self.k1) + ' |\t' + \
                       'k2: ' + str(self.k2) + ' |\t' + \
                       'p1: ' + str(self.p1) + ' |\t' + \
                       'p2: ' + str(self.p2) + ' |\t' + \
                       'k3: ' + str(self.k3) + ' |\t' + \
                       'k4: ' + str(self.k4) + ' |\t' + \
                       'k5: ' + str(self.k5) + ' |\t' + \
                       'k6: ' + str(self.k6) + ' |\t' + \
                       's1: ' + str(self.s1) + ' |\t' + \
                       's2: ' + str(self.s2) + ' |\t' + \
                       's3: ' + str(self.s3) + ' |\t' + \
                       's4: ' + str(self.s4) + ' |\t' + \
                       't1: ' + str(self.t1) + ' |\t' + \
                       't2: ' + str(self.t2)
        return coeff_string

class CameraParameters:
    """
    class to contain all necessary camera and stream information
    """
    def __init__(self):
        # ----------------------------------------------------------
        self.camera_type_name: str = ''
        self.u_id: uuid = None
        self.stream_id: int = -1
        # -------------------------------------------------------
        self.resolution: CameraResolution = CameraResolution()
        self.calib_resolution: CameraResolution = CameraResolution()
        # -----------------------------------------------------------------------
        self.inner_coeff: InnerCoeff = InnerCoeff()
        self.dist_coeff: DistCoeff = DistCoeff()
        self.extrinsic: ExtrinsicPara = ExtrinsicPara()


class CameraDict:
    """
    Alternative to CameraList to experiment with easier two-way identification of specific camera in camera_calibration.py.
    """
    def __init__(self):
        self.devices: Dict[uuid.UUID: CameraParameters] = {}


def read_cam_calib_json(cam_identifier: Union[str, uuid.UUID], json_file_path,
                        default_calib: bool = False,
                        read_extrinsic_parameter: bool = True)\
        -> Tuple[bool, CameraParameters]:
    """Gets a specific camera key (camera type name or camera individual uuid) from a camera calibration json-file parsed.

    :param cam_identifier: Either generic name of the camera type or a camera individual uuid. (Identifier is searched in subkeys of the json)
    :param json_file_path: Path to camera calibration json-file.
    :param default_calib: if True, uuid will be ignored and None returned.
    :param read_extrinsic_parameter: If True, extrinsic parameters will be extracted from file, else it returns a default ExtrinsicPara instance.
    :return: Sequence of parametrized object instances if cam_identifier was found or default object instances else.
    """

    cam = CameraParameters()
    cam_identifier = str(cam_identifier)

    with open(json_file_path) as file:
        data = json.load(file)

    # TODO: check if all cameras are in the calibration files
    #       -> part one in type calibration
    #       -> part two in system calibration
    
    if cam_identifier in data.keys():
        found_cam_identifier = True
        # -------------------------------------------------------------------------------- calibration resolution
        try:
            cam.resolution.x = data[cam_identifier]['calibration resolution']['x']
            cam.resolution.y = data[cam_identifier]['calibration resolution']['y']
        except KeyError:
            warnings.warn(f'\nNo valid resolution found for "{cam_identifier}" in "{json_file_path}".\n'
                          f'Used default resolution of {cam.resolution.x} by {cam.resolution.y}px.', stacklevel=2)

        # -------------------------------------------------------------------------------------------------- uuid
        if not default_calib:
            try:
                cam.u_id = uuid.UUID(data[str(cam_identifier)]['uuid']) # TODO: falsch hier !
            except (KeyError, ValueError):
                warnings.warn(f'\nNo valid uuid found for "{cam_identifier}" in "{json_file_path}".\n'
                              f'New uuid issued.', stacklevel=2)

        # -------------------------------------------------------------------------------------- camera parameter
        cam.inner_coeff.fx = data[cam_identifier]['intrinsic_values']['fx']
        cam.inner_coeff.fy = data[cam_identifier]['intrinsic_values']['fy']
        cam.inner_coeff.cx = data[cam_identifier]['intrinsic_values']['cx']
        cam.inner_coeff.cy = data[cam_identifier]['intrinsic_values']['cy']
        # ------------------------------------------------------------------------------- distortion coefficients
        cam.dist_coeff.k1 = data[cam_identifier]['distortion_coefficients']['k1']
        cam.dist_coeff.k2 = data[cam_identifier]['distortion_coefficients']['k2']
        cam.dist_coeff.p1 = data[cam_identifier]['distortion_coefficients']['p1']
        cam.dist_coeff.p2 = data[cam_identifier]['distortion_coefficients']['p2']
        cam.dist_coeff.k3 = data[cam_identifier]['distortion_coefficients']['k3']
        # ----------------------------------------------------------------------------------- extrinsic parameter
        if read_extrinsic_parameter:
            cam.extrinsic.x = data[cam_identifier]['extrinsic_values']['x']
            cam.extrinsic.y = data[cam_identifier]['extrinsic_values']['y']
            cam.extrinsic.z = data[cam_identifier]['extrinsic_values']['z']
            cam.extrinsic.rx = data[cam_identifier]['extrinsic_values']['rx']
            cam.extrinsic.ry = data[cam_identifier]['extrinsic_values']['ry']
            cam.extrinsic.rz = data[cam_identifier]['extrinsic_values']['rz']

    else:
        warnings.warn(f'\nCam identifier "{cam_identifier}" is not in "{json_file_path}".\n'
                      f'Default parameters used instead.', stacklevel=2)

    #---------------------new Anne
        found_cam_identifier = True

        # -------------------------------------------------------------------------------- calibration resolution
        try:
            cam.resolution.x = data[str(cam_identifier)]['calibration resolution']['x']
            cam.resolution.y = data[str(cam_identifier)]['calibration resolution']['y']
        except KeyError:
            warnings.warn(f'\nNo valid resolution found for "{cam_identifier}" in "{json_file_path}".\n'
                          f'Used default resolution of {cam.resolution.x} by {cam.resolution.y}px.', stacklevel=2)

        # -------------------------------------------------------------------------------------------------- uuid
        if not default_calib:
            try:
                cam.u_id = uuid.UUID(data[str(cam_identifier)]['uuid']) # TODO: without UUID by default
            except (KeyError, ValueError):
                warnings.warn(f'\nNo valid uuid found for "{cam_identifier}" in "{json_file_path}".\n'
                              f'New uuid issued.', stacklevel=2)

        # -------------------------------------------------------------------------------------- camera parameter
        cam.inner_coeff.fx = data[str(cam_identifier)]['intrinsic_values']['fx']
        cam.inner_coeff.fy = data[str(cam_identifier)]['intrinsic_values']['fy']
        cam.inner_coeff.cx = data[str(cam_identifier)]['intrinsic_values']['cx']
        cam.inner_coeff.cy = data[str(cam_identifier)]['intrinsic_values']['cy']
        # ------------------------------------------------------------------------------- distortion coefficients
        cam.dist_coeff.k1 = data[str(cam_identifier)]['distortion_coefficients']['k1']
        cam.dist_coeff.k2 = data[str(cam_identifier)]['distortion_coefficients']['k2']
        cam.dist_coeff.p1 = data[str(cam_identifier)]['distortion_coefficients']['p1']
        cam.dist_coeff.p2 = data[str(cam_identifier)]['distortion_coefficients']['p2']
        cam.dist_coeff.k3 = data[str(cam_identifier)]['distortion_coefficients']['k3']
        # ----------------------------------------------------------------------------------- extrinsic parameter
        if read_extrinsic_parameter:
            cam.extrinsic.x = data[str(cam_identifier)]['extrinsic_values']['x']
            cam.extrinsic.y = data[str(cam_identifier)]['extrinsic_values']['y']
            cam.extrinsic.z = data[str(cam_identifier)]['extrinsic_values']['z']
            cam.extrinsic.rx = data[str(cam_identifier)]['extrinsic_values']['rx']
            cam.extrinsic.ry = data[str(cam_identifier)]['extrinsic_values']['ry']
            cam.extrinsic.rz = data[str(cam_identifier)]['extrinsic_values']['rz']

#----------------new Anne end


    return found_cam_identifier, cam


def write_cam_calib_json(cam_name, jsonFile, mtx, dist):
    # add class for parameter?
    # class from geometric_model.py from OCVExercisesCalib-main/src/ (Jana Kramp)
    fx = mtx[0][0]
    fy = mtx[1][1]
    cx = mtx[0][2]
    cy = mtx[1][2]
    k1 = dist[0][0]
    k2 = dist[0][1]
    p1 = dist[0][2]
    p2 = dist[0][3]
    k3 = dist[0][4]

    aDict = {cam_name: {"camera system": "base_frcam1", "extrinsic_ref_system": "world",
                        "intrinsic_values": {"fx": fx, "fy": fy,
                                             "cx": cx,
                                             "cy": cy,
                                             "fx_std": 0.0,
                                             "fy_std": 0.0,
                                             "cx_std": 0.0,
                                             "cy_std": 0.0},
                        "distortion_coefficients": {
                            "k1": k1,
                            "k2": k2,
                            "p1": p1,
                            "p2": p2,
                            "k3": k3,
                            "k1_std": 0.0,
                            "k2_std": 0.0,
                            "p1_std": 0.0,
                            "p2_std": 0.0,
                            "k3_std": 0.0},
                        "extrinsic_values": {
                            "x": 0.0,
                            "y": 0.0,
                            "z": 0.0,
                            "rx": 0.0,
                            "ry": 0.0,
                            "rz": 0.0,
                            "x_std": 0.0,
                            "y_std": 0.0,
                            "z_std": 0.0,
                            "rx_std": 0.0,
                            "ry_std": 0.0,
                            "rz_std": 0.0
                        }}}

    json_object = json.dumps(aDict, indent=4)

    # Writing to sample.json
    with open(jsonFile, "w") as outfile:
        outfile.write(json_object)
