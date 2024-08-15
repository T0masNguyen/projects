import warnings
import cv2 as cv
import numpy as np
from enum import Enum
from typing import Union, Tuple, List


# ====================================================================================================================================================
class ArucoParaSet(Enum):
    DEFAULT = 1
    LARGE_DIST = 2


# ------------------------------------------------------------------------------------------------------------------------------------------------
def get_para_set(enum_entry: ArucoParaSet, print_para: bool = False) -> cv.aruco_DetectorParameters:
    """Contains aruco parameter sets which can be imported to be used for OpenCV aruco detection.

    Add other detailed aruco parameter sets inside this method.
    """
    if enum_entry == ArucoParaSet.DEFAULT:
        aruco_para = cv.aruco.DetectorParameters_create()
    elif enum_entry == ArucoParaSet.LARGE_DIST:
        aruco_para = cv.aruco.DetectorParameters_create()
        aruco_para.adaptiveThreshConstant = 11
        aruco_para.minMarkerPerimeterRate = 0.008
        aruco_para.maxMarkerPerimeterRate = 0.6
        aruco_para.polygonalApproxAccuracyRate = 0.1
        aruco_para.minCornerDistanceRate = 0.005
        aruco_para.minMarkerDistanceRate = 0.01
        aruco_para.perspectiveRemoveIgnoredMarginPerCell = 0.15
    else:
        raise ValueError('Unknown aruco parameter set enum entry passed. Abort')

    # optional printing -----------------------------------------------------
    if print_para:
        attributes = [attr for attr in dir(aruco_para)
                      if not attr.startswith('__') and attr != 'create']
        print(f'Aruco detection parameters of parameter set "{enum_entry.name}:')
        for attr_str in attributes:
            attr = getattr(aruco_para, attr_str)
            print(f'\t{attr_str}:\t{attr}')

    return aruco_para


# ====================================================================================================================================================
class ArucoDetection:
    def __init__(self, aruco_dict_name: str = 'DICT_4X4_250',
                 aruco_para: ArucoParaSet = ArucoParaSet.DEFAULT):
        # initialize aruco library and parameters
        self._aruco_dict: cv.aruco_Dictionary = self.__get_detection_dict(aruco_dict_name)
        self._aruco_para: cv.aruco_DetectorParameters = get_para_set(aruco_para)

    @property
    def aruco_dict(self):
        return self._aruco_dict

    @aruco_dict.setter
    def aruco_dict(self, new_name: str):
        if isinstance(new_name, str):
            self._aruco_dict: cv.aruco_Dictionary = self.__get_detection_dict(new_name)

    # ------------------------------------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def __get_detection_dict(dict_str: str) -> cv.aruco_Dictionary:
        error = False
        if not dict_str.isupper():
            warnings.warn(f'Input correction recommended: String for aruco dict "{dict_str}" isn\'t '
                          f'all uppercase as expected ("DICT_<i>X<i>_<size>").', stacklevel=2)
        ds = dict_str.upper()
        parts = ds.split('_')
        if len(parts) == 3:
            has_prefix = parts[0] == 'DICT'
            mp = parts[1].split('X')
            bit_match = len(mp) == 2 and mp[0].isnumeric() and mp[0] == mp[1]
            has_dict_size = parts[2].isnumeric()
            if not (has_prefix and bit_match and has_dict_size):
                error = True
        else:
            error = True
        if error:
            raise AttributeError(f'String for aruco dict "{dict_str}" doesn\'t match the required pattern "DICT_<i>X<i>_<size>" (case sensitive).')
        key = getattr(cv.aruco, ds)
        aruco_dict = cv.aruco.Dictionary_get(key)

        return aruco_dict

    # ------------------------------------------------------------------------------------------------------------------------------------------------
    def  detect_markers(self, draw_img, image: np.ndarray, do_subpix: bool = False, draw_markers: bool = False) -> Union[dict, None]:
        """Detects aruco markers visible in the passed image.

        Subpixel refinement is optional.

        Output dictionary has keys... \n
        "_meta":
            "_marker_type": str, \n
            "_detected_ids": List[int], \n
            "_rejected_bboxs": Tuple[np.ndarray(1,4,2)] \n
        "data":
            "<marker_id>" one for each detected marker id, whith subkeys
                "center_pt": [<pixel_x>, <pixel_y>] and \n
                "bbox_pt_1" to "bbox_pt_4" (same value format) \n
                "bbox_array": complete array of marker bbox.

        :param draw_markers: Draws detected marker bounding boxes and ids into input image if True. Default: False.
        :param image: loaded image which is subject for aruco detection.
        :param do_subpix: if False, raw aruco detection results are returned, if True, subpix refinement is done for every detected aruco marker. Default: False.
        :return: Dictionary with aruco marker data of the image as explained above, None if no aruco markers detected.
        """
        # --------------------------------------------------------------------------------------- convert bgr to to gray
        if len(image.shape) == 3:
            img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        else:
            img = image

        # --------------------------------------------------------------------------------------------------------------
        bboxs, ids, rej_bboxs = cv.aruco.detectMarkers(img, self._aruco_dict, parameters=self._aruco_para)


        if isinstance(ids, np.ndarray) and ids.size > 1:                         # ids should be a simple list of aruco ids
            ids = ids.squeeze()
            marker_dict = {
                '_meta': {
                    '_marker_type': 'aruco',
                    '_detected_ids': ids.tolist(),
                    '_rejected_bboxs': rej_bboxs
                },
                'data': {}
            }

            for idx, ar_id in enumerate(ids.tolist()):
                ar_bbox = bboxs[idx][0]
                # store point data of all aruco markers detected in the image
                cntr_pt = [(ar_bbox[0][0] + ar_bbox[1][0] + ar_bbox[2][0] + ar_bbox[3][0]) / 4,
                           (ar_bbox[0][1] + ar_bbox[1][1] + ar_bbox[2][1] + ar_bbox[3][1]) / 4]
                marker_dict['data'].update({
                    str(ar_id): {
                        'center_pt': cntr_pt,
                        'bbox_array': ar_bbox,
                        'bbox_pt_1': ar_bbox[0],
                        'bbox_pt_2': ar_bbox[1],
                        'bbox_pt_3': ar_bbox[2],
                        'bbox_pt_4': ar_bbox[3]
                    }
                })
                if do_subpix:
                    marker_dict['data'].update({str(ar_id): self.refine_marker_subpix(img, ar_bbox)})
            if draw_markers:
                cv.aruco.drawDetectedMarkers(draw_img, bboxs, ids)
        else:
            marker_dict = {
                '_meta': {
                    '_marker_type': 'aruco',
                    '_detected_ids': [],
                    '_rejected_bboxs': rej_bboxs
                },
                'data': {}
            }

        return marker_dict

    # ------------------------------------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def refine_marker_subpix(image: np.ndarray, ar_bbox: np.ndarray) -> dict:
        """Does a subpixel refinement on the passed raw marker bbox and returns updated marker dict part.

        :param image: input OpenCV image, will be converted to grayscale if not already converted.
        :param ar_bbox: "raw" bbox of a single marker detected by OpenCV Aruco detection. Expected shape: (4,2).
        :returns: Dict with the updated content to replace the "raw" marker data of a aruco id in the marker dict (center_pt, bbox_array, bbox_pt_x).
        """

        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        if len(image.shape) == 3:
            img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        else:
            img = image

        bbox_subpix = cv.cornerSubPix(img, ar_bbox, (11, 11), (-1, -1), criteria)
        # store point data of all aruco markers detected in the image
        cntr_pt = [(bbox_subpix[0][0] + bbox_subpix[1][0] + bbox_subpix[2][0] + bbox_subpix[3][0]) / 4,     # first [0] to ignore irrelevant first layer of bbox
                   (bbox_subpix[0][1] + bbox_subpix[1][1] + bbox_subpix[2][1] + bbox_subpix[3][1]) / 4]
        updated_marker = {
            'center_pt': cntr_pt,
            'bbox_array': bbox_subpix,
            'bbox_pt_1': bbox_subpix[0],
            'bbox_pt_2': bbox_subpix[1],
            'bbox_pt_3': bbox_subpix[2],
            'bbox_pt_4': bbox_subpix[3]
        }

        return updated_marker

    # ------------------------------------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def marker_dict_to_numpy_arrays(marker_dict: dict, incl_id_list: List[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Converts specified aruco ids in a marker dict into synced arrays of aruco ids and aruco center points.

        (Optional) Passed aruco ids not in the marker dict will be skipped.

        :param marker_dict: formatted data input with ["center_pt"] marker center points for each aruco id.
        :param incl_id_list: (optional) all aruco ids whose points should be included in the returned arrays. If None or empty, all ids are considered. Default: None.
        :return: array with the x available input aruco ids (shape x,) and an array (shape x,2) of the corresponding center points.
        """

        incl_id_list = list(map(str, incl_id_list)) if incl_id_list and isinstance(incl_id_list, list) else None   # str conversion for all ids in list
        ar_id_array = []
        ar_point_array = []
        for m_key, m_val in marker_dict['data'].items():
            if incl_id_list and m_key not in incl_id_list:
                continue
            try:
                if (isinstance(m_val['center_pt'], list) and
                        len(m_val['center_pt']) == 2 and
                        all(map(lambda f: isinstance(f, float), m_val['center_pt']))):
                    ar_point_array.append(m_val['center_pt'])
                else:
                    raise TypeError('"center_pt" value for an id in marker dict is expected be a list of exactly 2 floats.')
            except KeyError as k:
                raise KeyError(f'Aruco marker dictionary doesn\'t have the expected structure for aruco id {m_key}.') from k
            ar_id_array.append(m_key)
        ar_id_array = np.array(ar_id_array, int)
        ar_point_array = np.array(ar_point_array, float)

        return ar_id_array, ar_point_array


if __name__ == '__main__':
    print("unit for aruco detection \n \t -> run facial_landmarks_pose_detection.py  \n \t    with option measure_aruco=True")

