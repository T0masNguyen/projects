import paths
import threading
import time
import numpy as np
from typing import Union, List, Tuple
import uuid
# ---------------------------------------------------------------------------------------------------------------- tools
from Tools.camera_utils import CameraResolution, CameraDict, CameraParameters
from Tools.camera_assignment import filter_connected_cameras, get_camera_assignment, undistort_image
# ------------------------------------------------------------------------------------------------------ image operators
from Tools.scatter_plot import scatter_plot
from Playground.plot import plot_3d


import mediapipe as mp
import cv2 as cv
from Measurement_operators.high_precision_targets import findHighPrecisionTargets
from Measurement_operators.chessboard_detection import findChessboardCorner

from Measurement_operators.aruco_detection import ArucoDetection
from Measurement_operators.aruco_detection import ArucoParaSet

from Measurement_operators.mediapipe_facial_landmarks import findFacialLandmarks
from Measurement_operators.mediapipe_pose_detection import findPose
from Measurement_operators.mediapipe_hand_detection import findHands

# ====================================================================================================================================================
class CameraThreadHandler:
    def __init__(self, cam_type_whitelist: List[str], resolution: Tuple[int, int] = (1280, 720), manual_inp_control: bool = False,
                 system_calibration_file: str = "",
                 measure_facial_landmarks: bool = True, measure_aruco: bool = True, measure_precision_targets: bool = False,
                 measure_pose: bool = False, measure_hands: bool = False, measure_chessboard=False):
        """

        :param cam_type_whitelist:
        :param resolution:
        :param manual_inp_control:
        """
        # --------------------------------------------------------------------------------------------------------------
        self.resolution: CameraResolution = CameraResolution(resolution[0], resolution[1])      # Auflösungen:  320, 180, 640, 360, 1280, 720, 1920, 1080, 2560, 1440
        self.cam_type_whitelist: List[str] = self.__init_whitelist(cam_type_whitelist)
        self.camera_dict: CameraDict = CameraDict()
        self.system_calibration_file = system_calibration_file
        # ---------------------------------------------------------------------------------------------
        self.aruco_dict: str = 'DICT_4X4_250'
        self.measure_facial_landmarks = measure_facial_landmarks
        self.measure_aruco = measure_aruco
        self.measure_precision_targets = measure_precision_targets
        self.measure_pose = measure_pose
        self.measure_hands = measure_hands
        self.measure_chessboard = measure_chessboard
        # --------------------------------------------------------------------------------------------------------------
        self.thread_lock = threading.Lock()
        self.calculation_barrier: threading.Barrier = None      # barrier thread to collect iteration camThread states for calculation main thread

        self.__inp_enbld = True if isinstance(manual_inp_control, bool) and manual_inp_control else False
        self.input_thread: InputThread = None                   # to catch the input from console
        self.thread_list: List[CameraThread] = []

    # ------------------------------------------------------------------------------------------------------------------------------------------------
    @classmethod
    def __init_whitelist(cls, cam_name_whitelist: List[str]) -> List[str]:
        if isinstance(cam_name_whitelist, list) and all(map(lambda x: isinstance(x, str), cam_name_whitelist)):
            whitelist = cam_name_whitelist
        else:
            raise TypeError('Expected list of strings for camera type whitelist.')

        return whitelist

    # ------------------------------------------------------------------------------------------------------------------------------------------------
    def startup_threads(self):
        global thread_mode  # INTER-THREAD STATE COMMUNICATION

        self.camera_dict = self.get_and_assign_cameras()   # -> all connected cameras identified and matched with stream id, TODO: last bit of matching
        self.thread_list.clear()
        # Create barrier threads used for synchronising cycles between threads
        self.calculation_barrier = threading.Barrier(len(self.camera_dict.devices) + 1 )
        # Create threads
        self.create_threads()
        # Wait on init for all
        self.calculation_barrier.wait()

    # ------------------------------------------------------------------------------------------------------------------------------------------------
    def create_threads(self):
        # TODO: matplot is not thread save!!!
        # print('Initiate scatter_plot thread:')
        # coords = []
        # aruco = []
        # self.plot= plot_3d()
        # self.plot.start()

        print('Initiate camera threads:')
        for cam in self.camera_dict.devices.values():
            thread = CameraThread(cam,
                                  self.calculation_barrier,
                                  self.thread_lock,
                                  measure_facial_landmarks=self.measure_facial_landmarks,
                                  measure_aruco=self.measure_aruco,
                                  measure_chessboard=self.measure_chessboard,
                                  measure_precision_targets=self.measure_precision_targets,
                                  measure_pose=self.measure_pose,
                                  measure_hands=self.measure_hands)
            # ------------------------------------------------------------------------ set parameter for aruco detection
            thread.aruco_detection.aruco_dict = self.aruco_dict
            thread.aruco_detection.aruco_para = ArucoParaSet.DEFAULT
            # ------------------------------------------------------------------------ start threads
            thread.start()
            print("\tRun camera thread with ID: " + str(thread.thread_id))
            self.thread_list.append(thread)
        if self.__inp_enbld:
            self.create_input_thread()

    # ------------------------------------------------------------------------------------------------------------------------------------------------
    def create_input_thread(self):
        self.input_thread = InputThread(self.calculation_barrier)
        self.input_thread.start()
        print("\tRun input thread")

    # ------------------------------------------------------------------------------------------------------------------------------------------------
    def get_and_assign_cameras(self) -> CameraDict:
        """get camera information and filter it"""
        filtered_cam_dict = filter_connected_cameras(self.cam_type_whitelist, self.resolution)
        filtered_cam_dict = get_camera_assignment(filtered_cam_dict,
                                                  paths.TYPE_CALIB_FILE.path,
                                                  self.system_calibration_file)
        if not filtered_cam_dict.devices:
            raise IOError('No whitelisted camera type connected.')
        # TODO: get full cam identification in filtered_cam_dict, too

        return filtered_cam_dict

    # ------------------------------------------------------------------------------------------------------------------------------------------------
    def toggle_input(self, active: bool):
        """Enables/Disables InputThread and therefore controls console input for image set grabbing and camThread status.

        :param active: If True, InputThread runs normally and allows image grabbing process with stream freezing and confirmation via console. If False, Input Thread idles"""
        self.input_thread.is_active = active


# ====================================================================================================================================================
class CameraThread(threading.Thread):
    def __init__(self, cam_calib_para: CameraParameters, calculation_barrier: threading.Barrier, thread_lock: threading.Lock,
                 measure_aruco: bool = True, measure_precision_targets: bool = False, measure_facial_landmarks: bool = True,
                 measure_pose: bool = True, measure_hands: bool = True, measure_chessboard: bool = True):
        """
        base class for a thread to measure in a opencv image stream

        :param cam_calib_para:
        :param calculation_barrier:
        :param measure_aruco:
        :param measure_precision_targets:
        :param measure_facial_landmarks:
        """

        threading.Thread.__init__(self)
        self.thread_lock = thread_lock
        self.thread_id = cam_calib_para.stream_id
        self.calculation_barrier = calculation_barrier

        self.preview_name: str = "Camera " + cam_calib_para.camera_type_name + '_' + str(cam_calib_para.stream_id)
        self.cam_stream_id: int = cam_calib_para.stream_id
        self.cam_name: str = cam_calib_para.camera_type_name
        self.cam_uuid: uuid.UUID = cam_calib_para.u_id
        self.stream_obj: cv.VideoCapture = None
        # --------------------------------------------------------------------------------------------------------------------------------------------
        self.cam_calib_para: CameraParameters = cam_calib_para
        # ---------------------------------------------------------------------------------------------------------------------------- aruco detection
        self.measure_aruco: bool = measure_aruco
        self.aruco_detection: ArucoDetection = ArucoDetection(aruco_para=ArucoParaSet.LARGE_DIST)
        self.results_aruco = []
        # --------------------------------------------------------------------------------------------------
        self.measure_chessboard: bool = measure_chessboard
        self.chessboard_para: list = []
        self.results_chessboard = []
        # -------------------------------------------------------------------------------------------------------------------------- precision targets
        self.measure_precision_targets: bool = measure_precision_targets
        self.results_precision_targets = []
        # --------------------------------------------------------------------------------------------------------------------------- facial landmarks
        self.measure_facial_landmarks: bool = measure_facial_landmarks
        self.facial_lm_para: list = []
        self.results_facial_landmarks = []
        # ------------------------------------------------------------------------------------------------------------------- mediapipe pose detection
        self.measure_pose: bool = measure_pose
        self.pose_para: list = []
        self.results_pose = []
        # ------------------------------------------------------------------------------------------------------------------- mediapipe pose detection
        self.measure_hands: bool = measure_hands
        self.hand_para: list = []
        self.results_hand = []
        # ----------------------------------------------------------------------------------------------------------------------3D coords of Mediapipe
        self.results_facial_landmarks_3D = []

    def run(self):
        self.init_stream()
        self.cam_stream()

    def init_stream(self):
        # ------------------------------------------------------------------------------------------------------------------------------- init streams
        cv.namedWindow(self.preview_name)
        cap = cv.VideoCapture(self.cam_calib_para.stream_id, cv.CAP_DSHOW)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, self.cam_calib_para.resolution.x)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.cam_calib_para.resolution.y)
        real_resolution = CameraResolution(x=cap.get(3), y=cap.get(4))
        if real_resolution.x == self.cam_calib_para.resolution.x and real_resolution.y == self.cam_calib_para.resolution.y:
            print(self.preview_name + ' -> image size: ' + '\t' + str(real_resolution.x) + ' x ' + str(real_resolution.y) + ' Pixel')
        else:
            print('WARNING: image resolution is not correct!')
            print('\t' + self.preview_name + ' -> target image size: ' + '\t' + str(self.cam_calib_para.resolution.x) + ' x '
                  + str(self.cam_calib_para.resolution.y) + ' Pixel')
            print('\t' + self.preview_name + ' -> real image size:   ' + '\t' + str(real_resolution.x) + ' x ' + str(real_resolution.y) + ' Pixel')
        # -------------------------------------------------------------------------------------------------------------- init facial landmark settings
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_face_mesh = mp.solutions.face_mesh
        mp_pose = mp.solutions.pose
        mp_hands = mp.solutions.hands
        drawing_spec = mp_drawing.DrawingSpec(thickness=5, circle_radius=1)
        self.stream_obj = cap
        self.facial_lm_para = [mp_face_mesh, mp_drawing_styles, mp_drawing]
        self.pose_para = [mp_pose, mp_drawing_styles, mp_drawing]
        self.hands_para = [mp_hands, mp_drawing_styles, mp_drawing]

    def cam_stream(self) -> bool:
        # ----------------------------------------------------------------------------------- init parameters for distortion and for spatial resection
        mtx = self.cam_calib_para.inner_coeff.get_camera_matrix()
        dist_coeff = self.cam_calib_para.dist_coeff.get_dist_coeff_values()

        if self.stream_obj.isOpened():
            rval, img = self.stream_obj.read()
            undist_img, cropped_mtx = undistort_image(img, mtx, dist_coeff, True)  #for facial landmarks mandatory required!
        else:
            rval = False

        font = cv.FONT_HERSHEY_SIMPLEX
        delay_ms = 50
        cnt = 1
        avg_fps = 0
        holdto = cnt

        self.calculation_barrier.wait()
        while rval:
            start_time = time.time()
            # ---------------------------------------------------------------------------------------------------------------------- undistorted image
            rval, img = self.stream_obj.read()
            undist_img, cropped_mtx = undistort_image(img, mtx, dist_coeff, False) # for facial landmarks mandatory required!
            # --------------------------------------------------------------------------------------------------- image operators with grayscale image
            undistored_gray = cv.cvtColor(undist_img, cv.COLOR_BGR2GRAY)

            self.thread_lock.acquire()
            if self.measure_aruco:
                self.results_aruco = self.aruco_detection.detect_markers(undist_img, undistored_gray, True, True)   # TODO undistored_gray????
            if self.measure_precision_targets:
                findHighPrecisionTargets(undist_img, undistored_gray)
            if self.measure_chessboard:
                self.results_chessboard = findChessboardCorner(undist_img, undistored_gray, draw=True)
            # --------------------------------------------------------------------------------------------------------------- operators with rgb image
            undist_rgb = cv.cvtColor(undist_img, cv.COLOR_BGR2RGB)
            undist_rgb.flags.writeable = False
            if self.measure_facial_landmarks:
                self.results_facial_landmarks = findFacialLandmarks(self.facial_lm_para[0], self.facial_lm_para[1], self.facial_lm_para[2], undist_img, undist_rgb, draw=True)
            if self.measure_pose:
                self.results_pose = findPose(self.pose_para[0], self.pose_para[1], self.pose_para[2], undist_img, undist_rgb, draw=True)
            if self.measure_hands:
                self.results_hand = findHands(self.hands_para[0], self.hands_para[1], self.hands_para[2], undist_img, undist_rgb, draw=True)

            self.thread_lock.release()

            #-------------------------------------------------------------------------------------------------------------------3D-coords of media Pipe
            #self.results_facial_landmarks_3D = Image_to_ThreeD_coords(self.results_facial_landmarks)


            # ----------------------------------------------------------------------------------------------------------------------------------------
            end_time = time.time()
            avg_fps = self._display_fps(start_time, end_time, avg_fps, cnt, undist_img, font)
            if holdto - cnt > 0:
                cv.putText(undist_img, 'Wait for console', (10, 70), font, 1.0, (0, 0, 255), 3, cv.LINE_AA)
            cnt += 1
            cv.imshow(self.preview_name, undist_img)
            key = cv.waitKey(delay_ms)
            if key != -1:
                holdto = cnt + int(1 * avg_fps)

            self.calculation_barrier.wait()

        cv.destroyWindow(self.preview_name)
        return True

    @staticmethod
    def _display_fps(start_time: time, end_time: time, old_avg_fps: float, cnt: int, undist_img: np.ndarray, font):
        curr_fps = 1.0 / (end_time - start_time)
        curr_fps_string = "FPS: %4.1f" % curr_fps
        avg_fps = (old_avg_fps * (cnt - 1) + curr_fps) / cnt
        avg_fps_string = "AVG: %4.1f" % avg_fps
        cv.putText(undist_img, curr_fps_string, (10, 25), font, 0.75, (0, 255, 0), 2, cv.LINE_AA)
        cv.putText(undist_img, avg_fps_string, (160, 25), font, 0.75, (0, 128, 0), 2, cv.LINE_AA)

        return avg_fps


class InputThread(threading.Thread):
    def __init__(self, calculation_barrier):
        """
        class for a special thread which handles console input
        """

        threading.Thread.__init__(self)
        self.calculation_barrier = calculation_barrier
        self.is_active: bool = False                        # dynamic input

    def run(self):
        end_calib = False
        self.calculation_barrier.wait()
        while True:
            if self.is_active:
                inp1 = input('>>> Press "Enter" to capture an image set or type "e" to exit application: ')
                if inp1 == '':
                    # NEEDS SIGNAL TO HALT CALCULATION LOOP
                    inp2 = input('>>> Input "a" to accept image set or "r" to reject image set and return to video stream: ')
                    if inp2 == 'r':
                        continue
                    elif inp2 == 'a':
                        self.is_active = False

                        print('\r<<< Accepted image set for calculation.')
                elif inp1 == 'e':
                    end_calib = True
                    break
                else:
                    continue
            else:
                pass
        if end_calib:
            pass


# ====================================================================================================================================================
class CamThreadState:
    def __init__(self):
        self.__data: dict = {
            'calib_para': None,
            'aruco_meas': None,
            'aruco_meas_ids': None
        }

    def __str__(self):
        if isinstance(self.calib_para, CameraParameters):
            string = f'camThreadState | camera "{self.calib_para.camera_type_name}"\t| {self.calib_para.u_id}'
        else:
            string = 'camThreadState'
        return string

    @property
    def calib_para(self) -> CameraParameters:
        return self.__data['calib_para']

    @calib_para.setter
    def calib_para(self, new_calib_para: CameraParameters):
        if isinstance(new_calib_para, CameraParameters):
            self.__data['calib_para'] = new_calib_para
        else:
            raise TypeError('Attribute "calib_para" needs to be of type CameraParameters.')

    @property
    def aruco_meas(self) -> Union[dict, np.ndarray, None]:
        return self.__data['aruco_meas']

    @aruco_meas.setter
    def aruco_meas(self, new_object: Union[dict, np.ndarray, None]):
        if isinstance(new_object, (dict, np.ndarray)):
            self.__data['aruco_meas'] = new_object

    @property
    def aruco_meas_ids(self) -> Union[np.ndarray, None]:
        return self.__data['aruco_meas_ids']

    @aruco_meas_ids.setter
    def aruco_meas_ids(self, new_array: Union[np.ndarray, None]):
        if isinstance(new_array, np.ndarray):
            self.__data['aruco_meas_ids'] = new_array
