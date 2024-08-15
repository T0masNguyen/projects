import pathlib
import sys

from typing import List, Tuple, Union

from Tools.camera_assignment import *
from Measurement_operators.aruco_detection import ArucoDetection
from Tools.camera_thread import CameraThreadHandler, CamThreadState
from Tools.trafo_module import PoseTrafo


class CameraCalibration:
    """
    Performs a intrinsic and extrinsic camera calibration of one or multiple whitelisted cameras with a 2D or 3D aruco target.

    Usage:
        1. CameraCalibration(cam_type_whitelist) instantiation
        2. calibrate()          ->  execute calibration routine (Dev: see calibrate() process flow to follow intended usage of internal methods)
        3. ...()                ->  Ergebnisse schreiben        TODO
    """

    def __init__(self, cam_type_whitelist: List[str], resolution: Tuple[int, int] = (1280, 720)):
        """

        :param cam_type_whitelist:
        :param resolution:
        """

        self.aruco_detection: ArucoDetection = ArucoDetection()
        self.aruco_target_ids: Union[np.ndarray, None] = None
        self.aruco_target_points: Union[np.ndarray, None] = None
        self.threads: CameraThreadHandler = CameraThreadHandler(cam_type_whitelist, resolution, manual_inp_control=True)

    # ================================================================================================================================================
    def get_target_data(self, alt_file: str = None):
        """Load target data specified in aruco target file of the project.

        Writes instance variables aruco_target_ids, aruco_target_points.

        :param alt_file: (optional) alternative file name for target data file in the Data folder. If None, default configuration file will be used.
        """

        if alt_file and isinstance(alt_file, str) and alt_file.endswith('.ini'):
            paths.ARUCO_TARGET_FILE = paths.PathObject(pathlib.Path(paths.DATA_DIR.path, alt_file))
        ppp = PointPatternParser(paths.ARUCO_TARGET_FILE.path)
        ppp.parse_file()
        self.aruco_target_ids = ppp.ar_id_array
        self.aruco_target_points = ppp.ar_point_array
        self.aruco_detection.aruco_dict = ppp.processed_data['settings']['aruco_dict']
        self.threads.aruco_dict = self.aruco_detection.aruco_dict
        del ppp

    # ================================================================================================================================================
    def calibrate(self):
        """Main routine for camera calibration."""
        self.threads.startup_threads()
        i = 0
        end_calib = False
        try:
            while i <= 10000 and not end_calib:   # grab one state
                print(f'=== ITERATION {i} =======================================')

                # INPUT HANDLING

                self.threads.calculation_barrier.wait()
                self.threads.thread_lock.acquire()
                raw_state_meas = self.grab_state_meas()
                self.threads.thread_lock.release()
                print("Berechnung", i)
                # frame Rohmessungen aus bild threads rausholen
                if any(map(lambda m: True if m.aruco_meas['data'] else False, raw_state_meas)):
                    # Vorbereiten der Messwerte selbst
                    com_state_meas, com_state_target_ids, com_state_target_points = self.compare_features(raw_state_meas)
                    com_state_meas = self.meas_dict_to_numpy_arrays(com_state_meas)
                    # align meas and target (order)
                    aln_state_meas, aln_state_aruco_ids, aln_state_target_points = self.align_features(com_state_meas,
                                                                                                       com_state_target_ids,
                                                                                                       com_state_target_points)
                    # äußere Orientierungen
                    self.calc_exterior_orient(aln_state_meas, aln_state_target_points)
                    # relative orientierung
                    # add to frame pile
                    # rel_orient = self.calc_relative_orient()
                    # reduced l = L - L0
                    l_red = self.reduce_observations()
                    # gleich an calib weitergeben
                    # res = self.adjustment_solver()
                    # write iteration output
                    self.write_calib_results()
                    i += 1
                else:
                    print('Frames of state rejected. No aruco markers could be found in any thread.')
        except KeyboardInterrupt:
            sys.exit(0)
        print("Exit\n")

    # ================================================================================================================================================
    def grab_state_meas(self) -> List[CamThreadState]:
        frame_meas = []
        if self.threads.thread_list:
            # lock acquire
            for thread in self.threads.thread_list:
                thread_state = CamThreadState()
                thread_state.calib_para = thread.cam_calib_para
                thread_state.aruco_meas = thread.results_aruco
                frame_meas.append(thread_state)
            # lock release

        return frame_meas

    # ================================================================================================================================================
    def compare_features(self, state_meas: List[CamThreadState]) -> Tuple[List[CamThreadState], List[np.ndarray], List[np.ndarray]]:
        """Compares and shortens detected aruco markers and the target data to match contained aruco markers.

        :param state_meas: List of CamThreadState.
        :return: Cleaned list of CamThreadStates, aruco target id arrays and aruco target point arrays.
        """
        state_target_ids = []
        state_target_points = []
        for thread_repr in state_meas:
            if thread_repr.aruco_meas['data']:
                target_ids = list(map(str, self.aruco_target_ids))
                surplus_meas = [ar_id for ar_id in thread_repr.aruco_meas['data'].keys() if ar_id not in target_ids]
                for ar_id in surplus_meas:
                    thread_repr.aruco_meas['data'].pop(ar_id)
                    thread_repr.aruco_meas['_meta']['_detected_ids'].remove(int(ar_id))
                surplus_target = [idx for idx, ar_id in enumerate(target_ids) if ar_id not in thread_repr.aruco_meas['data'].keys()]
                thread_target_ids = np.delete(self.aruco_target_ids, surplus_target)
                thread_target_points = np.delete(self.aruco_target_points, surplus_target, axis=0)
                # print(*list(zip(thread_target_ids, thread_target_array)), sep='\n')
                state_target_ids.append(thread_target_ids)
                state_target_points.append(thread_target_points)

        return state_meas, state_target_ids, state_target_points  # -> order of target lists is synchronized with state_meas thread order

    # ------------------------------------------------------------------------------------------------------------------------------------------------
    def align_features(self, state_meas: List[CamThreadState], state_target_ids: List[np.ndarray], state_target_points: List[np.ndarray]) \
            -> Tuple[List[CamThreadState], List[np.ndarray], List[np.ndarray]]:
        """Brings measurement and target arrays in one consistent order.

        Requires unaltered order of the threads in the all three input lists.

        :param state_meas:
        :param state_target_ids:
        :param state_target_points:
        :returns:
        """

        for idx, thread_repr in enumerate(state_meas):
            # starting point:
            #       both id arrays contain same ids, but in different orders and point arrays match their respective id array (see compare_features())

            # sort both id arrays respectively from lowest to highest id and synchronize the corresponding point array
            meas_sorted_idxs = np.argsort(thread_repr.aruco_meas_ids)
            thread_repr.aruco_meas_ids = thread_repr.aruco_meas_ids[meas_sorted_idxs]
            thread_repr.aruco_meas = thread_repr.aruco_meas[meas_sorted_idxs]

            target_sorted_idxs = np.argsort(state_target_ids[idx])
            state_target_ids[idx] = state_target_ids[idx][target_sorted_idxs]
            state_target_points[idx] = state_target_points[idx][target_sorted_idxs]

            # consistency check
            if not np.array_equal(thread_repr.aruco_meas_ids, state_target_ids[idx]):
                raise ValueError(f'Implementation altered! Processed thread target data and thread measurement data don\'t contain identical sets '
                                 f'of aruco ids. Check for implementation issues in methods {self.compare_features.__name__}() and less '
                                 f'likely {self.meas_dict_to_numpy_arrays.__name__}().'
                                 f'Don\'t alter order in thread_list and derived thread target data after {self.compare_features.__name__}()!')

        return state_meas, state_target_ids, state_target_points

    # ================================================================================================================================================
    def meas_dict_to_numpy_arrays(self, state_meas: List[CamThreadState]) -> List[CamThreadState]:
        for thread_repr in state_meas:
            if thread_repr.aruco_meas['data']:
                meas_aruco_ids, meas_aruco_array = self.aruco_detection.marker_dict_to_numpy_arrays(thread_repr.aruco_meas)
                thread_repr.aruco_meas = meas_aruco_array       # overwrite dict on purpose, not needed anymore
                thread_repr.aruco_meas_ids = meas_aruco_ids
            else:
                thread_repr.aruco_meas = None
                thread_repr.aruco_meas_ids = None
        return state_meas

    # ================================================================================================================================================
    def calc_exterior_orient(self, state_meas: List[CamThreadState], state_target_points: List[np.ndarray]):
        for idx, thread_repr in enumerate(state_meas):
            cam_mtx = thread_repr.calib_para.inner_coeff.get_camera_matrix()
            cam_dist = thread_repr.calib_para.dist_coeff.get_dist_coeff_values()
            flags = cv.SOLVEPNP_ITERATIVE

            if len(thread_repr.aruco_meas_ids) < 6:
                raise ValueError(f'To few aruco markers were detected for processing the exterior orientation. '
                                 f'Found {len(thread_repr.aruco_meas_ids)}, need 6.')

            ret, rvecs, tvecs = cv.solvePnP(state_target_points[idx], thread_repr.aruco_meas, cam_mtx, cam_dist, flags)
            trafo = PoseTrafo('rodrigues', 'rad', rvecs, tvecs)
            t_rvec = trafo.get_rvec('eulerxyz', 'deg')
            t_tvec = trafo.tvec
            # TODO: Dürfen die äußeren Orientierungen hier schon in die einzelnen CameraParameters Instanzen rein?

            pass

    # ------------------------------------------------------------------------------------------------------------------------------------------------
    def calc_relative_orient(self, current_image_set):
        """COPIED STRAIGHT OUT OF OCVEXCERCISESCALIB"""
        # relative_orient = {}  # clear for every new object sate -> easier to handle
        # # ::if only one camera, its relative orientation is [0,0,0,0,0,0] and the targets relative orientation is the cameras exterior orientation
        # if len(model.model_state_list[-1].camera_element_list) == 1:
        #     print('Relative orientation can\'t be calculated with only one stream. Skip relative orientation.')
        #     return None
        #
        # # ::camera of the first stream as origin for every other camera (Hint: might have to be sure its always the same camera as an origin - first in CameraList?)
        # first_cam = model.model_state_list[-1].camera_element_list[0].camera_type_name  # define first camera as first camera in cam_list
        # print('The coordinate origin is in camera ', first_cam)
        #
        # for cam in model.model_state_list[-1].camera_element_list:
        #     cam = cast(geometric_model.CameraElement, cam)  # get cam object from list
        #     cam_name = cam.camera_type_name
        #     # ::Basiskamera
        #     if cam_name == first_cam:
        #         rvec_first_cam = cam.exterior_orientation.rvecs
        #         tvec_first_cam = cam.exterior_orientation.tvecs
        #         T0_S_in_C = PoseTransformation(None, rvec_first_cam, tvec_first_cam)  # Object includes pose from Chessboard to camera
        #     # ------------------------------------------------- aktuelle Pose (Schachbrett in Kamera) der zweiten Kamera
        #     # ::andere Kameras in Bezug auf Basiskamera
        #     else:
        #         rvec_other_cam = cam.exterior_orientation.rvecs
        #         tvec_other_cam = cam.exterior_orientation.tvecs
        #         Tj_C_in_S = PoseTransformation(None, rvec_other_cam, tvec_other_cam)
        #
        #         # ---------------------------------------------------relative Pose Kamera als Ursprung zu restlichen Kameras
        #         ci2_cj_relative_pose = T0_S_in_C.__T4x4 * Tj_C_in_S.T4x4_inv
        #         relative_orient[f'{first_cam}_to_{cam_name}'] = {'Ci2Cj_relative_Pose': ci2_cj_relative_pose}
        #
        # # -------------------------------------- calculate the absolute Pose for the Cameras --> the Origin is first cam
        # # ---------------------------- So the relative pose between the first cam and the other cam is the absolute pose
        #
        # for cam in model.model_state_list[-1].camera_element_list:
        #     cam = cast(geometric_model.CameraElement, cam)
        #     cam_name = cam.camera_type_name
        #     if cam_name == first_cam:  # first cam define the origin so rvec and tvec = [0]
        #         absolute_pose = np.matrix([[1.0, 0.0, 0.0, 0.0],
        #                                    [0.0, 1.0, 0.0, 0.0],
        #                                    [0.0, 0.0, 1.0, 0.0],
        #                                    [0.0, 0.0, 0.0, 1.0]])
        #     else:
        #         absolute_pose = relative_orient[f'{first_cam}_to_{cam_name}']['Ci2Cj_relative_Pose']  # pose = relative pose to first cam
        #     cam.set_absolute_pose(absolute_pose)
        pass

    # ------------------------------------------------------------------------------------------------------------------------------------------------
    def reduce_observations(self):
        """l = L - L0"""
        pass

    # ================================================================================================================================================
    def adjustment_solver(self):
        """COPIED OUT MODEL_BASED_TRANSFORMATION"""
        # start_time = datetime.now()
        # if self.size_n < self.size_u:  # condition: n > u
        #     raise Exception(f'Not enough measurements for the number of unknowns. measurements: {self.size_n} and unknowns: {self.size_u}.')
        # print(f'Amount of observations: {self.size_n}\nAmount of unknown paras: {self.size_u}')
        #
        # # stochastic model ---------------------------------------------------------------------------------------------------------------------------
        # Qll = np.eye(self.size_n)  # cofactor matrix of observations (n,n)
        # iteration = 0
        #
        # # iterate until exit condition is met or maximum number of iterations is reached
        # while True:
        #     print(f'---Iteration {iteration} adjustment...')
        #     l_values = self.reduced_l_values.copy()  # get reduced l vec
        #     A_Mat = self.reduced_A_matrix  # get design matrix A
        #     Pll = np.eye(self.size_n)
        #     N = A_Mat.T @ Pll @ A_Mat  # matrix of normal equations (u,u)
        #     n = A_Mat.T @ Pll @ l_values  # absolute term (u,1)
        #     # cofactor matrix of unknown parameters (u, u)
        #     try:
        #         print(f'N det: {np.linalg.det(N)}')
        #         Qxx = np.linalg.inv(N)
        #     except np.linalg.LinAlgError:
        #         raise ValueError(f'Numpy LinAlgError because of singular matrix N. Has rank: {np.linalg.matrix_rank(N)}')
        #     Qvv = np.linalg.inv(Pll) - A_Mat @ Qxx @ A_Mat.T
        #     # improvements of the unknown parameters
        #     x_hat = Qxx @ n
        #     # improvements (Verbesserungen v) of observations
        #     vll = (A_Mat @ x_hat) - l_values  # residuals
        #     # difference of observation and unknown parameter.py number
        #     redundancy = self.size_n - self.size_u
        #     # standard deviation a posteriori
        #     s0_hat = sqrt(vll.T @ Pll @ vll / redundancy)
        #     print(f's0_hat: {s0_hat}')
        #     # element of the cofactor matrix of the unknowns (it's main diagonal) -> a vector
        #     quu_diag = np.diag(Qxx)
        #     # standard deviation of the unknowns
        #     su_hat = np.array([s0_hat * sqrt(abs(quu_element)) for quu_element in quu_diag])  # -> is a vector
        #
        #     # variance-covariance matrix
        #     sigma_xx = np.square(s0_hat) * Qxx
        #     # derive correlation matrix
        #     corr_matrix = np.copy(sigma_xx)
        #     for i in range(sigma_xx.shape[0]):
        #         for j in range(sigma_xx.shape[1]):
        #             # correlation between parameters of the variance-covariance matrix
        #             corr_matrix[i, j] = sigma_xx[i, j] / (np.sqrt(sigma_xx[i, i]) * np.sqrt(sigma_xx[j, j]))
        #
        #     # apply improvements x_hat (x^) onto model parameters and recalculate model --------------------------------------------------------------
        #     self.geom_model.update_X_values(
        #         x_hat)  # TODO /Hinweis: Modell hat danach aktualisierte Paras, ist aber noch nicht neu durchgerechnet worden
        #     # update the parameters standard deviations of this now fully calculated iteration
        #     self.geom_model.update_X_std(su_hat)  # TODO /Hinweis: in dieser Iteration berechnete Standardabweichungen
        #
        #     # rooted mean square (RMS) of the reduced l-vector
        #     RMS_l = sqrt(np.mean(np.square(
        #         l_values)))  # TODO: !!!!       /Hinweis: l_values ist Stand Anfang der Ausgleichung, für Aktualisierung neues do_assignment erforderlich
        #     print(f"RMS_l: {RMS_l}")
        #
        #     # Evaluate exit criteria =================================================================================================================
        #     if abs(s0_hat - old_s0_hat) < self.__s0_exit_crit * s0_hat and iteration != 0:
        #         print('\n-----------------------------------------------------------------------------------------')
        #         print(f'Optimization finished after {iteration} iterations - time: {datetime.now().strftime("%H:%M:%S")}')
        #         print('Exit condition: reached local minimum')
        #         print('\n-----------------------------------------------------------------------------------------')
        #         break
        #
        #     elif iteration >= max_iter:
        #         print('\n-----------------------------------------------------------------------------------------')
        #         print(f'Optimization finished - time: {datetime.now().strftime("%H:%M:%S")}')
        #         print('Exit condition: max. number of iterations reached')
        #         print('\n-----------------------------------------------------------------------------------------')
        #         break
        #
        #     # ========================================================================================================================================
        #     # if exit criteria are not met, continue adjustment and recalculate updated model for the next iteration
        #     old_s0_hat = s0_hat
        #     # TODO Hinweis Mirko: diese Zeilen bilden das ursprüngliche assignment. U.U. andere Platzierung notwendig + Naherungswertnachbestimmung?
        #     # recalculate model and get model data
        #     # TODO: Näherungswertupdate?
        #     self.geom_model.compose_sim_A_matrix(epsilon=self.__epsilon)
        #     self.do_assignment()  # confirmation of new values by recalculation of model, beginning the next iteration
        #
        #     iteration += 1
        #
        #     # adjusted model calculation for next iteration ==========================================================================================
        #     print(f'{datetime.now().strftime("%H:%M:%S")} Iteration {iteration} model recalculation...')
        #
        #     # ---- end of adjustment loop ------------------------------------------------------------------------------------------------------------
        # calc_time = datetime.now() - start_time
        # hr = str(calc_time.seconds // 3600).zfill(2)
        # m = str(calc_time.seconds % 3600 // 60).zfill(2)
        # sec = str(calc_time.seconds % 3600 % 60).zfill(2)
        # print(f'Duration of adjustment: {hr}:{m}:{sec}')
        # calib_path = self.__write_calib_results()
        # print(f'Calibration results written in file {calib_path}.')
        pass

    # ================================================================================================================================================
    def write_calib_results(self):
        # TODO: high-level method handling calib output writing via CalibDataHandler methods
        pass


# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
if __name__ == '__main__':
    # settings
    cam_type_whitelist = ['Depstech webcam', 'GENERAL WEBCAM', 'Integrated Webcam', 'Integrated Camera']
    resolution = (1280, 720)

    # init camera calibration
    cc = CameraCalibration(cam_type_whitelist, resolution)
    cc.get_target_data(alt_file=None)       # Override option provided
    cc.calibrate()
    pass
