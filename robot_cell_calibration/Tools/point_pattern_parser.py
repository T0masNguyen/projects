"""
Parser to read a aruco-target file list -> intended for Data\aruco_pattern.ini
return: aruco structure as a dict or as synced aruco id and points numpy arrays (see class description)
"""

import paths            # ONLY necessary for usage example at the bottom
import json             # ONLY necessary for usage example at the bottom (nicer print look)
import pathlib
import warnings
from typing import Union, Tuple
import numpy as np
from Tools.Parser._base_ini_parser import BaseIniParser
from Tools.Parser.parser_tools.line_parser import Separators


class PointPatternParser(BaseIniParser):
    """High level FileParser for object-point-pattern ini-files following the set up constraints for this file.

    High level child class is capsuled from the underlying base class parsing process and only needs to address use case specific
    file structures (parsing constraints/data separators) file path and optional custom post processing method.
    For implementation details and requirements check documentation of base class BaseIniParser

    1.  PointPatternParser(file_path: pathlib.Path | str)
    2.  [processed_data (-> dict) =] PointPatternParer.parse_file()
    ----------
    3a. processed_data (-> dict) = PointPatternParser.processed_data
    OR as synced numpy arrays:
    3b. aruco_ids = PointPatternParser.ar_id_array
        aruco_points = PointPatternParser.ar_point_array
    """

    def __init__(self, file_path: Union[pathlib.Path, str]):
        """Initialize instance of PointPatternParser.

        Wraps baseclass __init__ to enable explicit file_path constructor argument instead of static declaration in implemented _assign_work_path().

        :param file_path: full path (incl. file extension) of the .ini-file with 3D object point definitions in dictated syntax. Default: None for empty init.
        """

        self.__work_path: pathlib.Path = pathlib.Path(file_path)
        self.ar_id_array: Union[np.ndarray, None] = None
        self.ar_point_array: Union[np.ndarray, None] = None
        super().__init__()

    def _assign_work_path(self):
        path = pathlib.Path(self.__work_path)
        if not path.exists():
            raise ValueError(f'Path "{self.__work_path}" doesn\'t exist.')

        return path

    def _define_constraints(self) -> dict:
        constraints = {
            'section_constraints': {
                'settings': {
                    'force_options': ['aruco_dict'],
                    'max_elements': 1,
                    'min_elements': 1
                },
                'point_pattern': {
                    'head': ['id', 'size', 'x', 'y', 'z', 'sx', 'sy', 'sz'],
                    'max_elements': 8,
                    'head_idx_for_subdict': 0
                }
            },
            'structure_constraints': {
                'MUST': ['settings', 'point_pattern']
            }
        }
        return constraints

    def _set_separators(self):
        return Separators(lvl1_sep=',', comment_sep='#')

    def _post_process_data(self, proc_data: dict) -> dict:
        """(Overrides base implementation) Do post processing

        Ensure number types. Create additional numpy array representation of marker ids and marker coordinates as instance variables.

        :param proc_data: Already processed data as input for post processing on top.
        :returns: Finalized processed data dictionary.
        """

        proc_data = self._force_numerals(proc_data)
        proc_data['settings']['aruco_dict'] = proc_data['settings']['aruco_dict'][0]
        ar_id_array, ar_point_array = self._format_as_numpy_arrays(proc_data)
        self.ar_id_array: np.ndarray = ar_id_array
        self.ar_point_array: np.ndarray = ar_point_array

        return proc_data

    def _force_numerals(self, proc_data: dict) -> dict:
        """Ensure number types for elements that should be numerals.

        Anything that isn't a number but should in these elements will be replaced with int 0 or float 0.0.
        Method has to be adapted, too, if changes in config user names are done.

        :param proc_data: Already processed data as input for post processing on top.
        :return: Processed data dictionary with casted numerals.
        """

        force_int = ['id']
        force_float = ['size', 'x', 'y', 'z', 'sx', 'sy', 'sz']

        for s, section_data in proc_data.items():
            for e, element_data in section_data.items():
                trig_warn = False
                if not any((True if k in force_int or k in force_float else False for k in element_data)):
                    break
                for k, v in element_data.items():
                    if k in force_int and not isinstance(v, int):
                        try:
                            element_data[k] = int(v)
                        except (ValueError, TypeError):
                            element_data[k] = 0
                            trig_warn = True
                    elif k in force_float and not isinstance(v, float):
                        try:
                            element_data[k] = float(v)
                        except (ValueError, TypeError):
                            element_data[k] = 0.0
                            trig_warn = True
                if trig_warn:
                    warnings.warn(f'{self.__work_file} | {s} | {e}: Non-numerical input in numerical field had to be converted to 0. '
                                  f'Keep it consistent!', stacklevel=2)
        return proc_data

    @staticmethod
    def _format_as_numpy_arrays(proc_data: dict) -> Tuple[np.ndarray, np.ndarray]:
        """Create additional synced numpy array representation of marker ids and marker coordinates as instance variables.

        :param proc_data: Already processed data as input for post processing on top
        :return: array with the x available input aruco ids (shape x,) and an array (shape x,3) of the corresponding center points.
        """

        ar_id_array = []
        ar_point_array = []
        for m_key, m_val in proc_data['point_pattern'].items():
            try:
                ar_point_array.append([m_val['x'], m_val['y'], m_val['z']])
            except KeyError as k:
                raise KeyError(f'Processed point pattern dictionary doesn\'t have the expected structure for aruco id {m_key}.') from k
            ar_id_array.append(m_key)
        ar_id_array = np.array(ar_id_array, int)
        ar_point_array = np.array(ar_point_array, float)

        return ar_id_array, ar_point_array


# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
if __name__ == '__main__':
    print(f'---- Example on how to use PointPatternParser ----\n')
    # parser handling
    markers = PointPatternParser(pathlib.Path(paths.ARUCO_TARGET_FILE.path))
    markers.parse_file()

    # access results
    ar_id_array = markers.ar_id_array
    ar_point_array = markers.ar_point_array
    processed_data = markers.processed_data

    # output prints
    print('Object string: ', markers)
    print(f'Aruco IDs:\t(Type: {type(ar_id_array).__name__})\n')
    print(ar_id_array)
    print('\n==================================================')
    print(f'Aruco points:\t(Type: {type(ar_point_array).__name__})\n')
    print(ar_point_array)
    print('\n==================================================')
    print(f'Aruco process data:\t(Type: {type(processed_data).__name__})\n')
    print(json.dumps(processed_data, indent=2))                 # json for nicer print only
