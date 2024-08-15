"""
Automatically configures all static paths via path object names set up by user in configuration.ini.

Access the static paths by importing this module (paths) or parts of it at the top of your active python file.
Each available PathObject has the attributes PathObject.path (type: pathlib.Path) and PathObject.name (file or directory name, type: str).
Paths are independent of the actual location of import statement.
"""

from Tools.path_object import PathObject
import pathlib
from Tools.Parser.parser_tools.file_parser_ini import FileParserIni

# derived from: https://stackoverflow.com/questions/25389095/python-get-path-of-root-project-structure/25389715#25389715
ROOT_DIR = PathObject(pathlib.Path(__file__).resolve().parent)
ini_parser = FileParserIni()
ini_parser.parse(pathlib.Path(ROOT_DIR.path, 'configuration.ini'))
__config_data = ini_parser.get_all_data()['DEFAULT']
if not all(req in __config_data for req in ['DATA_DIR', 'IMG_SET_DIR', 'ARUCO_TARGET_FILE', 'TYPE_CALIB_FILE', 'INDIV_CALIB_FILE']):
    raise AttributeError('At least one of the mandatory attributes in the configuration.ini is missing or faulty.')

# Available Path objects -----------------------------------------------------------------------------------------------------------------------------
DATA_DIR = PathObject(pathlib.Path(ROOT_DIR.path, __config_data['DATA_DIR']))
IMG_SET_DIR = PathObject(pathlib.Path(DATA_DIR.path, __config_data['IMG_SET_DIR']))
ARUCO_TARGET_FILE = PathObject(pathlib.Path(DATA_DIR.path, __config_data['ARUCO_TARGET_FILE']))
TYPE_CALIB_FILE = PathObject(pathlib.Path(DATA_DIR.path, __config_data['TYPE_CALIB_FILE']))
INDIV_CALIB_FILE = PathObject(pathlib.Path(DATA_DIR.path, __config_data['INDIV_CALIB_FILE']))
# ----------------------------------------------------------------------------------------------------------------------------------------------------
