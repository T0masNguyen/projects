import logging
import pathlib
from typing import List, Union
import configparser as c

def logger_config():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(module)s.%(funcName)s(), line %(lineno)d - %(message)s')


# ####################################################################################################################################################
class FileParserIni:
    def __init__(self):
        """Low-level parser for .ini-files wrapping configparser functions to provide a consistent and safe parser.

        Lean initialization. File path has to be provided once in the first method call. If a different file path is provided in later
        method calls, the parser instance will be rebased to this file.
        """

        self._config: c.ConfigParser = c.ConfigParser()
        self._config.optionxform = lambda option: option     # necessary to preserve case of string data entries
        self.__file_path: Union[str, None] = None
        self.__parsable: bool = False
        self.__default_strings: List[str] = ['DEFAULT', 'Default', 'default', 'def']
        logger_config()

    def __str__(self):
        return f'FileParser(file_path={self.__file_path})'

    # Properties =====================================================================================================================================
    @property
    def file_path(self):
        return self.__file_path

    @file_path.setter
    def file_path(self, new_path):
        self.__new_file_parse(new_path)

    @property
    def parsable(self):
        return self.__parsable

    # Parsing methods ================================================================================================================================
    def parse(self, file_path: str) -> bool:
        """Try parsing any .ini file.

        :param file_path: must be absolute or relative path to a file including .ini extension.
        :returns: True if file could be parsed, False if not, ValueError if file path is empty.
        """

        # check if file exists
        if not pathlib.Path(file_path).exists():
            raise FileNotFoundError(f'File path "{file_path}" doesn\'t exist. Abort.')

        parsable = False
        if file_path in ['', None]:
            raise ValueError('File path is None OR empty string. Abort.')
        self.__reset_parser()
        self.__file_path = file_path
        try:
            ret = self._config.read(file_path)
            if ret:
                parsable = True
                self.__parsable = parsable
        except c.ParsingError as pe:
            logging.warning(f'Parsing Error: {pe.message} at file: {pe.source}.')
        except c.Error as e:
            logging.warning(f'Error: {e.message}.')

        return parsable

    # ------------------------------------------------------------------------------------------------------------------------------------------------
    def __reset_parser(self):
        """Resets instance variables to init state for clean reassignment."""
        self._config = c.ConfigParser()
        self._config.optionxform = lambda option: option     # necessary to preserve case of string data entries
        self.__file_path: Union[str, None] = None
        self.__parsable: bool = False

    # ------------------------------------------------------------------------------------------------------------------------------------------------
    def __new_file_parse(self, file_path: str) -> bool:
        """Checks if a new file is presented and tries to embed it in the instance.

        Returns either with the readable unchanged file embedded or the readable new file.

        :param file_path: new file path - same rules as in the main methods.
        :returns: True if successfully parsed new file, False if no new file, raises ValueError if new file not parsable.
        """

        if file_path not in ['', None]:
            if not self.parse(file_path):
                raise ValueError(f'Newly passed file "{file_path}" has an issue in parsing. Abort.')
            return True
        else:
            if not self.__file_path:
                raise ValueError('No valid file path set. Abort.')
            return False

    # Checking methods ===============================================================================================================================
    def check_for_section(self, section: str, file_path: str = None) -> bool:
        """Checks if current file or a new file has this section.

        :param section: section which should be checked.
        :param file_path: (Optional) alternative file (path) in which the section should be looked for.
        :returns: True if section is present, False if not.
        """

        self.__new_file_parse(file_path)
        ret = False
        try:
            if section in self.__default_strings:
                ret = True if self.get_default_data() else False
            else:
                ret = self._config.has_section(section)
        except Exception as e:
            logging.warning(f'The file: "{self.__file_path}" has an issue: {e}.')
        if not ret:
            logging.info(f'The file "{self.__file_path}" does not contain section "{section}".')
        return ret

    # ------------------------------------------------------------------------------------------------------------------------------------------------
    def check_for_option(self, section: str, option: str, file_path: str = None):
        """Checks if current file or a new file has this option in the given section.

        :param section: section which should be checked.
        :param option: option which is assumed to be in the section.
        :param file_path: (Optional) alternative file (path) in which the section should be looked for.
        :returns: True if option is present, False if not.
        """

        self.__new_file_parse(file_path)
        ret = False
        try:
            if section in self.__default_strings:
                ret = True if option in self.get_default_data() else False
            else:
                ret = self._config.has_option(section, option)
        except Exception as e:
            logging.warning(f'The file: "{self.__file_path}" has an issue: {e}.')
        if not ret:
            logging.info(f'The section "{section}" in file "{self.__file_path}" does not contain option "{option}".')
        return ret

    # Listing methods ================================================================================================================================
    def get_section_list(self, file_path: str = None) -> List[str]:
        """Returns all sections from an .ini-file after parsing checks.

        :param file_path: (Optional) alternative file (path) in which the section should be looked for.
        :returns: all sections in a .ini-file (but DEFAULT).
        """

        self.__new_file_parse(file_path)
        sections = self._config.sections()
        return sections

    # ------------------------------------------------------------------------------------------------------------------------------------------------
    def get_option_list(self, section: str, file_path: str = None) -> List[str]:
        """Returns all options of a section after parsing checks.

        :param section: section for which options should be collected.
        :param file_path: (Optional) alternative file (path) in which the section should be looked for.
        :returns: options of a section.
        """

        if self.check_for_section(section, file_path):
            options = self._config.options(section)
        else:
            raise ValueError(f'There is no section "{section}" in "{self.__file_path}". Abort.')
        return options

    # Main data getter methods =======================================================================================================================
    def get_default_data(self, file_path: str = None) -> dict:
        """Returns all option-value pairs of the default section after parsing checks.

        :param file_path: (Optional) alternative file (path) in which the section should be looked for. Default: None (Use previous file).
        :returns: option-value pairs of the DEFAULT-section as a dictionary.
        """

        self.__new_file_parse(file_path)
        defaults = dict(self._config.defaults())

        return defaults

    # ------------------------------------------------------------------------------------------------------------------------------------------------
    def get_option_value(self, section: str, option: str, file_path: str = None) -> str:
        """Return value for an option within a section.

        :param section: section which contains the option.
        :param option: option for which value should be collected.
        :param file_path: (Optional) alternative file (path) in which the section should be looked for.
        :returns: value-only of an option.
        """

        line_data = None
        if self.check_for_section(section, file_path):
            if self.check_for_option(section, option):
                if section in self.__default_strings and self.check_for_option(section, option):
                    line_data = self.get_default_data()[option]
                else:
                    line_data = self._config.get(section, option)
        return line_data

    # ------------------------------------------------------------------------------------------------------------------------------------------------
    def get_section_data(self, section: str, file_path: str = None) -> dict:
        """Returns all option-value pairs of a section.

        :param section: section for which option-value pairs should be collected.
        :param file_path: (Optional) alternative file (path) in which the section should be looked for.
        :returns: option-value pairs of a section as a dictionary.
        """

        section_data = {}
        if self.check_for_section(section, file_path):
            items = self._config.items(section)
            for idata in items:
                section_data[idata[0]] = idata[1]
        return section_data

    # ------------------------------------------------------------------------------------------------------------------------------------------------
    def get_all_data(self, file_path: str = None) -> dict:
        """Returns all section-option-value pairings of a file as a dictionary.

        Behaviour of the DEFAULT-section is unchanged from default configparser handling. It's data will be included in every section.

        :param file_path: (Optional) alternative file (path) in which the section should be looked for.
        :returns: full file raw data as a dictionary.
        """

        file_data = {}
        sections = self.get_section_list(file_path)
        if not sections:
            file_data.update({'DEFAULT': self.get_default_data()})
        else:
            for sec in sections:
                file_data.update({sec: self.get_section_data(sec)})
        return file_data


# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
if __name__ == "__main__":
    file_path = "../../../../data/99_PARSERTEST/config_user_input.ini"
    file_path2 = "../../../../data/99_PARSERTEST/session_sensor_list.ini"
    fpi = FileParserIni()
    fpi.parse(file_path2)
    a = fpi.get_all_data()
    pass
