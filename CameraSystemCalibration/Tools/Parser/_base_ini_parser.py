"""The parser package is copied from model based transformation (commit state of 2022-05-30). Changes were made to import paths and the Config class
was replaced by the repos own PathConfig class."""

from Tools.Parser.constraints import Constraints
from Tools.Parser.parser_tools.file_parser_ini import FileParserIni
from Tools.Parser.parser_tools.line_parser import LineParser, Separators
import abc
import pathlib
from typing import Tuple
from functools import reduce
from operator import xor


# ====================================================================================================================================================
class BaseIniParser(metaclass=abc.ABCMeta):
    """This is the base class for high level ini-file parsers.

    High level child classes are capsuled from the underlying base class parsing process and only need to address use case specific
    file structures (parsing constraints/data separators) and the file path.

    For additional custom functionality, _post_process_data() can be implemented (optionally) in child class to include any additional checks,
    conversions or indirect method calls.

    Abstract methods to implement:      \n
    -   _assign_work_path() -> str      \n
    -   _define_constraints() -> dict   \n
    -   _set_separators() -> Separators

    Optional overriding of processed data for post processing:  \n
    -   _post_process_data() -> dict

    Main method called after instantiation:    \n
    -   parse_file() -> dict with processed file data

    Explicit access to parsed file data (property):    \n
    -   self.processed_data
    """

    def __init__(self):
        """Initialize class"""

        self.__work_path: pathlib.Path = pathlib.Path(self._assign_work_path())
        self.__work_file: str = pathlib.PurePath(self.__work_path).name
        self.__processed_data: dict = {}

        self._fpi = FileParserIni()
        self._lp = LineParser()
        self._constr = Constraints(self._define_constraints())
        self.__seprt: Separators = self._set_separators()

    def __str__(self):
        return f'{self.__class__.__name__} for {self.__work_file} | Status: {"parsed" if self.__processed_data else "not parsed"}'

    @property
    def work_path(self) -> pathlib.Path:
        return self.__work_path

    @property
    def work_file(self) -> str:
        return self.__work_file

    @work_file.setter
    def work_file(self, name: str):
        self.__work_file = name
        self.__work_path = pathlib.Path(self._assign_work_path())

    @property
    def constraints(self) -> dict:
        return self._constr.constraints

    @property
    def processed_data(self) -> dict:
        return self.__processed_data

    # MAIN METHOD ====================================================================================================================================
    def parse_file(self):
        """High-level method handling parsing process. Returns fully processed .ini-file data as a dictionary."""
        raw_data = self._read_file()
        data_sections = list(raw_data.keys())
        self._check_structure_constr(data_sections)
        translated_data = self._translate_data(raw_data)
        self._check_section_constr(translated_data)
        processed_data = self._format_data(translated_data)
        # enable custom data post processing
        processed_data = self._post_process_data(processed_data)
        self.__processed_data = processed_data

        return processed_data

    # REQUIRED IMPLEMENTATION ========================================================================================================================
    @abc.abstractmethod
    def _assign_work_path(self) -> str:
        """
        Return work path for .ini-file of this parser implementation. If it returns a string and not already a pathlib.Path object,
        it will be converted to a pathlib.Path object automatically.
        """
        pass

    # ------------------------------------------------------------------------------------------------------------------------------------------------
    @abc.abstractmethod
    def _define_constraints(self) -> dict:
        """Define the dictionary used as a template for structural constraints and return it.

        Contains all expected and required section names, possible headers or mandatory options as well as minimum or maximum
        expected count of elements in a data row per section.
        Template with dict keys to use:

        constraints = {
            'section_constraints': {
                '<section_name>': {
                    'head': List[str] | None,           Default: None   -> set if header must be considered
                    'head_idx_for_subdict': int | None, Default: None   -> head[idx] to be used as subdict-key for elements
                    'min_elements': int | None,         Default: None   -> min. nr. of elements in line if more than just the fixed elements are expected
                    'max_elements': int | None,         Default: None   -> max. total nr. of elements in line
                    'force_options': List[str] | None,  Default: None   -> specified options have to appear
                },
                '<section_name2>': { },
            },
            'structure_constraints': {
                'MUST': List[str] | None,                   Default: None
                'XOR': List[str]] | List[ List[str] ] | None,     Default: None
                'OR': List[str]] | List[ List[str] ] | None,     Default: None
                'OPTIONAL': List[str] | None                Default: None
            }
        }
        """
        pass

    @abc.abstractmethod
    def _set_separators(self) -> Separators:
        """Configure the set of separators used to interpret, separate & convert raw data strings by initializing a Separator object and return it."""
        pass

    # BASE FUNCTIONALITY =============================================================================================================================
    def _read_file(self) -> dict:
        """Get raw content from file."""
        raw_data = self._fpi.get_all_data(self.__work_path)
        return raw_data

    # DATA PROCESSING --------------------------------------------------------------------------------------------------------------------------------
    def _translate_data(self, raw_data: dict) -> dict:
        """Separates elements of the raw data strings as implied by the set rules for separators and converts them into their plausible data types.

        Separation detects dedicated headers and will distinct between fixed part elements and more loosely defined umbrella part elements which
        contain their own named declaration.

        :param raw_data: Unprocessed data dictionary from basic file parsing with values buried in line strings.
        :returns: Dict with input key structure but data separated and converted. Headers as List[str], fixed data in tuple[1] (always list), umbrella data in tuple[2] (always list).
        """
        transl_data = {}
        for section in raw_data:
            sect_dict = {}
            umbr_idx = None
            try:
                req_header = True if self._constr.constraints['section_constraints'][section]['head'] not in [None, '', []] else False
            except KeyError:
                req_header = False
            if req_header:
                try:
                    raw_header = raw_data[section]['head']
                except KeyError as k:
                    raise KeyError(f'The section "{section}" in "{self.__work_file}" requires a header with the option name "head".') from k
                # No head_lvl2_sep in base ini parser, since it can't be rearranged automatically. Handling of this should be application dependent.
                sep_header, umbr_idx = self._lp.separate_header(raw_header,
                                                                self.__seprt.lvl1_sep,
                                                                None,
                                                                self.__seprt.umbrella_sep,
                                                                self.__seprt.comment_sep)
                # update sect_dict with header
                sect_dict['head'] = sep_header

            option_gen = (s for s in raw_data[section].keys() if s != 'head')
            for option in option_gen:
                raw_content = raw_data[section][option]
                try:
                    sep_content = self._lp.separate_data(raw_content,
                                                         self.__seprt.lvl1_sep,
                                                         self.__seprt.data_lvl2_sep,
                                                         self.__seprt.data_lvl3_sep,
                                                         self.__seprt.decimal_sep,
                                                         self.__seprt.comment_sep,
                                                         umbr_idx,
                                                         True,
                                                         self.__seprt.ignore_chars)
                except IndexError as ie:
                    raise ValueError(f'Possible input data error/missing entry in option "{option}" of section "{section}" '
                                     f'in "{self.__work_file}".') from ie
                sect_dict[option] = sep_content
            transl_data[section] = sect_dict

        return transl_data

    # ------------------------------------------------------------------------------------------------------------------------------------------------
    def _format_data(self, transl_data: dict) -> dict:
        """Arrange translated data according to the individual settings and conditions of the sections into an organized dictionary.

        If sections have a specified header, the data will be arranged according to the header, with thoroughly named umbrella elements.
        If specified by the constraints, a specific element of each option will be used as the new option name.
        Without a header, the iterables are cleaned and flattened to one level.

        :param transl_data: Preprocessed file data dictionary with separated and converted elements.
        :returns: Arranged and formatted data dictionary.
        """
        fmtd_data = transl_data     # same dict but cleaner for return
        for section in transl_data:
            tmp_sect = {}
            if 'head' in transl_data[section].keys():
                header = transl_data[section].pop('head')
                hkey_idx = self._constr.constraints['section_constraints'][section]['head_idx_for_subdict']
                section_has_umbrella = any([True if l[1] else False for l in transl_data[section].values()])
                for option in transl_data[section]:
                    if option == 'head':
                        continue
                    key = transl_data[section][option][0][hkey_idx] if hkey_idx is not None else option
                    line_data = transl_data[section][option]
                    # fixed elements part
                    tmp_sect[key] = dict(zip(header, line_data[0]))
                    if section_has_umbrella:
                        tmp_sect[key][header[-1]] = {}
                    # umbrella part with subdictionary, values without assigned name will be stacked in "-undefined-" key
                    if line_data[1]:
                        for item in line_data[1]:
                            if isinstance(item, list):
                                tmp_sect[key][header[-1]].update({item[0]: item[1]})
                            else:
                                if '-unknown-' in tmp_sect[key][header[-1]]:
                                    tmp_sect[key][header[-1]]['-unknown-'].append(item)
                                else:
                                    tmp_sect[key][header[-1]]['-unknown-'] = [item]
            else:
                tmp_sect = {option: elements[0] for (option, elements) in transl_data[section].items()}
            fmtd_data[section] = tmp_sect

        return fmtd_data

    # PLACEHOLDER METHOD (OPTIONAL OVERRIDING) -------------------------------------------------------------------------------------------------------
    def _post_process_data(self, proc_data: dict) -> dict:
        """Base method which can be overriden by child class to do implementation specific data transformation after generally acting _format_data().

        On default just returns input parameter.
        Signal overwrite status in new docstring of child method."""

        return proc_data

    # CHECKERS ---------------------------------------------------------------------------------------------------------------------------------------
    def _check_structure_constr(self, data_sections: list) -> bool:
        """Checks if the sections of the raw file data matches the structural constraints set for this file parser use case.

        :param data_sections: Section strings of the data dictionary in a list.
        :returns: True if raw data meets the structure constraints for this parser.
        """

        self.__must_check(data_sections)
        if self._constr.constraints['structure_constraints']['OR']:
            self.__or_check(data_sections)
        if self._constr.constraints['structure_constraints']['XOR']:
            self.__xor_check(data_sections)
        return True

    # ------------------------------------------------------------------------------------------------------------------------------------------------
    def _check_section_constr(self, transl_data: dict) -> bool:
        """Checks the sections and options whether they meet the section constraints set for this file parser use case.

        :param transl_data: Preprocessed file data dictionary with separated and converted elements.
        :returns: True if translated data meets the individual section constraints for this parser."""

        for section in transl_data:
            try:
                constr = self._constr.constraints['section_constraints'][section]
            except KeyError:
                continue
            self.__force_options_check(constr, section, transl_data)
            self.__head_check(constr, section, transl_data)
            self.__min_elements_check(constr, section, transl_data)
            self.__max_elements_check(constr, section, transl_data)
            self.__head_idx_check(constr, section, transl_data)

        return True

    # ------------------------------------------------------------------------------------------------------------------------------------------------
    def __force_options_check(self, constr, section, transl_data) -> bool:
        """Checks if section has all required option keys."""
        if constr['force_options']:
            missing_opt = [o for o in constr['force_options'] if o not in transl_data[section]]
            if missing_opt:
                raise ValueError(f'The required option(s) "{", ".join(missing_opt)}" for section "{section}" could not be found '
                                 f'in "{self.__work_file}".')
        return True

    def __head_check(self, constr, section, transl_data) -> bool:
        """Checks if section head matches its head constraints."""
        if constr['head']:
            if 'head' not in transl_data[section].keys():
                raise KeyError(f'Section "{section}" in "{self.__work_file}" requires a '
                               f'header but none could be found. Needs to be an dedicated option with the name "head".')
            missing_para = [str(p) for p in constr['head'] if p not in transl_data[section]['head']]
            if missing_para:
                raise ValueError(f'The required header parameter(s) "{", ".join(missing_para)}" for section "{section}" could not be found '
                                 f'in "{self.__work_file}".')
        return True

    def __min_elements_check(self, constr, section, transl_data) -> bool:
        """Checks if the minimum amount of elements is present for the options of a section."""
        if constr['min_elements']:
            short_options = []
            short_options_len = []
            min_len = constr['min_elements']
            for option in transl_data[section]:
                if option == 'head':
                    continue
                fix_len = len(transl_data[section][option][0])
                umbr_len = len(transl_data[section][option][1])
                options_len = fix_len + umbr_len
                # logically fix_len itself must be >= min_len, however this check is integrated in _translate_data() error handling
                if options_len < min_len:
                    short_options.append(option)
                    short_options_len.append(str(options_len))
            if short_options:
                raise ValueError(f'The minimum element count for option(s) "{", ".join(short_options)}" '
                                 f'in section "{section}" of "{self.__work_file}" is not satisfied. '
                                 f'Requires at least {min_len} (found {", ".join(short_options_len)}).')
        return True

    def __max_elements_check(self, constr, section, transl_data) -> bool:
        """Checks if the maximum amount of elements is not exceeded for the options of a section."""
        if constr['max_elements']:
            long_options = []
            long_options_len = []
            max_len = constr['max_elements']
            for option in transl_data[section]:
                if option == 'head':
                    continue
                fix_len = len(transl_data[section][option][0])
                umbr_len = len(transl_data[section][option][1])
                options_len = fix_len + umbr_len
                if options_len > max_len:
                    long_options.append(option)
                    long_options_len.append(str(options_len))
            if long_options:
                raise ValueError(f'The maximum element count for option(s) "{", ".join(long_options)}" '
                                 f'in section "{section}" of "{self.__work_file}" is not satisfied. '
                                 f'Requires at most {max_len} (found {", ".join(long_options_len)}).')
        return True

    def __head_idx_check(self, constr, section, transl_data) -> bool:
        """Checks of using the data and header of the section is compatible with the requested index as a new key."""
        if constr['head_idx_for_subdict'] is not None:
            hkey_idx = constr['head_idx_for_subdict']
            key_candidates = [transl_data[section][option][0][hkey_idx] for option in transl_data[section] if option != 'head']
            if len(key_candidates) != len(set(key_candidates)):
                raise ValueError(f'Can\'t use head index {hkey_idx} ("{transl_data[section]["head"][hkey_idx]}") as sub dictionary keys '
                                 f'in section "{section}" of "{self.__work_file}".'
                                 f'The "{transl_data[section]["head"][hkey_idx]}" of the options are not unique.')
            elif any(filter(self.__is_list, key_candidates)):
                raise ValueError(f'Can\'t use head index {hkey_idx} ("{str(transl_data[section]["head"][hkey_idx])}") as sub dictionary keys '
                                 f'in section "{section}" of "{self.__work_file}" on multi-layer data.')
        return True

    # ------------------------------------------------------------------------------------------------------------------------------------------------
    def __must_check(self, data_sect) -> bool:
        """Does the structure constraints MUST-check ("every section has to be there")."""
        missing_sect = [s for s in self._constr.constraints['structure_constraints']['MUST'] if s not in data_sect]
        if missing_sect:
            raise ValueError(f'MUST check: The required section(s) "{", ".join(missing_sect)}" could not be found '
                             f'in "{self.__work_file}".')
        return True

    def __or_check(self, data_sect) -> bool:
        """Does the structure constraints OR-check ("at least one section has to be there")."""
        def __logical_or(elements: list) -> bool:
            or_sect = [True if s in data_sect else False for s in elements]
            return True if any(or_sect) else False

        if not isinstance(self._constr.constraints['structure_constraints']['OR'][0], list):
            if not __logical_or(self._constr.constraints['structure_constraints']['OR']):
                raise ValueError(f'OR check: At least one of the sections "{", ".join(self._constr.constraints["structure_constraints"]["OR"])}" '
                                 f'must be included in "{self.__work_file}".')
        else:
            for e in self._constr.constraints['structure_constraints']['OR']:
                if not __logical_or(e):
                    raise ValueError(f'OR check: At least one of the sections "{", ".join(e)}" must be included '
                                     f'in "{self.__work_file}".')
        return True

    def __xor_check(self, data_sect) -> bool:
        """Does the structure constraints XOR-check ("exactly one section has to be there")."""
        def __logical_xor(elements: list) -> Tuple[bool, int]:
            xor_sect = [True if s in data_sect else False for s in elements]
            nr = xor_sect.count(True)
            res = reduce(xor, xor_sect)
            return res, nr

        if not isinstance(self._constr.constraints['structure_constraints']['XOR'][0], list):
            check, nr = __logical_xor(self._constr.constraints['structure_constraints']['XOR'])
            if not check:
                raise ValueError(f'XOR check: Exactly one of the sections "{", ".join(self._constr.constraints["structure_constraints"]["XOR"])}" '
                                 f'must be included in "{self.__work_file}". Found {nr}.')
        else:
            for e in self._constr.constraints['structure_constraints']['XOR']:
                check, nr = __logical_xor(e)
                if not check:
                    raise ValueError(f'XOR check: Exactly one of the sections "{", ".join(e)}" must be included '
                                     f'in "{self.__work_file}". Found {nr}')
        return True

    @staticmethod
    def __is_list(l) -> bool:
        return True if isinstance(l, list) else False


# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
if __name__ == '__main__':
    pass
