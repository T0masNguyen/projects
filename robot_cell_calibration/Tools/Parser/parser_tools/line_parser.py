import warnings
from typing import Union, List, Tuple
import Tools.Parser.parser_tools.string_data_type_converter as conv


class Separators:
    def __init__(self, lvl1_sep: str = None, data_lvl2_sep: str = None, data_lvl3_sep: str = None, head_lvl2_sep: str = None,
                 head_umbrella_sep: str = None, decimal_sep: str = None, comment_sep: str = None, ignore_chars: str = None):
        """Object storing separator settings for data parsing with line parser functions.

        Separator characters have to be unique and can't be empty strings. If "comment separator" is specified, the string following the character
        will be ignored.

        The "umbrella separator" in a header -put in just before the last header element-, signals the start of a container like area for all
        data lines below the header which collects an arbitrary amount of parameters in each line under the name of the last header
        element (acts as an "umbrella"), e.g. to enable a variably declared "parameter" list in addition to the "fixed parameters" before the
        umbrella separator. \n
        Ideally combined with lvl2 and lvl3 separators in data lines (to enable named expressions like "id=1;2;3, fx=100").

        :param lvl1_sep: character separating basic elements of a line from each other, e.g. "v, w, x". Default: ",".
        :param data_lvl2_sep: (used for data lines) char. used to distinguish multiple parts of a single lvl1-element, e.g. "=" in assignment "x=1". Default: None.
        :param data_lvl3_sep: (used for data lines) char. used to distinguish multiple parts of a single lvl2-element, e.g. ";" in list "x=[1;2;3]". Default: None.
        :param head_lvl2_sep: (used for header only) char. used to distinguish multiple parts of a single lvl1-element, e.g. "/" in "elmnt/elmnt_para" distinction. Default: None.
        :param head_umbrella_sep: special char. used to signal start of an *args-like part with a single "umbrella" term for varying amount of values underneath. Default: None.
        :param decimal_sep: char. used as a identifier for  the decimal point for automatic type conversion. Default: ".".
        :param comment_sep: char. to mark start of end-of-line comments (part of string will be ignored by function). Default: None.
        :param ignore_chars: additional char's. all in one string, that will be stripped from the strings, e.g. non-functional "list" brackets. Default: None.
        """
        self.lvl1_sep: str = lvl1_sep if lvl1_sep else ','
        self.data_lvl2_sep: str = data_lvl2_sep if data_lvl2_sep else None
        self.data_lvl3_sep: str = data_lvl3_sep if data_lvl3_sep else None
        self.head_lvl2_sep: str = head_lvl2_sep if head_lvl2_sep else None
        self.decimal_sep: str = decimal_sep if decimal_sep else '.'
        self.comment_sep: str = comment_sep if comment_sep else None
        self.umbrella_sep: str = head_umbrella_sep if head_umbrella_sep else None
        self.ignore_chars: str = ignore_chars if ignore_chars else None


# ----------------------------------------------------------------------------------------------------------------------------------------------------
class LineParser:
    """Low-level parser providing string processing methods for single lines of data."""
    def separate_data(self, data_line: str,
                      lvl1_separator: str = ',',
                      lvl2_separator: str = None,
                      lvl3_separator: str = None,
                      decimal_separator: str = '.',
                      comment_separator: str = None,
                      umbrella_start_idx: int = None,
                      type_conversion: bool = True,
                      ignore_chars: Union[str, List[str]] = None) -> Tuple[list, list]:
        """Separates a single line of data, given as a cumulative string, into distinguished elements.

        Separator characters have to be unique and can't be empty strings. If "comment separator" is specified, the string following the character
        will be ignored.

        :param data_line: raw data string to be separated.
        :param lvl1_separator: (Optional) character separating basic elements of a line from each other, e.g. "v, w, x". Default: ",".
        :param lvl2_separator: (Optional) char. used to distinguish multiple parts of a single lvl1-element, e.g. "=" in assignment "x=1". Default: None.
        :param lvl3_separator: (Optional) char. used to distinguish multiple parts of a single lvl2-element, e.g. ";" in list "x=[1;2;3]". Default: None.
        :param decimal_separator: (Optional) char. used as a decimal point. Used if type conversion is activated. Default: ".".
        :param comment_separator: (Optional) char. to mark start of end-of-line comments (part of string will be ignored by function). Default: None.
        :param umbrella_start_idx: (Optional) if integer is passed, all line elements starting with that index will be regarded as variable "umbrella" elements. Default: None.
        :param type_conversion: (Optional) if False, the separated strings will be returned. If True, elements are converted to most plausible data type. Default: True.
        :param ignore_chars: (Optional) additional char's. all in one string, that will be stripped from the strings, e.g. non-functional "list" brackets. Default: None.
        :returns: 1st tuple element is (nested) list of separated fixed data parts, 2nd tuple element is a (nested) list of (optional) variable umbrella data parts.
        """

        umbrella_elements = []
        # --- input checks and basic formatting -----------------
        remove_chars = []
        raw_str = self._str_arg_check('data_line', data_line, True, True)
        lvl1_separator = self._str_arg_check('lvl1_separator', lvl1_separator, False, False)
        lvl2_separator = self._str_arg_check('lvl2_separator', lvl2_separator, True, False)
        lvl3_separator = self._str_arg_check('lvl3_separator', lvl3_separator, True, False)
        decimal_separator = self._str_arg_check('decimal_separator', decimal_separator, True, False)
        comment_separator = self._str_arg_check('comment_separator', comment_separator, True, False)
        if ignore_chars not in [None, '']:
            for ic in ignore_chars:
                ic = self._str_arg_check(f'ignore_char (\'{ic}\')', ic, True)
                remove_chars.append(ic)

        if not self._str_redundancy_check(lvl1_separator, lvl2_separator, lvl3_separator, comment_separator, *remove_chars, strip_args=False):
            raise ValueError('Arguments can\'t be redundant.')

        # --- structural string separation ---------------------
        if comment_separator not in [None, '']:
            raw_str = raw_str.split(comment_separator)[0].strip()
        # lvl1 to lvl3 separation
        lvl1_part_list = self._separate_level(raw_str, lvl1_separator, *remove_chars)
        element_list = []
        if lvl2_separator not in [None, '']:
            for lvl1_part in lvl1_part_list:
                lvl2_part_list = []
                if lvl2_separator in lvl1_part:
                    lvl2_part_list = self._separate_level(lvl1_part, lvl2_separator)
                    # single value lists only permitted in lvl1 for consistent return handling
                if lvl3_separator not in [None, ''] and lvl3_separator in lvl1_part:
                    # lvl3 separator could possibly be used in lvl1 without a lvl2 before it
                    if not lvl2_part_list:
                        lvl2_part_list = self._separate_level(lvl1_part, lvl3_separator)
                    # lvl3 separation inside of lvl2 parts
                    else:
                        for idx, lvl2_part in enumerate(lvl2_part_list):
                            if lvl3_separator in lvl2_part:
                                lvl3_part_list = self._separate_level(lvl2_part, lvl3_separator)
                                lvl3_part_list = lvl3_part_list[0] if len(lvl3_part_list) == 1 else lvl3_part_list
                            else:
                                lvl3_part_list = lvl2_part
                            lvl2_part_list[idx] = lvl3_part_list

                if lvl2_part_list:
                    lvl2_part_list = lvl2_part_list[0] if len(lvl2_part_list) == 1 else lvl2_part_list
                else:
                    lvl2_part_list = lvl1_part

                element_list.append(lvl2_part_list)

        elif lvl3_separator:
            raise ValueError('Level 3 separator specified but no level 2 separator. No gaps allowed.')
        else:
            element_list = lvl1_part_list

        # --- data conversion ----------------------------------
        if type_conversion:
            element_list = conv.convert_type_iter(element_list, decimal_separator)

        # --- split elements in fixed and umbrella parts -------
        if isinstance(umbrella_start_idx, int):
            if umbrella_start_idx <= len(element_list):
                fixed_elements = element_list[:umbrella_start_idx]
                umbrella_elements = element_list[umbrella_start_idx:]
            else:
                raise IndexError(f'The "umbrella_start_idx" must start _beyond_ all fixed elements. '
                                 f'Index is {umbrella_start_idx} but there are only {len(element_list)} elements.')
        else:
            fixed_elements = element_list

        return fixed_elements, umbrella_elements

    # ----------------------------------------------------------------------------------------------------------------------------------------------------
    def separate_header(self, header_line: str,
                        lvl1_separator: str = ',',
                        lvl2_separator: str = None,
                        umbrella_separator: str = None,
                        comment_separator: str = None) -> Tuple[list, Union[int, None]]:
        """Separates a header line, given as a cumulative string, into distinguished elements and provides location of logic-relevant separator.

        Separator characters have to be unique and can't be empty strings. If "comment separator" is specified, the string following the character
        will be ignored. No blank spaces allowed as separators.

        The "umbrella separator" in a header -put in just before the last header element-, signals the start of a container like area for all
        data lines below the header which collects an arbitrary amount of parameters in each line under the name of the last header
        element (acts as an "umbrella"), e.g. to enable a variably declared "parameter" list in addition to the "fixed parameters" before the
        umbrella separator. \n
        Ideally combined with lvl2 and lvl3 separators in data lines (to enable named expressions like "id=1;2;3, fx=100").

        :param header_line: raw header string to be separated.
        :param lvl1_separator: (Optional) character separating basic elements of a line from each other. Default: ",".
        :param lvl2_separator: (Optional) char. used to distinguish multiple parts of a single lvl1-element, e.g. "/" in "elmnt/elmnt_para" distinction. Default: None.
        :param umbrella_separator: (Optional) special char. used to signal start of an *args-like part with a single "umbrella" term for varying amount of values underneath. Default: None.
        :param comment_separator: (Optional) char. used to introduce an end-of-line comment (part of string will be ignored by function). Default: None
        :returns: 1st tuple element is list of separated header strings (or nested list if lvl2 sep. active), 2nd tuple element signals start idx of optional umbrella part.
        """

        umbrella_start_idx: int = None
        # --- input checks and basic formatting -----------------
        raw_str = self._str_arg_check('header_line', header_line, True, True)
        lvl1_separator = self._str_arg_check('lvl1_separator', lvl1_separator, False, False)
        lvl2_separator = self._str_arg_check('lvl2_separator', lvl2_separator, True, False)
        umbrella_separator = self._str_arg_check('umbrella_separator', umbrella_separator, True, False)
        comment_separator = self._str_arg_check('comment_separator', comment_separator, True, False)

        if not self._str_redundancy_check(lvl1_separator, lvl2_separator, umbrella_separator, comment_separator, strip_args=False):
            raise ValueError('Arguments can\'t be redundant.')

        # --- structural string separation ---------------------
        if comment_separator not in [None, '']:
            raw_str = raw_str.split(comment_separator)[0].strip()
        if umbrella_separator not in [None, ''] and umbrella_separator in raw_str:
            if raw_str.count(umbrella_separator) == 1:
                # filter with str.split strips elements and removes empty/spaced ones in one line, map still necessary for stripping
                fixed_str, umbrella_str = self._separate_level(raw_str, umbrella_separator)
            else:
                raise SyntaxError('Ambiguity in header: The "umbrella" separator can only appear once since it catches _all_ overflowing data elements.')
        else:
            fixed_str = raw_str
            umbrella_str = None

        # fixed part - lvl1 and lvl2 separation
        element_list = self._separate_level(fixed_str, lvl1_separator)
        if lvl2_separator not in [None, '']:
            element_list = [self._separate_level(e, lvl2_separator) if lvl2_separator in e else e for e in element_list]
            # single value lists only permitted in lvl1 for consistent return handling
            element_list = [e[0] if len(e) == 1 else e for e in element_list]

        # umbrella part
        if umbrella_str not in [None, '']:
            umbrella_start_idx = len(element_list)
            if lvl1_separator in umbrella_str:
                warnings.warn('\nFunction separate_headers() in line_parser.py: There are syntax issues in the collective "umbrella" part of the header. '
                              '\nThere shouldn\'t be additional separators after the umbrella separator (beside max. one comment separator).', stacklevel=2)
                # might still be valid if it's sitting empty front or back of string
                umbrella_list = self._separate_level(umbrella_str, lvl1_separator)
                if len(umbrella_list) == 1:
                    element_list.append(umbrella_list[0].strip())
                else:
                    raise SyntaxError('Only one name/element is allowed behind the "umbrella" separator, but multiple lvl1-separators have been detected.')
            else:
                element_list.append(umbrella_str.strip())

        distinct_elements = element_list

        return distinct_elements, umbrella_start_idx

    # Helpers ============================================================================================================================================
    @staticmethod
    def _str_redundancy_check(*args: Union[str, List[str]], strip_args: bool = True) -> bool:
        """Helper function to determine if any passed string arguments are redundant among themselves, includes string type check.

        None arguments will be ignored.

        :param args: function call arguments to be checked against each other. Takes single strings in combination with list of strings, too.
        :param strip_args: If True, the args will be stripped before evaluation against each other. Default: True
        :returns: True if arguments are unique among each other, False if even one is redundant.
        """

        arg_list = []
        for arg in args:
            if isinstance(arg, list):
                for a in arg:
                    arg_list.append(a)
            elif arg is not None:
                arg_list.append(arg)
        if any([not isinstance(arg, str) for arg in arg_list]):
            raise TypeError(f'Method requires all string arguments.')
        if strip_args:
            arg_list = list(map(str.strip, arg_list))
        if len(arg_list) != len(set(arg_list)):
            return False
        else:
            return True

    # ----------------------------------------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def _str_arg_check(var_name: str, var_value, none_valid: bool = False, is_text: bool = False) -> Union[str, None]:
        """Does a string type check and returns core string without leading and trailing spaces.

        Has option to tolerate None type and raises TypeError if nothing fits otherwise.

        :param var_name: name of variable displayed in error message if one is raised.
        :param var_value: value of the variable to be checked and shortened.
        :param none_valid: If True, None for var_value won't throw an error. Default: False.
        :returns: stripped string or None in case of tolerated None value.
        """

        if isinstance(var_value, str):
            stripped_str = var_value.strip()
            if stripped_str == '':
                raise ValueError(f'The argument "{var_name}" can\'t be an empty string.')
            elif stripped_str.isalnum() and not is_text:
                raise ValueError(f'The argument "{var_name}" can\'t be alphanumerical.')
            else:
                return stripped_str
        else:
            if var_value is not None:
                if not none_valid:
                    raise TypeError(f'The argument "{var_name}" must be a string.')
                else:
                    return None

    # ----------------------------------------------------------------------------------------------------------------------------------------------------
    def _separate_level(self, work_str: str, lvl_separator: str, *ignore_chars: str) -> List[str]:
        """Helper function to try a separation and strip the resulting part strings of leading and trailing spaces as well as specified characters.

        NO input checks included!

        :param work_str: base string to separate.
        :param lvl_separator: separator character.
        :param ignore_chars: (Optional) characters which will be removed from part strings
        :returns: list with separated part strings.
        """

        part_strings = list(map(str.strip, filter(str.split, work_str.split(lvl_separator))))
        if ignore_chars:
            part_strings = [self.full_str_strip(ps, *ignore_chars) for ps in part_strings]


        return part_strings

    # ----------------------------------------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def full_str_strip(input_str: str, *strip_chars: str) -> str:
        """Function able to strip characters completely from string, regardless of their positions in the string.

        :param input_str: string to be stripped.
        :param strip_chars: characters to be stripped from the working string.
        :returns: stripped string.
        """

        output_str = input_str

        if any([not isinstance(c, str) for c in strip_chars]):
            raise TypeError(f'Method requires all string arguments.')

        for c in strip_chars:
            parts = output_str.split(c)
            output_str = ''.join(parts)

        return output_str


# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
if __name__ == "__main__":

    from file_parser_ini import FileParserIni

    cui_path = "../../../../data/99_PARSERTEST/config_user_input.ini"
    sc_path = "../../../data/99_PARSERTEST/simulation_config.ini"
    spi_path = "../../../data/99_PARSERTEST/set_parameter_input.ini"
    ssl_path = "../../../../data/99_PARSERTEST/session_sensor_list.ini"
    ifp = FileParserIni()

    line_cui_opt_key_single = ifp.get_option_value('settings', 'target_file_names', cui_path)
    line_cui_opt_key_multi = ifp.get_option_value('settings', 'cams_trg_pose_est', cui_path)
    line_cui_nr_key = ifp.get_option_value('coordinate_systems', '002', cui_path)
    line_cui_comment = ifp.get_option_value('object_points', '0004', cui_path)

    line_sc_header_umbrella_part = ifp.get_option_value('simulation___state_parameter_assignment', 'head', sc_path)
    line_sc_2lvl_nested = ifp.get_option_value('simulation___state_parameter_assignment', '0003', sc_path)
    line_ssl_header_umbrella_part = ifp.get_option_value('DEFAULT', '000', ssl_path)
    line_spi_header_detail = ifp.get_option_value('parameter_value', 'para', spi_path)
    line_ssl_3lvl_nested = ifp.get_option_value('DEFAULT', '005', ssl_path)

    # DO TESTS ----------------------------------------------------------
    lp = LineParser()
    header, umbr_idx = lp.separate_header(line_ssl_header_umbrella_part, lvl1_separator=';', lvl2_separator=None, umbrella_separator='#', comment_separator=None)
    print(*header, sep=',\n')
    print('Umbrella start index: ', umbr_idx)

    fixed, umbrella = lp.separate_data(line_ssl_3lvl_nested, lvl1_separator=';', lvl2_separator='=', lvl3_separator=',', decimal_separator='.', comment_separator='#', umbrella_start_idx=umbr_idx, ignore_chars=None)
    pass
