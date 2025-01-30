import copy
from typing import List


def convert_type_iter(data_input: List[str], decimal_separator: str = '.'):
    """Tries to find the plausible data type of passed string content by looking for type conditions and converts the data accordingly.

    Supports up to 5 nested dimensions.

    :param data_input: one or multiple strings to be converted into their plausible data types.
    :param decimal_separator: (Optional) helps identifying values to convert to floats. If None is passed, no float conversion happens. Default: ".".
    :returns: converted data in the input format.
    """

    inp = copy.deepcopy(data_input)
    if isinstance(inp, str):
        data_output = convert_type_single(inp)
    elif isinstance(inp, (list, tuple)):
        data_output = inp
        for nidx, val in traverse(inp):
            # if there is or will be the option to access nested list items by a tuple index without manual distinction, feel free to update function
            if len(nidx) == 1:
                data_output[nidx[0]] = convert_type_single(val, decimal_separator)
            elif len(nidx) == 2:
                data_output[nidx[0]][nidx[1]] = convert_type_single(val, decimal_separator)
            elif len(nidx) == 3:
                data_output[nidx[0]][nidx[1]][nidx[2]] = convert_type_single(val, decimal_separator)
            elif len(nidx) == 4:
                data_output[nidx[0]][nidx[1]][nidx[2]][nidx[3]] = convert_type_single(val, decimal_separator)
            elif len(nidx) == 5:
                data_output[nidx[0]][nidx[1]][nidx[2]][nidx[3]][nidx[4]] = convert_type_single(val, decimal_separator)
            else:
                raise IndexError('Only 5 nested dimensions supported.')
    else:
        raise TypeError('Function input either needs to be string or list of strings or tuple of strings.')

    return data_output


# ----------------------------------------------------------------------------------------------------------------------------------------------------
def convert_type_single(data_string: str, decimal_separator: str = '.'):
    """Tries to find the plausible data type of passed string by looking for type conditions and converts the input string accordingly.

    Tries int, float, None-type and bool. If nothing fits, the original string is returned.

    :param data_string: string to be converted into its plausible data type.
    :param decimal_separator: (Optional) helps identifying values to convert to floats. If None is passed, no float conversion happens. Default: ".".
    :returns: converted value.
    """

    if not isinstance(data_string, str):
        raise TypeError('Function requires string input.')

    data_output = data_string
    converted = False
    if decimal_separator not in [None, ''] and decimal_separator in data_string:
        data_string = data_string.replace(decimal_separator, '.')
        try:
            data_output = float(data_string)
        except ValueError:
            pass
    else:
        if string_is_none(data_string):
            data_output = None
            converted = True
        elif string_is_true(data_string):
            data_output = True
            converted = True
        elif string_is_false(data_string):
            data_output = False
            converted = True

        if not converted:
            try:
                data_output = int(data_string)
            except ValueError:
                pass

    return data_output


# ----------------------------------------------------------------------------------------------------------------------------------------------------
def string_is_true(str_value: str) -> bool:
    """Checks if a single string value is a boolean with value "True". Accepted strings values for True: "true", "t", "yes", "y", "on".

    :param str_value: input string to check. All upper, lower case or capitalized.
    :returns: True only if string is a boolean with VALUE "True". Otherwise returns False.
    """
    accptd = ['true', 'True', 't', 'T', 'yes', 'Yes', 'y', 'Y', 'on', 'On', 'TRUE', 'YES', 'ON']
    if not isinstance(str_value, str):
        raise TypeError(f'Input value "{str_value}" is not a string.')
    else:
        return str_value in accptd


# ----------------------------------------------------------------------------------------------------------------------------------------------------
def string_is_false(str_value: str) -> bool:
    """Checks if a single string value is a boolean with value "False". Accepted strings values for False: "false", "f", "no", "n", "off" .

    :param str_value: input string to check. All upper, lower case or capitalized.
    :returns: True only if string is a boolean with VALUE "False". Otherwise returns False.
    """

    accptd = ['false', 'False', 'f', 'F', 'no', 'No', 'n', 'N', 'off', 'Off', 'FALSE', 'NO', 'OFF']
    if not isinstance(str_value, str):
        raise TypeError(f'Input value "{str_value}" is not a string.')
    else:
        return str_value in accptd


# ----------------------------------------------------------------------------------------------------------------------------------------------------
def string_is_none(str_value: str) -> bool:
    """Checks if a single string value is a boolean with value "None". Accepted strings values for None: "none", "null", "void", "".

    :param str_value: input string to check. All upper, lower case or capitalized.
    :returns: True, if string represents a None-like expression (see description). Otherwise returns False.
    """

    accptd = ['none', 'None', '', 'null', 'Null', 'void', 'Void']
    if not isinstance(str_value, str):
        raise TypeError(f'Input value "{str_value}" is not a string.')
    else:
        return str_value.strip() in accptd


# ----------------------------------------------------------------------------------------------------------------------------------------------------
def traverse(item, tree_types: tuple = (list, tuple)):
    """Generator yielding each value of a inhomogeneous, arbitrarily nested list and its corresponding n-dimensional nested list index."""

    if isinstance(item, tree_types):
        for idx, value in enumerate(item):
            for subtuple in traverse(value):
                subidx: tuple = subtuple[0]
                subvalue = subtuple[1]
                if not subidx:
                    nest_idx = (idx,)
                else:
                    nest_idx = (idx,) + subidx
                yield nest_idx, subvalue
    else:
        yield None, item
