class Constraints:
    def __init__(self, constraints: dict):
        """Class contains the constraints for a specific .ini file parser. Performs a self-check on the constraints bound to it.

        Passed constraints contain all expected and required section names, possible headers or mandatory options as well as minimum or maximum
        expected count of elements in a data row per section.
        Template with dict keys to use:

        constraints = {
            'section_constraints': {
                '<section_name>': {
                    'head': List[str] | None,           Default: None   -> set if header must be considered
                    'head_idx_for_subdict': int | None, Default: None   -> head[idx] to be used as subdict-key for elements
                    'min_elements': int | None,         Default: None
                    'max_elements': int | None,         Default: None
                    'force_options': List[str] | None,  Default: None
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

        :param constraints: Expected structure as above.
        """

        self.constraints: dict = constraints
        self.self_check()

    def self_check(self) -> bool:
        """Verification determining if structure of constraint dictionary was defined correctly by the developer.

        Checks:     \n
        -   existence of main structural dict keys
        -   section count consistency in section and structural constraints
        -   data format of category values in structure constraints

        :returns: True if constraint dictionary is valid.
        """

        if any([False if e in self.constraints else True for e in ['section_constraints', 'structure_constraints']]):
            raise ValueError('The sections "section_constraints" and "structure_constraints" must be in the constraints dictionary. \n'
                             'Refer to the documentation for the constraints dictionary template.')
        if not any([False if self.constraints['structure_constraints'][k] in [None, [], [[]]] else True for k in self.constraints['structure_constraints']]):
            raise ValueError('None of the structure constraints have been configured.')

        self.__check_sect_consistency()
        self.__check_struct_constr()
        self.__check_sect_constr()

        return True

    # ------------------------------------------------------------------------------------------------------------------------------------------------
    def __check_sect_consistency(self):
        sect_in_sect_constr = list(self.constraints['section_constraints'].keys())
        sect_in_struct_constr = self.__flatten(list(self.constraints['structure_constraints'].values()))
        if len(sect_in_struct_constr) != len(set(sect_in_struct_constr)):  # it's important these two clear before the other checks below
            raise ValueError('There are duplicates in the structure constraint definition. Each section name can only appear in one category.')
        # section consistency check and back check between constraint keys
        missing_in_struct_constr = [s for s in sect_in_sect_constr if s not in sect_in_struct_constr]
        if missing_in_struct_constr:
            raise ValueError(f'Section(s) "{", ".join(missing_in_struct_constr)}" in section constraints do(es) not appear in structure constraints. '
                             f'The constraint dictionary must be consistent.')
        missing_in_sect_constr = [s for s in sect_in_struct_constr if s not in sect_in_sect_constr]
        if missing_in_sect_constr:
            raise ValueError(f'Section(s) "{", ".join(missing_in_sect_constr)}" in structure constraints do(es) not appear in section constraints. '
                             f'The constraint dictionary must be consistent.')

    # ------------------------------------------------------------------------------------------------------------------------------------------------
    def __check_struct_constr(self):
        stc_categories = ['MUST', 'OR', 'XOR', 'OPTIONAL']
        for key in stc_categories:
            try:
                if self.constraints['structure_constraints'][key] in [None, '', [[]]]:
                    raise KeyError
            except KeyError:
                self.constraints['structure_constraints'][key] = []
        # distinguish types because only OR and XOR can have a more complicated structure to enable comparisons
        simple_lists = [self.constraints['structure_constraints']['MUST'], self.constraints['structure_constraints']['OPTIONAL']]
        nested_lists = [self.constraints['structure_constraints']['XOR'], self.constraints['structure_constraints']['OR']]
        self.__subdict_type_check(simple_lists, False, 0)
        self.__subdict_type_check(nested_lists, True, 2)

    # ------------------------------------------------------------------------------------------------------------------------------------------------
    def __check_sect_constr(self):
        sec_categories = ['head', 'head_idx_for_subdict', 'force_options', 'min_elements', 'max_elements']
        for sect in self.constraints['section_constraints']:
            section_constraints = self.constraints['section_constraints'][sect]
            for constr in sec_categories:
                try:
                    if section_constraints[constr] in ['', []]:
                        raise KeyError
                except KeyError:
                    self.constraints['section_constraints'][sect][constr] = None

            string_lists = ['force_options', 'head']
            for constr in string_lists:
                if section_constraints[constr]:
                    if isinstance(section_constraints[constr], list):
                        self.__subdict_type_check([section_constraints[constr]], True, 0)
                    else:
                        raise TypeError(f'Section {sect}: "{constr}" constraints only accept list of strings.')

            if section_constraints['head_idx_for_subdict'] not in ['', None]:
                if not section_constraints['head']:
                    raise ValueError(f'Section {sect}: "head_idx_for_subdict" constraints can only be set in combination with a "head" constraint.')
                elif not isinstance(section_constraints['head_idx_for_subdict'], int):
                    raise TypeError(f'Section {sect}: "head_idx_for_subdict" constraints only accept integers.')
                elif section_constraints['head_idx_for_subdict'] >= len(section_constraints['head']):
                    raise IndexError(f'Section {sect}: "head_idx_for_subdict" constraint is out of reach for "head" constraint.')
            else: section_constraints['head_idx_for_subdict'] = None        # explicitly None for easy differentiation from 0 when using this option

            integers = ['min_elements', 'max_elements']
            for constr in integers:
                if section_constraints[constr] and not isinstance(section_constraints[constr], int):
                    raise TypeError(f'Section {sect}: "{constr}" constraints only accept integers.')

    # ------------------------------------------------------------------------------------------------------------------------------------------------
    @classmethod
    def __subdict_type_check(cls, iterable: list, nested: bool, min_elements: int) -> bool:
        """Internal method verifying the input data types of the constraints are as expected.

        Input is normal list or nested list of strings as expected.
        NO check against parsed file content. Recursive for nested lists.

        :param iterable: (content of the) individual structure constraint subkeys to be checked in one call.
        :param nested: True if iterable CAN contain nested lists, False if this should be considered improper input.
        :param min_elements: 0 if it doesn't matter how many section names are in the list, 2 if multiple names are required (for later OR, XOR comparison).
        :returns: True if constrain dictionary is OK.
        """

        for constraint in iterable:
            if constraint:
                if not isinstance(constraint, list):
                    raise TypeError(f'"{constraint}" constraint needs to be either {"(nested) list" if nested else "a list"} of strings or None.')
                else:
                    element_count = 0
                    for part in constraint:
                        if isinstance(part, list):
                            if nested:
                                cls.__subdict_type_check([part], False, min_elements)
                            else:
                                raise TypeError(f'Unexpected level of nested lists encountered. Expected string instead of {part}.')
                        elif isinstance(part, str):
                            element_count += 1
                        else:
                            raise TypeError(f'"{part}" of "{constraint}" in constraints is not a string.')
                    if element_count < min_elements and not nested:
                        raise ValueError(f'Structure constraints OR and XOR need at least {min_elements} to compare.')
        return True

    # ------------------------------------------------------------------------------------------------------------------------------------------------
    @classmethod
    def __flatten(cls, x):
        """Utility flattening variably nested lists of strings."""
        if isinstance(x, list):
            return [a for i in x for a in cls.__flatten(i)]
        else:
            return [x]
