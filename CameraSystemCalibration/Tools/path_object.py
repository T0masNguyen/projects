import pathlib

"""
get the current path structure ....
"""


# NEW CLASS TO REPLACE PathElement AND PathConfig IN COMBINATION WITH configuration.ini AND paths.py
class PathObject:
    def __init__(self, path: pathlib.Path):
        if isinstance(path, pathlib.Path):
            self._path: pathlib.Path = path
        else:
            raise TypeError(f'Expected pathlib.Path object. Got {type(path)}.')
        self._name: str = self._path.name

    def __str__(self):
        return f'PathObject: {self.path}'

    def __repr__(self):
        return f'PathObject(pathlib.Path("{self.path}"))'

    @property
    def path(self):
        return self._path

    @property
    def name(self):
        return self._name


# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
if __name__ == '__main__':
    pass
