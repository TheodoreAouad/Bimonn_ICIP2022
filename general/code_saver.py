from typing import Tuple
import fnmatch
from os.path import join, exists
from shutil import rmtree, copytree

from .utils import get_next_same_name


class CodeSaver:

    def __init__(self, src_path: str, temporary_path: str = None, ignore_patterns: Tuple[str] = ('*__pycache__*',)):
        self.src_path = src_path
        self.temporary_path = get_next_same_name(temporary_path, 'temporary') if temporary_path is not None else None
        self.ignore_patterns = self.apply_ignore_patterns(*ignore_patterns)

    def save_in_temporary_file(self, dst_path: str = None):
        # Save code
        if self.temporary_path is None:
            self.temporary_path = get_next_same_name(dst_path, 'temporary')
        if exists(self.temporary_path):
            rmtree(self.temporary_path)
        copytree(self.src_path, join(self.temporary_path, 'code'), ignore=self.ignore_patterns)

    def save_in_final_file(self, dst):
        # Save code
        copytree(join(self.temporary_path, 'code'), dst, ignore=self.ignore_patterns)

    def delete_temporary_file(self):
        rmtree(self.temporary_path)

    def apply_ignore_patterns(self, *patterns):
        """Function that can be used as copytree() ignore parameter.

        Patterns is a sequence of glob-style patterns
        that are used to exclude files"""
        def _ignore_patterns(path, names):
            ignored_names = []
            for pattern in patterns:
                ignored_names.extend(fnmatch.filter(names, pattern))
            return set(ignored_names)
        return _ignore_patterns
