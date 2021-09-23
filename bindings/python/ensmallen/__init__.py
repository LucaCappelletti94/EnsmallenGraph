"""Module offering fast graph processing and graph datasets."""

import cpuinfo

required_flags = ("sse", "sse2", "ssse3", "sse4_1", "sse4_2",
                  "avx", "avx2", "bmi1", "bmi2", "popcnt")

flags = cpuinfo.get_cpu_info()["flags"]

unavailable_flags = set(required_flags) - set(flags)

if len(unavailable_flags) > 0:
    raise ValueError(
        (
            "This library was compiled assuming that SIMD instruction "
            "commonly available in CPU hardware since 2013 are present "
            "on the machine where this library is intended to run.\n"
            "On the current machine, the flags {} are not available.\n"
            "You could still compile Ensmallen on this machine and have "
            "a version of the library that can execute here, but the library "
            "has been extensively designed to use SIMD instructions, so "
            "you would have a version slower than the one provided on Pypi."
        ).format(unavailable_flags)
    )

from .ensmallen import preprocessing  # pylint: disable=import-error
from .ensmallen import Graph  # pylint: disable=import-error
from .ensmallen import edge_list_utils  # pylint: disable=import-error
from . import datasets

__all__ = ["edge_list_utils", "Graph", "preprocessing", "datasets"]
