# Copyright 2021 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.

from hitmix.hitting_time_moments import *

import warnings
def ignore_warnings(ignore=True):
    if ignore:
        warnings.simplefilter('ignore')
    else:
        warnings.simplefilter('default')

ignore_warnings(True)
