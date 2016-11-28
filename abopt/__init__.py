from .version import __version__
from .abopt import *

from numpy.testing import Tester
test = Tester().test
bench = Tester().bench
