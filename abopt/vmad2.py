import warnings

warnings.warn("vmad2 has been deprecated and moved to abopt.legacy.vmad2. "
              "Replace abopt.vmad2 with abopt.legacy.vmad2. "
              "Use the new python package vmad for new models.", FutureWarning)

from abopt.legacy.vmad2 import *
