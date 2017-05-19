#!/usr/bin/env python
from distutils.core import setup

def find_version(path):
    import re
    # path shall be a plain ascii text file.
    s = open(path, 'rt').read()
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              s, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Version not found")

setup(name="abopt", version=find_version("abopt/version.py"),
      author="Grigor Aslanyan, Yu Feng, et al",
      maintainter="Yu Feng",
      maintainter_email="rainwoodman@gmail.com",
      description="Optimization of abstract data types in Python",
      zip_safe=True, # this should be pure python
      packages=["abopt", "abopt.tests"],
      license='GPLv3',
      install_requires=['numpy'], # maybe not needed by the core, but we need numpy for testing
      )
