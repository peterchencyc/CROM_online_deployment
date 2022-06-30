import os
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)
parentdir = os.path.dirname(currentdir)
rootrdir = os.path.dirname(parentdir)
commo_dir = os.path.join(rootrdir,'common')
sys.path.append(commo_dir)


import numpy as np
import json