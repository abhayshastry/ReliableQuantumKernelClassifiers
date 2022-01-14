import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time, os, itertools, pickle, sys
srcpath = os.path.abspath(os.path.join(os.path.abspath(''),  '..',  'src'))
sys.path.append(srcpath)
from q_kernels import *
from exp_utils import *
from qml_utils import *
from math_utils import *

kd_list = [("QAOA", "Checkerboard" ), ("Havliscek,2", "Two_Moons" ), ("Circ-Hubr,2", "Two_Moons" ),
 ("Circ-Hubr", "Generated" ), ("Havliscek,2", "Checkerboard" ),  ("QAOA,2", "Two_Moons" ),  ("Angle,2", "Two_Moons" ), ("Angle", "Generated" ),  ("QAOA" ,"Generated" ), ("Havliscek", "Generated" ),
 ("QAOA,2", "SymDonuts" ), ("QAOA", "Two_Moons" ),  ("Circ-Hubr,2", "Checkerboard" ), ("QAOA,2", "Checkerboard" )]

C = 4 

def find_N_star(key, m, C, N_trials = 100,  delta = 0.1):
