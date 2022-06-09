import argparse
import json
import numpy as np
import torch.nn as nn
from tqdm import tqdm

import sys
import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from commons import trainModel, testModel, saveModel, makeEnv, utils
from sb3_contrib.trpo.trpo import TRPO
from adaptive_dr.BayRn import get_bc

# INIZIALIZZO TRPO CON LA MIGLIOR CONFIGURAZIONE TROVATA IN STEP4

# TRAIN TRPO_BEST_CONFIG SU SOURCE USANDO I BOUNDS DI GET_BC() DI BAYRN

# TEST TRPO_BEST_CONFIG SU TARGET