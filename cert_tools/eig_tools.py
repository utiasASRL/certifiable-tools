# Maths
import numpy as np
import scipy.sparse as sp
# 
import sys, os

def min_eigs_lanczos(H, k = 6):
    """Use the Lanczos process to get an approximation of minimum eigenpairs.
    """
    