import os
import numpy as np
import datetime as d
import random as r
import copy as c

from constants import *
from utilities import *
# from data_gen import Robot, DataCollection
# from ik_simulator import IKTable, IKSimulator
# from ikpy.chain import Chain

def IKCCD(chain, target_pos, target_ori=None, initial=None, threshold=1e-5, maxIter=100000):
    def optimize_basis(x):
        y = chain.active_to_full(x, initial)
        fk = chain.forward_kinematics(y)

        return fk

    if initial is None:
        joint = [0.0*len(chain)]
        def error(x):
            fk = optimize_basis(x)
            err = np.linalg.norm(fk[:3, 3] - target_pos)

            return err
    else:
        joint = initial
        def error(x):
            fk = optimize_basis(x)
            err = np.linalg.norm(fk[:3, 3] - target_pos)
            err_ori = np.linalg.norm(fk[:3, 2] - target_ori)

            return err + err_ori

    t_start = d.datetime.now()
    statue = False
    for i in range(maxIter):
        err = error(joint)
        if err < threshold:
            status = True
            break

        for j in range(len(chain)-1, -1, -1):
            if chain.active_links_mask[j]:
                pass

    time = d.datetime.now() - t_start

    return joint, time, status, i
