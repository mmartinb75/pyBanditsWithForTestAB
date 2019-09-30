# -*- coding: utf-8 -*-
'''Delays generation for online architecture'''

__author__ = "Miguel Martin, Antonio Jimenez, Alfonso Mateos"
__version__ = "1.0"


from math import isinf, exp, log
from random import random, seed


class Not_Delayed_conf:

    def getDelayedReward(self, reward, t):
        return reward, 1

    def restart(self):
        pass






