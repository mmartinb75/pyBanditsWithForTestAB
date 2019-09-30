# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
'''The Possibilistic Policy.
  Reference: [Miguel Martin].'''

__author__ = "Miguel Martin"
__version__ = "1.0"


from math import sqrt, log, exp
import random as rand
import numpy as np


from IndexPolicy import IndexPolicy


class TestAB(IndexPolicy):
    """Class that implements the UCB-V policy.
    """

    def __init__(self, nbArms, amplitude=1., lower=0., scale=1):
        self.nbArms = nbArms
        self.factor = amplitude
        self.nbDraws = dict()
        self.cumReward = dict()
        self.confidence = 0.1

        # ucb var variables:
        self.Ks = dict()
        self.Exs = dict()
        self.Ex2s = dict()
        self.ucb_vars = dict()
        self.sample_var = dict()

    def startGame(self):
        self.t = 1
        for arm in range(self.nbArms):
            self.nbDraws[arm] = 0
            self.cumReward[arm] = 0.0
            self.Ks[arm] = 0.0
            self.Exs[arm] = 0.0
            self.Ex2s[arm] = 0.0
            self.ucb_vars[arm] = 1
            self.sample_var[arm] = 0

    def computeIndex(self, arm):
        return 0

    def getReward(self, arm, reward):
        self.nbDraws[arm] += 1
        self.cumReward[arm] += reward

        norm_reward = float(reward)/self.factor

        self.update_ucb_var(arm, norm_reward)

        self.t += 1
    
    def update_ucb_var(self, arm, norm_reward):
        if self.nbDraws[arm] == 1:
            self.Ks[arm] = norm_reward

        self.Exs[arm] += float(norm_reward - self.Ks[arm])
        self.Ex2s[arm] += (norm_reward - self.Ks[arm]) * (norm_reward - self.Ks[arm])

        if self.nbDraws[arm] < 2:
            self.sample_var[arm] = self.Ex2s[arm] - (self.Exs[arm]*self.Exs[arm])
        else:
            den = self.nbDraws[arm]
            self.sample_var[arm] = (self.Ex2s[arm] - (self.Exs[arm]*self.Exs[arm])/den) / (den-1)
        
        if self.sample_var[arm] == 0:
            self.ucb_vars[arm] = 0.25
        else:
            self.ucb_vars[arm] = min([0.25, self.sample_var[arm] + sqrt(log(self.confidence)/(-2*self.nbDraws[arm]))])















