# -*- coding: utf-8 -*-
'''The PR3 Policy of possibilistic reward family.
  Reference: [M. Martín, A. Jiménez & A. Mateos, Neurocomputing, 2018].'''

__author__ = "Miguel Martin, Antonio Jimenez, Alfonso Mateos"
__version__ = "1.0"


from math import sqrt, log, exp
import random as rand
import numpy as np
import decimal
decimal.getcontext().prec = 100
from IndexPolicy import IndexPolicy
import time

class ProbabilityMatching(IndexPolicy):
    """Class that implements the PR3 policy.
    """

    def __init__(self, nbArms, amplitude=1., lower=0., weight=0.5, confidence=0.1):
        self.nbArms = nbArms
        self.factor = amplitude
        self.lower = lower
        self.nbDraws = dict()
        self.cumReward = dict()
        self.cumReward2 = dict()
        self.sampleMeans = dict()
        self.vars = dict()
        self.outDraws = dict()
        # ucb var variables:
        self.Ks = dict()
        self.Exs = dict()
        self.Ex2s = dict()
        self.ucb_vars = dict()
        self.sample_var = dict()
        self.arm_indexes = np.zeros(nbArms)
        self.arm_prob = np.zeros(nbArms)
        self.update_pending = True

        self.meansDiff = 1
        self.confidence=confidence
        self.reamining_value = {"PR2": 1, "PR3": 1}

        for i in range(nbArms):
            self.outDraws[i] = np.array([])

        self.weight = weight
        print self.weight
        self.scale = 1

    def startGame(self):

        self.t = 1

        for arm in range(self.nbArms):
            self.nbDraws[arm] = 0
            self.cumReward[arm] = 0.0
            self.cumReward2[arm] = 0.0
            self.vars[arm] = 1.0
            self.Ks[arm] = 0.0
            self.Exs[arm] = 0.0
            self.Ex2s[arm] = 0.0
            self.ucb_vars[arm] = 1
            self.sampleMeans[arm] = 0
            self.sample_var[arm] = 0
            
        self.reamining_value = {"PR2": 1, "PR3": 1}

    def computeIndex(self, arm):
        if self.update_pending:
            arm_weights = self.sampling(1000, 'PR2')
            self.sampling(1000, 'PR3')
            self.arm_prob = np.array(arm_weights)/np.sum(arm_weights)
            self.update_pending = False
            
        if arm == 0:
            self.update_arm_indexes(self.arm_prob)
            
        return self.arm_indexes[arm]

    def getReward(self, arm, reward):
        self.nbDraws[arm] += 1
        self.cumReward[arm] += reward
        self.cumReward2[arm] += float(reward)/self.factor

        norm_reward = float(reward)/self.factor
    
        # Calculate upper confidence variance

        self.update_ucb_var(arm, norm_reward)
        
   
        self.t += 1
        self.update_pending = True
    
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

    def getConfidence_by_sampling(self, sims, eval_method='PR2'):
        return self.reamining_value[eval_method]
        
    def sampling(self, sims, eval_method='PR2'):
        sim_results = np.zeros((sims, self.nbArms))
        sim_bests = np.zeros(self.nbArms)
        for i in range(sims):
            sim = np.zeros(self.nbArms)
            for arm in range(self.nbArms):
                sim[arm] = self.getSample(arm, eval_method)
            best = np.argmax(sim)
            sim_bests[best] += 1
            sim_results[i, :] = sim
            
        def safe_division(n, d):
            if d == 0:
                return 0
            return n/d

        # means = [safe_division(i, j) for i, j in zip(self.cumReward.values(),  self.nbDraws.values())]
        best_arm = np.argmax(sim_bests)
        best_confidence = np.max(sim_bests)/np.sum(sim_bests)
        remaining_values = (np.max(sim_results, 1) - sim_results[:, best_arm])/sim_results[:, best_arm]
        # print("confidence: " + str(best_confidence) + "; remaining: " + str(np.percentile(remaining_values, 95)))
        # print(self.t)
        # return best_confidence, np.percentile(remaining_values, 95), means, self.nbDraws.values()
        self.reamining_value[eval_method] = np.percentile(remaining_values, 95)
        return sim_bests
    
    def update_arm_indexes(self, prob_vector):
        self.arm_indexes = np.zeros(self.nbArms)
        res = np.random.choice(int(self.nbArms), p=prob_vector)
        self.arm_indexes[res] += 1
    
    def getConfidence(self, sims, eval_method='PR2'):
        
        if eval_method != 'empirical_bernstain': 
            return super(ProbabilityMatching, self).getConfidence(sims, eval_method)
        else:
            return 0
