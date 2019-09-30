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

class Egreedy(IndexPolicy):
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
        self.reamining_value = 1
        self.reamining_values = {"PR2": 1, "PR3": 1}

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
            
        self.reamining_value = 1
        self.reamining_values = {"PR2": 1, "PR3": 1}

    def computeIndex(self, arm):
        if self.nbDraws[arm] < 1:
            return 1
        else:
            if not all([self.nbDraws[i] >= 1000  and  self.cumReward[i] >= 50 for i in range(self.nbArms)]):
                #print("arms: " + str(self.nbDraws.values()))
                #print("rewards: " + str(min(self.cumReward)))
                return 0


            if self.update_pending:  
                arm_weights = self.bernstain_confidence()
                self.arm_prob = np.array(arm_weights)/np.sum(arm_weights)
                self.reamining_values["PR2"] = super(Egreedy, self).getConfidence_by_sampling(1000, "PR2")
                self.reamining_values["PR3"] = super(Egreedy, self).getConfidence_by_sampling(1000, "PR3")
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
        print(self.t)
        return self.reamining_values[eval_method]
        
    
    def update_arm_indexes(self, prob_vector):
        self.arm_indexes = np.zeros(self.nbArms)
        if rand.random() > 0.8:
            res = np.random.choice(int(self.nbArms), p=prob_vector)
        else:
            res = np.random.choice(int(self.nbArms))

        self.arm_indexes[res] += 1
    
    def bernstain_bound(self, arm, bound_type='max'):
        n = self.nbDraws[arm]
        if n == 0:
            if bound_type == 'max':
                return 1
            else:
                return 0
        
        #var = min([0.25, self.sample_var[arm] + np.sqrt(np.log(0.025)/(-2*self.nbDraws[arm]))])

        mu = self.cumReward[arm]/self.nbDraws[arm]
        #bound = ((np.log(2/0.025)/n)*(np.sqrt(n)/3 + np.sqrt(n/9 + 2*var*n*n/np.log(2/0.025))))/np.sqrt(n)
        bound = np.sqrt(self.sample_var[arm]*2*np.log(3/0.05)/n) + 3*np.log(3/0.05)/n
        if bound_type == 'max':
            return mu + bound
        else:
            return mu - bound

    def bernstain_confidence(self):
        def safe_div(a,b):
            if b == 0 and a == 0:
                return 0
            if b == 0:
                return 1
            return float(a)/float(b)
        

        get_bound = self.bernstain_bound

        best_arm = np.argmax([safe_div(self.cumReward[arm], self.nbDraws[arm]) for arm in range(self.nbArms)])
        #print("best arm:" + str(best_arm))
        best_sample_mean = np.max([safe_div(self.cumReward[arm], self.nbDraws[arm]) for arm in range(self.nbArms)])
        #print("best_sample_mean:" + str([safe_div(self.cumReward[arm], self.nbDraws[arm]) for arm in range(self.nbArms)]))
        ucb_min = get_bound(best_arm, 'min')
        #print("ucb_min:" + str(ucb_min))
        ucb_max = np.max([get_bound(arm, 'max') for arm in range(self.nbArms) if arm != best_arm])

        #print("ucb_max:" + str([self.bernstain_bound(arm, 'max') for arm in range(self.nbArms) if arm != best_arm]))
        #print("confidence :" + str(safe_div((ucb_max - ucb_min), best_sample_mean)))
        self.reamining_value = safe_div((ucb_max - ucb_min), best_sample_mean)

        ucb_arms = [get_bound(i, 'max') for i in range(self.nbArms)]
        first_index = 0
        first_value = 0
        second_index = 0
        second_value = 0
        for i,v in enumerate(ucb_arms):
            if v > first_value:
                second_index = first_index
                second_value = first_value
                first_index = i
                first_value = v

            elif v > second_value:
                second_index = i
                second_value = v

        def first_or_second(arm):
            if arm in [first_index, second_index]:
                return 1.0
            else: 
                return 0.0
        

        return [first_or_second(i) for i in range(self.nbArms)]

    def getConfidence(self, sims, eval_method='PR2'):
        if eval_method != 'empirical_bernstain': 
            return super(Egreedy, self).getConfidence(sims, eval_method)
        else:
            return super(Egreedy, self).getConfidence(sims, eval_method)
            #return self.reamining_value

       