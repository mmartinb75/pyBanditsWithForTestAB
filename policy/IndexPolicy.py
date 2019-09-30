# -*- coding: utf-8 -*-
'''Generic index policy.'''

__author__ = "Olivier Cappé, Aurélien Garivier"
__version__ = "$Revision: 1.5 $"


from random import choice
import random as rand
import numpy as np

from scipy import stats

from Policy import *

class IndexPolicy(Policy):
    """Class that implements a generic index policy."""

#  def __init__(self):

#  def computeIndex(self, arm):

    def choice(self):
        """In an index policy, choose at random an arm with maximal index."""
        #init_time = time.time()
        index = dict()
        for arm in range(self.nbArms):
            index[arm] = self.computeIndex(arm)
        maxIndex = max (index.values())
        bestArms = [arm for arm in index.keys() if index[arm] == maxIndex]
        #elapsed_time = (time.time() - init_time)*1000
        #print("elapsed_time: " + str(elapsed_time))
        return choice(bestArms)
    def getSample(self, arm, eval_method='PR2'):
        if self.nbDraws[arm] < 1:
            return rand.betavariate(1, 1)
        else:
            if eval_method == 'PR3':
                ucb_vars = min([0.25, self.sample_var[arm] + np.sqrt(np.log(2/0.025)/(2*self.nbDraws[arm]))])
                mu1 = self.cumReward[arm]/self.nbDraws[arm]
                mu = mu1/self.factor
                s = self.nbDraws[arm]
                r = max([1, mu*(1-mu)/ucb_vars])
                n = s*r
                a = mu*n
                b = n - a
                sample = rand.betavariate(1+a, 1+b)
        
            else:
                a = self.cumReward[arm]
                b = self.nbDraws[arm] - a
                sample = rand.betavariate(1+a, 1+b)

            return sample

    def bernstain_bound(self, arm, bound_type='max', type='PR2'):
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

    def get_pr3_params(self, arm):
        #ucb_vars = min([0.25, self.sample_var[arm] + np.sqrt(np.log(0.025)/(-2*self.nbDraws[arm]))])
        ucb_vars = min([0.25, self.sample_var[arm] + np.sqrt(np.log(2/0.025)/(2*self.nbDraws[arm]))])
        mu1 = self.cumReward[arm]/self.nbDraws[arm]
        mu = mu1/self.factor
        s = self.nbDraws[arm]
        r = max([1, mu*(1-mu)/self.ucb_vars[arm]])
        n = s*r
        a = mu*n
        b = n - a

        return a, b
    
    def get_pr2_params(self, arm):
        a = self.cumReward[arm]
        b = self.nbDraws[arm] - a
        #print("a : " + str(a) + ", b: " + str(b))

        return a, b

    def pr_bound(self, arm, bound_type='max', type='PR2'):
        n = self.nbDraws[arm]
        if n == 0:
            if bound_type == 'max':
                return 1
            else:
                return 0
        if type == 'PR3':
            alpha, beta = self.get_pr3_params(arm)
        else:
            alpha, beta = self.get_pr2_params(arm)
        
        if bound_type == 'max':
            result = stats.beta.ppf([0.975], alpha + 1, beta + 1)[0]
        else:
            result = stats.beta.ppf([0.025], alpha + 1, beta + 1)[0]
        return result
        
    def getConfidence(self, sims, eval_method='PR2'):
        # Get confidence by sampling Beta distribution
        if eval_method == 'sampling_beta':
            return self.getConfidence_by_sampling(sims, 'PR2')
        
        # Get confidence by sampling PR3 distribution

        if eval_method == 'sampling_beta_PR3':
            return self.getConfidence_by_sampling(sims, 'PR3')
        
        # Get sampling by confidence bounds (Beta or Bernstain)

        def safe_div(a,b):
            if b == 0 and a == 0:
                return 0
            if b == 0:
                return 1
            return float(a)/float(b)
        
        if eval_method == 'empirical_bernstain':
            get_bound = self.bernstain_bound

        elif eval_method in ['PR2', 'PR3']:
            get_bound = self.pr_bound
            
        else:
            return 1

        best_arm = np.argmax([safe_div(self.cumReward[arm], self.nbDraws[arm]) for arm in range(self.nbArms)])
        #print("best arm:" + str(best_arm))
        best_sample_mean = np.max([safe_div(self.cumReward[arm], self.nbDraws[arm]) for arm in range(self.nbArms)])
        #print("best_sample_mean:" + str([safe_div(self.cumReward[arm], self.nbDraws[arm]) for arm in range(self.nbArms)]))
        ucb_min = get_bound(best_arm, 'min', eval_method)
        #print("ucb_min:" + str(ucb_min))
        ucb_max = np.max([get_bound(arm, 'max', eval_method) for arm in range(self.nbArms) if arm != best_arm])

        #print("ucb_max:" + str([self.bernstain_bound(arm, 'max') for arm in range(self.nbArms) if arm != best_arm]))
        #print("confidence :" + str(safe_div((ucb_max - ucb_min), best_sample_mean)))
        #print(safe_div((ucb_max - ucb_min), best_sample_mean))
        return safe_div((ucb_max - ucb_min), best_sample_mean)
        
    def getConfidence_by_sampling(self, sims, eval_method='PR2'):
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
        #print("confidence: " + str(best_confidence) + "; remaining: " + str(np.percentile(remaining_values, 95)))
        # return best_confidence, np.percentile(remaining_values, 95), means, self.nbDraws.values()
        return np.percentile(remaining_values, 95)
