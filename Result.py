# -*- coding: utf-8 -*-
'''Utility class for handling the results of a Multi-armed Bandits experiment.'''

__author__ = "Olivier Cappé, Aurélien Garivier"
__version__ = "$Revision: 1.7 $"


import numpy as np


class Result:
    """The Result class for analyzing the output of bandit experiments."""
    def __init__(self, nb_arms, horizon, exit_mode=0):
        self.nbArms = nb_arms
        self.choices = np.zeros(horizon)
        self.rewards = np.zeros(horizon)
        self.vars = np.zeros(horizon)
        self.bads = np.zeros(horizon)
        self.nbDraws = dict()
        self.worst = np.zeros(horizon)
        if exit_mode == 0:
            self.nbEvents_method = {'all': 0, 'PR2': 0, 'sampling_beta': 0, 'empirical_bernstain': 0}
        else:
            self.nbEvents_method = {'all': 0, 'PR2': 0, 'PR3': 0, 'sampling_beta': 0, 'sampling_beta_PR3': 0, 'empirical_bernstain': 0}

        for arm in range(self.nbArms):
            self.nbDraws[arm] = 0

    def store(self, t, choice, reward):
        self.choices[t] = choice
        self.rewards[t] = reward
        self.nbDraws[choice] += 1
        self.nbEvents_method['all'] +=1

    def get_nb_pulls(self):
        nb_pulls_dict = {}
        if self.nbArms == float('inf'):
            for k,v in self.nbEvents_method.items():
                nb_pulls_dict[k] = np.array([])
        else:  
            for k,v in self.nbEvents_method.items():
                nb_pulls = np.zeros(self.nbArms)
                for choice in self.choices[:int(v)]:
                    nb_pulls[int(choice)] += 1
                nb_pulls_dict[k] = nb_pulls

        return nb_pulls_dict
    
    def get_means(self):
        means_dict = {}
        for k,v in self.nbEvents_method.items():
            arms_exp = np.zeros(self.nbArms)
            arms_times = np.zeros(self.nbArms)
            for reward, choice in zip(self.rewards[:v],self.choices[:v]):
                arms_exp[int(choice)] += reward
                arms_times[int(choice)] += 1
            def not_null_div(a, b):
                if a == 0 and b == 0:
                    return 0
                else:
                    return (a/float(b))
            
            means_dict[k] = [not_null_div(arms_exp[i], arms_times[i]) for i in range(self.nbArms)]

        return means_dict

    def get_regret(self, best_expectation):
        return np.cumsum(best_expectation-self.rewards)
