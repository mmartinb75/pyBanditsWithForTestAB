# -*- coding: utf-8 -*-
'''A utility class for evaluating the performance of a policy in multi-armed bandit problems.'''

__author__ = "Olivier Cappé, Aurélien Garivier"
__version__ = "$Revision: 1.10 $"


import numpy as np
from random import seed


class Evaluation:
  
    def __init__(self, env, pol, nb_repetitions, max_decisions, t_sav=[], r_seed=1):
        seed(int(r_seed))
        np.random.seed(int(r_seed))
        print("seed: " + str(r_seed))
        print("t_sav " + str(t_sav))
        print("max_dec:" + str(max_decisions))
        if len(t_sav) > 0:
            self.t_sav = t_sav
        else:
            self.t_sav = np.arange(max_decisions)

        print("self.tsav " + str(self.t_sav))

        self.env = env
        self.pol = pol

        self.env.restart()

        self.nbRepetitions = nb_repetitions
        self.nbArms = env.nbArms
        self.nbPulls = {'all': np.zeros((self.nbRepetitions, self.nbArms)),
                      'PR2': np.zeros((self.nbRepetitions, self.nbArms)),
                      'PR3': np.zeros((self.nbRepetitions, self.nbArms)),
                      'sampling_beta': np.zeros((self.nbRepetitions, self.nbArms)),
                      'sampling_beta_PR3': np.zeros((self.nbRepetitions, self.nbArms)),
                      'empirical_bernstain': np.zeros((self.nbRepetitions, self.nbArms))
                      }
        self.cumReward = np.zeros((self.nbRepetitions, len(self.t_sav)))
        self.cumRegret = np.zeros((self.nbRepetitions, len(self.t_sav)))
        self.choices = np.zeros((self.nbRepetitions, max_decisions))
        self.rewards = np.zeros((self.nbRepetitions, max_decisions))
        self.means = {'all': np.zeros((self.nbRepetitions, self.nbArms)),
                      'PR2': np.zeros((self.nbRepetitions, self.nbArms)),
                      'PR3': np.zeros((self.nbRepetitions, self.nbArms)),
                      'sampling_beta': np.zeros((self.nbRepetitions, self.nbArms)),
                      'sampling_beta_PR3': np.zeros((self.nbRepetitions, self.nbArms)),
                      'empirical_bernstain': np.zeros((self.nbRepetitions, self.nbArms))
                      }
                      
        self.exit_points = {'all': np.zeros(self.nbRepetitions, dtype=np.uint64), 
                            'PR2':np.zeros(self.nbRepetitions, dtype=np.uint64), 
                            'PR3': np.zeros(self.nbRepetitions, dtype=np.uint64),
                            'sampling_beta': np.zeros(self.nbRepetitions, dtype=np.uint64),
                            'sampling_beta_PR3': np.zeros(self.nbRepetitions, dtype=np.uint64),
                            'empirical_bernstain': np.zeros(self.nbRepetitions, dtype=np.uint64)
                            }
                 
        # progress = ProgressBar()
        min_events = float('+inf')
        for k in range(self.nbRepetitions): # progress(range(nbRepetitions)):
            if self.nbRepetitions < 10 or k % (self.nbRepetitions/10) == 0:
                print k
            result = env.play(pol, max_decisions)

            pulls_dict = result.get_nb_pulls()
            self.nbPulls['all'][k, :] = pulls_dict.get('all', 0)
            self.nbPulls['PR2'][k, :] = pulls_dict.get('PR2', 0)
            self.nbPulls['PR3'][k, :] = pulls_dict.get('PR3', 0)
            self.nbPulls['sampling_beta'][k, :] = pulls_dict.get('sampling_beta', 0)
            self.nbPulls['sampling_beta_PR3'][k, :] = pulls_dict.get('sampling_beta_PR3', 0)
            self.nbPulls['empirical_bernstain'][k, :] = pulls_dict.get('empirical_bernstain', 0)

            self.cumReward[k, :] = np.cumsum(result.rewards)[self.t_sav]
            self.choices[k, :] = result.choices
            self.rewards[k, :] = result.rewards

            #cum_regret using max expectation and expected num pulls. Use it instead of regret()
            choices_matrix = np.zeros([len(result.choices), self.nbArms], dtype=np.uint16)
            choices_matrix[np.arange(len(result.choices)), result.choices.astype(int)] = 1
            expectations = [arm.expectation for arm in self.env.arms]
            max_expectation = np.max(expectations)
            regrets = np.matmul(choices_matrix, max_expectation - expectations)
            self.cumRegret[k, :] = np.cumsum(regrets)[self.t_sav]



            self.exit_points['all'][k] = result.nbEvents_method.get('all', 0)
            self.exit_points['PR2'][k] = result.nbEvents_method.get('PR2', 0)
            self.exit_points['PR3'][k] = result.nbEvents_method.get('PR3', 0)
            self.exit_points['sampling_beta'][k] = result.nbEvents_method.get('sampling_beta', 0)
            self.exit_points['sampling_beta_PR3'][k] = result.nbEvents_method.get('sampling_beta_PR3', 0)
            self.exit_points['empirical_bernstain'][k] = result.nbEvents_method.get('empirical_bernstain', 0)
            print("index: " + str(self.exit_points['all']))
            print("index_pr2: " + str(self.exit_points['PR2']))
            print("index_pr3: " + str(self.exit_points['PR3']))
            print("index_sampling_beta: " + str(self.exit_points['sampling_beta']))
            print("index_sampling_beta_PR3: " + str(self.exit_points['sampling_beta_PR3']))
            print("index_empirical_bernstain: " + str(self.exit_points['empirical_bernstain']))


            means_dict = result.get_means()
            self.means['all'][k] = means_dict.get('all', 0)
            self.means['PR2'][k] = means_dict.get('PR2', 0)
            self.means['PR3'][k] = means_dict.get('PR3', 0)
            self.means['sampling_beta'][k] = means_dict.get('sampling_beta', 0)
            self.means['sampling_beta_PR3'][k] = means_dict.get('sampling_beta_PR3', 0)
            self.means['empirical_bernstain'][k] = means_dict.get('empirical_bernstain', 0)

            if self.exit_points['all'][k] < min_events:
                min_events = int(self.exit_points['all'][k])
            
        self.t_sav = np.arange(min_events)
        # progress.finish()
     
    def mean_reward(self):
        return sum(self.cumReward[:, -1])/len(self.cumReward[:, -1])

    def mean_nb_draws(self):
        return np.mean(self.nbPulls['all'], 0)

    def mean_regret(self):
        # return (1+self.tsav)*np.mean(self.bestExpect) - np.mean(self.cumReward, 0)
        return (1 + self.t_sav) * max([arm.expectation for arm in self.env.arms]) - np.mean(self.cumReward, 0)[self.t_sav]

    # Old regret. Based in sample rewards. deprecated.
    def regret(self):
        print("and now tsav " + str(self.t_sav))
        print(((1 + self.t_sav) * max([arm.expectation for arm in self.env.arms]))[-1])
        return (1 + self.t_sav) * max([arm.expectation for arm in self.env.arms]) - self.cumReward[:, self.t_sav]


    def get_rewards(self, exit_method='all'):
        result = np.zeros(self.nbRepetitions)
        max_index = self.exit_points[exit_method]

        for idx, val in enumerate(max_index):
            result[idx] = self.cumReward[idx, val]

        return result

    def get_regrets(self, exit_method='all'):
        result = np.zeros(self.nbRepetitions)

        max_index = self.exit_points[exit_method]

        for idx, val in enumerate(max_index):
            result[idx] = self.cumRegret[idx, val]

        return result





