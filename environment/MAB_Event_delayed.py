# -*- coding: utf-8 -*-
'''General simulation routine based in time events'''

__author__ = "Miguel Martin, Antonio Jimenez, Alfonso Mateos"
__version__ = "1.0"

import Result
from Environment import Environment
from Queue import PriorityQueue
import random as rn
import time
from sets import Set
import numpy as np


class MABEventDelayed(Environment):
    """Multi-armed bandit problem with arms given in the 'arms' list"""
    
    def __init__(self, arms, conf, rnd_process, exit_mode=0, tunning_threshold=0.05, tunning_sampling_iterations=50):
        self.rndProcess = rnd_process
        self.arms = arms
        self.nbArms = len(arms)
        self.conf = conf
        self.last_arm_index = 0
        self.events = PriorityQueue()
        self.policy = None
        self.black_box = False
        self.delay_decisions = dict()
        self.all_delays = []
        self.choice_delays = []
        self.update_delays = []
        self.confidence_check_max = 50000
        self.confidence_simulations = 10000
        self.sampling_iterations_PR2 = tunning_sampling_iterations
        self.sampling_iterations_PR3 = tunning_sampling_iterations
        self.max_remainder_PR2 = 0
        self.max_remainder_PR3 = 0
        self.confidence_iterations = 10
        self.confidence = False
        self.exit_mode = exit_mode
        self.tunning_threshold = tunning_threshold
        self.tunning_sampling_iterations = tunning_sampling_iterations
        if self.exit_mode == 0:
            self.confidence_methods = {'PR2': False, 'sampling_beta': False, 'empirical_bernstain': False}
        else:
            self.confidence_methods = {'PR2': False, 'PR3': False, 'sampling_beta': False, 'sampling_beta_PR3': False,'empirical_bernstain': False}
        
    def arm_event(self, t, sim_index, result):
        init_time = time.time()
        choice = self.policy.choice()

        elapsed_time = (time.time() - init_time)
        self.choice_delays += [elapsed_time]
        t_rew = self.arms[choice].draw()
        reward, r = self.conf.getDelayedReward(t_rew, t)
        delay = r + t

        # print "reward: " + str(reward)
        # print "delay reward: " + str(delay)

        self.events.put((delay, 'r', reward, choice, t, sim_index))

        t_next = self.rndProcess.draw(t/3600)

        if self.not_confidence(sim_index, result):
            self.events.put((t + t_next, 'a', 0, 0, t, sim_index))
            # for (key, value) in self.delay_decisions.items():
            #     self.delay_decisions[key] = value + 1

            # self.delay_decisions[sim_index] = 0
        else:
            print("sim_index: " + str(sim_index))

    def play(self, reference_policy, max_decisions):

        self.policy = reference_policy

        self.policy.startGame()

        self.confidence = False
        if self.exit_mode == 0:
            self.confidence_methods = {'PR2': False, 'sampling_beta': False, 'empirical_bernstain': False}
        else:
            self.confidence_methods = {'PR2': False, 'PR3': False, 'sampling_beta': False, 'sampling_beta_PR3': False, 'empirical_bernstain': False}


        self.sampling_iterations_PR2 = self.tunning_sampling_iterations
        self.sampling_iterations_PR3 = self.tunning_sampling_iterations
        self.max_remainder_PR2 = 0
        self.max_remainder_PR3 = 0
        print(str(self.sampling_iterations_PR2) + ' y ' + str(self.sampling_iterations_PR3))

        result = Result.Result(self.nbArms, max_decisions, self.exit_mode)

        sim_index = 0
        random_init_time = rn.uniform(0, 24)

        first_event = self.rndProcess.draw(random_init_time)
        self.arm_event(first_event, sim_index, result)

        while not self.events.empty():
            event_time, event_type, reward, arm, t, idx = self.events.get()
            if event_type == 'r':
                init_time = time.time()
                self.policy.getReward(int(arm), reward)
                update_time = (time.time() - init_time)
                self.update_delays +=[update_time]
                if idx < max_decisions:
                    result.store(idx, int(arm), reward)

                    # delay = self.delay_decisions.pop(idx, 'None')
                    # if delay is not 'None':
                    #     self.all_delays += [delay]

            else:
                sim_index += 1
                self.arm_event(event_time, sim_index, result)
                
        return result

    def restart(self):

        # for i in self.arms:
        #     i.restart()
        # self.conf.restart()

        self.events = PriorityQueue()
        self.confidence = False
        if self.exit_mode == 0:
            self.confidence_methods = {'PR2': False, 'sampling_beta': False, 'empirical_bernstain': False}
        else:
            self.confidence_methods = {'PR2': False, 'PR3': False, 'sampling_beta': False, 'sampling_beta_PR3': False, 'empirical_bernstain': False}
        
        self.max_remainder_PR2 = 0
        self.max_remainder_PR3 = 0
        self.sampling_iterations_PR2 = self.tunning_sampling_iterations
        self.sampling_iterations_PR3 = self.tunning_sampling_iterations

     
    def not_confidence(self, sim_index, result):
        if (sim_index % self.confidence_iterations) != 0:
            return not self.confidence
        
                        
        def update_confidence(method='PR2'):
            self.remainder_value = self.policy.getConfidence(self.confidence_simulations, method)

            if self.remainder_value < 0.01 and not self.confidence_methods[method]:
                self.confidence_methods[method] = True
                result.nbEvents_method[method] = sim_index
                print("method " + method +  "| rem value :" + str(self.remainder_value) + " sims: " + str(sim_index))
                if method in  ['PR2', 'PR3']:
                    b = self.policy.getConfidence_by_sampling(10000, method)
                    print("b: " + str(b))
                
        # get confidence using pr2 aproximation:
        if self.confidence_methods['PR2'] is not True:
            update_confidence('PR2')
        else:
            self.remainder_value = self.policy.getConfidence(self.confidence_simulations, 'PR2')

        # get confidence by beta sampling PR2:

        if self.remainder_value < self.tunning_threshold and(sim_index % self.sampling_iterations_PR2) == 0 and self.confidence_methods['sampling_beta'] is not True:
            update_confidence('sampling_beta')
            if self.max_remainder_PR2  == 0:
                self.max_remainder_PR2 = self.remainder_value - 0.01

            self.sampling_iterations_PR2 = min(self.tunning_sampling_iterations, max(50, int(50 + self.tunning_sampling_iterations*(self.remainder_value - 0.01)/self.max_remainder_PR2)))
            print("sampling iterations pr2: " + str(self.sampling_iterations_PR2))
            print("remainder pr2: " + str(self.remainder_value))
        
        # get confidence using pr3 aproximation
        if self.exit_mode != 0 and self.confidence_methods['PR3'] is not True:
            update_confidence('PR3')

        # get confidence by beta sampling PR3:
        if self.exit_mode != 0 and self.remainder_value < self.tunning_threshold and (sim_index % self.sampling_iterations_PR3) == 0 and self.confidence_methods['sampling_beta_PR3'] is not True:
            update_confidence('sampling_beta_PR3')

            if self.max_remainder_PR3  == 0:
                self.max_remainder_PR3 = self.remainder_value - 0.01
                print("max remainder pr3: " + str(self.max_remainder_PR3))

            self.sampling_iterations_PR3 = min(self.tunning_sampling_iterations, max(50, int(50 + self.tunning_sampling_iterations*(self.remainder_value - 0.01)/self.max_remainder_PR3)))
            print("sampling iterations pr3: " + str(self.sampling_iterations_PR3))
            print("remainder pr3: " + str(self.remainder_value))
        
        # get confidence using empirical bernstain:
        if self.confidence_methods['empirical_bernstain'] is not True:
            update_confidence('empirical_bernstain')
              
        self.confidence = all(self.confidence_methods.values())

        return not self.confidence
