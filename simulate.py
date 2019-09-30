# -*- coding: utf-8 -*-
'''Simulation of batch reward updated architectures'''

__author__ = "Miguel Martin, Antonio Jimenez, Alfonso Mateos"
__version__ = "1.0"


from environment.MAB_Event_delayed import MABEventDelayed
from environment.Not_Delayed_conf import Not_Delayed_conf
from environment.Delayed_Conf_batch_poisson import DelayedConfBatchPoisson
from environment.Delayed_dependent_reward_Conf_Random_poisson import Delayed_DR_Conf_Random_poisson
from environment.Delayed_dependent_reward_Conf_batch_poisson import Delayed_DR_ConfBatchPoisson
from environment.Delayed_Conf_Random_poisson import Delayed_Conf_Random_poisson
from arm.Bernoulli import Bernoulli
from arm.Bernoulli_Delay import Bernoulli_Delay
from policy.UCB import UCB
from policy.TestAB import TestAB
from policy.Egreedy import Egreedy
from numpy import *
from policy.DMED import DMED
from policy.klUCB import klUCB
from policy.klUCBplus import klUCBplus
from policy.PossibilisticReward import PossibilisticReward
from policy.PossibilisticReward_chernoff2 import PossibilisticReward_chernoff2
from policy.PossibilisticReward_Bern1 import PossibilisticReward_Bern1
from policy.ProbabilityMatching import ProbabilityMatching
from policy.besa import besa
from Evaluation import *
from kullback import *
from policy.BlackBox import BlackBox
from environment.NH_Poisson_Proccess import NHPoissonProcess
from environment.NH_Poisson_Proccess import generate_ratio_function
import sys
import json
import time


nbRep = 10
max_decisions = 5000000

params_low = [(0, 0.005),
              (8, 0.005),
              (10, 0.006),
              (12, 0.008),
              (13, 0.008),
              (15, 0.006),
              (17, 0.006),
              (19, 0.01),
              (20, 0.01),
              (21, 0.009)]


params_high = [(0, 0.005),
               (8, 0.025),
               (10, 0.075),
               (12, 0.55),
               (13, 0.55),
               (15, 0.2),
               (17, 0.2),
               (19, 0.65),
               (20, 0.65),
               (21, 0.55)]


def get_policy(nbArms, trunc_value, index):
    policies = {'TestAB': TestAB(nbArms),
                'ProbabilityMatching': ProbabilityMatching(nbArms),
                'PR2': PossibilisticReward_chernoff2(nbArms, trunc_value, scale=1),
                'PR3': PossibilisticReward_Bern1(nbArms, trunc_value, confidence=0.1),
                'Egreedy': Egreedy(nbArms)
                }

    if index == 'all':
        return policies.values
    else:
        return [policies[index]]

def get_config_reward(objective, policy, w_time=0):
    if objective == 'navigation_time' and policy in ['ProbabilityMatching', 'Egreedy']:
        config = Delayed_DR_ConfBatchPoisson(1./150, windows_time=w_time,  trunc=480)
    elif objective == 'navigation_time':
        config = Delayed_DR_Conf_Random_poisson(1./150, 480)
    elif objective == 'conversion_rate' and policy in ['ProbabilityMatching', 'Egreedy']:
        config = DelayedConfBatchPoisson(windows_time_hours=w_time)
    else:
        config = Delayed_Conf_Random_poisson(1./150, 480)

    return config

def get_arms_distributions(scenario, objective):
    if scenario == 'two_arms' and objective == 'conversion_rate':
        arms = [Bernoulli(p) for p in [0.05, 0.04]]
        
    elif scenario == 'many_arms' and objective == 'conversion_rate':
        arms = [Bernoulli(p) for p in [0.05, 0.045, 0.04, 0.04, 0.035, 0.035, 0.03, 0.03, 0.02, 0.02]]

    elif scenario == 'two_arms' and objective == 'navigation_time':
        arms = [Bernoulli_Delay(p, 1./150) for p in [0.5, 0.45]]
    else:
        arms = [Bernoulli_Delay(p, 1./150) for p in [0.5, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45]]
    
    return arms

def execute_simulation(nbRep, horizon, pol_index, traffic, scenario, objective, base_dir='data', windows_size=0, ex_seed=1):

    # Set traffic volume
    if traffic == 'high_traffic':
        params = params_high
    else:
         params = params_low

    f = generate_ratio_function(params)
    process = NHPoissonProcess(f)

    # config = Not_Delayed_conf()
    #config = Delayed_DR_Conf_Random_poisson(1./150, 480)
    #config = Delayed_DR_ConfBatchPoisson(1./150, windows_time=2,  trunc=480)
    # env = MABEventDelayed(
    #     [Bernoulli(p, samples=nbRep * horizon) for p in [0.1, 0.05]],
    #     config, process, exit_mode=0)

    config = get_config_reward(objective, pol_index, windows_size)
    arms =  get_arms_distributions(scenario, objective)

    trunc_value = 1
    if objective == 'conversion_rate':
        exit_mode = 0
    else:
        exit_mode = 1

    if scenario == 'two_arms':
        tunning_threshold = 1
        tunning_sim_iterations = 50
        print("tunning: " + str(tunning_threshold))
    else:
        tunning_threshold = 0.05
        tunning_sim_iterations = 4500
    
    env = MABEventDelayed(arms, config, process, exit_mode, tunning_threshold, tunning_sim_iterations)
    policies = get_policy(env.nbArms, trunc_value, pol_index)
    print 'policiy: ' + str(policies)

    # tsav = int_(linspace(100, horizon - 1, 200))
    print(config)
    print(arms)
    print("exit_mode: " + str(exit_mode))
    print("tunning_threshold: " + str(tunning_threshold))
    print(env)
    print(policies)
    for wrap_pol in policies:
        ev = Evaluation(env, wrap_pol, nbRep, max_decisions, r_seed=ex_seed)
        print ev.mean_reward()
        print ev.mean_nb_draws()
        prefix = ''

        means = ev.means
        regret = ev.regret()
        rewards = ev.get_rewards()

        if wrap_pol.__class__.__name__ == 'BlackBox':
            policy = wrap_pol.policy
            prefix = 'BlackBox_'
        else:
            policy = wrap_pol

        name_klucb = str(getattr(policy, "klucb", "none"))
        name_scale = str(getattr(policy, "scale", "none"))
        name_gap = str(getattr(policy, "gap", "none"))
        name_conf = str(getattr(policy, "confidence", "none"))
        base_name = base_dir + "/" + prefix + policy.__class__.__name__ + "_" + str(
            nbRep) \
                    + "-" + str(objective) + "_" + str(scenario) + "_wsize" + str(windows_size) + "_seed_" + str(ex_seed)

        
        tr = traffic
        # PR2:
        save(base_name + "_" + tr + "_" + "last_means_pr2", means['PR2'])
        save(base_name + "_" + tr + "_" + "last_regrets_pr2", ev.get_regrets('PR2'))
        save(base_name + "_" + tr + "_" + "last_rewards_pr2", ev.get_rewards('PR2'))
        save(base_name + "_" + tr + "_" + "num_events_pr2", ev.exit_points['PR2'])
        save(base_name + "_" + tr + "_" + "nbPulls_pr2", ev.nbPulls['PR2'])
        
        # All:

        save(base_name + "_" + tr + "_" + "last_means", means['all'])
        save(base_name + "_" + tr + "_" + "last_regrets", ev.get_regrets('all'))
        save(base_name + "_" + tr + "_" + "last_rewards", ev.get_rewards('all'))
        save(base_name + "_" + tr + "_" + "num_events", ev.exit_points['all'])
        save(base_name + "_" + tr + "_" + "nbPulls", ev.nbPulls['all'])

        # PR3:

        save(base_name + "_" + tr + "_" + "last_means_pr3", means['PR3'])
        save(base_name + "_" + tr + "_" + "last_regrets_pr3", ev.get_regrets('PR3'))
        save(base_name + "_" + tr + "_" + "last_rewards_pr3", ev.get_rewards('PR3'))
        save(base_name + "_" + tr + "_" + "num_events_pr3", ev.exit_points['PR3'])
        save(base_name + "_" + tr + "_" + "nbPulls_pr3", ev.nbPulls['PR3'])

        # sampling_beta:

        save(base_name + "_" + tr + "_" + "last_means_sampling_beta", means['sampling_beta'])
        save(base_name + "_" + tr + "_" + "last_regrets_sampling_beta", ev.get_regrets('sampling_beta'))
        save(base_name + "_" + tr + "_" + "last_rewards_sampling_beta", ev.get_rewards('sampling_beta'))
        save(base_name + "_" + tr + "_" + "num_events_sampling_beta", ev.exit_points['sampling_beta'])
        save(base_name + "_" + tr + "_" + "nbPulls_sampling_beta", ev.nbPulls['sampling_beta'])



        # sampling_beta_PR3:

        save(base_name + "_" + tr + "_" + "last_means_sampling_beta_PR3", means['sampling_beta_PR3'])
        save(base_name + "_" + tr + "_" + "last_regrets_sampling_beta_PR3", ev.get_regrets('sampling_beta_PR3'))
        save(base_name + "_" + tr + "_" + "last_rewards_sampling_beta_PR3", ev.get_rewards('sampling_beta_PR3'))
        save(base_name + "_" + tr + "_" + "num_events_sampling_beta_PR3", ev.exit_points['sampling_beta_PR3'])
        save(base_name + "_" + tr + "_" + "nbPulls_sampling_beta_PR3", ev.nbPulls['sampling_beta_PR3'])


        # empirical_bernstain:

        save(base_name + "_" + tr + "_" + "last_means_empirical_bernstain", means['empirical_bernstain'])
        save(base_name + "_" + tr + "_" + "last_regrets_empirical_bernstain", ev.get_regrets('empirical_bernstain'))
        save(base_name + "_" + tr + "_" + "last_rewards_empirical_bernstain", ev.get_rewards('empirical_bernstain'))
        save(base_name + "_" + tr + "_" + "num_events_empirical_bernstain", ev.exit_points['empirical_bernstain'])
        save(base_name + "_" + tr + "_" + "nbPulls_empirical_bernstain", ev.nbPulls['empirical_bernstain'])

        #regret:
        save(base_name + "_" + tr + "_" + "cum_regret_2", ev.cumRegret)


if __name__ == "__main__":
    init_time = time.time()

    config_file = str(sys.argv[1])

    with open(config_file) as f:
        config = json.load(f)
    
    nbRep = int(config['num_reps'])
    max_decision = int(config['max_horizon'])
    policy = config['policy']
    traffic = config['traffic']
    scenario = config['scenario']
    objective = config['objective']
    if 'seed' in config:
        sim_seed = float(config['seed'])
    else:
        sim_seed = 1


    if config['policy'] in ['ProbabilityMatching', 'Egreedy']:
        windows = int(config['windows_size'])
        execute_simulation(nbRep, max_decision, policy, traffic, scenario, objective, 'data', windows, ex_seed=sim_seed)
    else:
        execute_simulation(nbRep, max_decision, policy, traffic, scenario, objective, 'data', ex_seed=sim_seed)
    
    print("Duration: " + str(time.time() - init_time) + " secs")
    exit(0)
