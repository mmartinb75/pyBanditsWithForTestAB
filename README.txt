        pyBandits simiulation in for Test AB solutions
        It has been developed extending the pyBandits python code of  Olivier Capp�, Aur�lien Garivier, Emilie Kaufmann.
     ............................................................

     python classes of Olivier Capp�, Aur�lien Garivier, Emilie Kaufmann.:

            Evaluation.py		  Class for running a bandit experiment
            kullback.py			  Module with Kullback-Leibler utilities
            Result.py			  Class for summarizing a bandit experiments

            arm/
                arm/__init__.py
                arm/Arm.py		  Generic arm class
                arm/Bernoulli.py		  Class for common arm distributions (possibly
                arm/Exponential.py	    truncated)
                arm/Gaussian.py
                arm/Poisson.py

            C/
                C/kullback.c		  C-coded version of the Kullback-Leibler
                C/Makefile		    utility module (will supersede kullback.py
                C/README.txt		    if installed)
                C/setup.py

            environment/
                environment/__init__.py
                environment/Environment.py  Generic environment class
                environment/MAB.py	  Multi-armed bandit class (note that a MAB is
    				    a collection of arms and can thus use
    				    arms with differents types)

            policy/
                policy/__init__.py
                policy/Policy.py		  Generic policy classes
                policy/IndexPolicy.py
                policy/BayesUCB.py	  Class for policies, names should be explicit
                policy/KLempUCB.py	    (note that klUCB can use the different
                policy/klUCB.py		    forms of KL divergences defined in
                policy/Thompson.py	    kullback, in particular, UCB is a special
                policy/UCB.py                 case of klUCB)
                policy/UCBV.py

            posterior/
                posterior/__init__.py
                posterior/Posterior.py	  Generic class for posteriors
                posterior/Beta.py		  Posteriors in Bernoulli/Beta experiments


     New python classes created by  Miguel Martín, Antonio Jiménez and Alfonso Mateos:


            
            simulate.py
            omparing results.ipynb


            arm/
                arm/Bernoulli_Delay.py


            environment/
                environment/Delayed_dependent_reward_Conf_batch_poisson.py
                environment/Delayed_Conf_batch_poisson.py
                environment/Delayed_Conf_Random_poisson.py
                environment/Delayed_dependent_reward_Conf_Random_poisson.py
                environment/MAB_Event_delayed.py
                environment/NH_Poisson_Proccess.py

            policy/
                policy/besa.py
                policy/BlackBox.py
                policy/PossibilisticReward.py
                policy/PossibilisticReward_Bern1.py
                policy/PossibilisticReward_chernoff2.py
                policy/ProbabilityMatching.py
                policy/TestAB.py

-----------------------------------------------------------
If you want to simulate again any scenario  execute different simulations write on of these:

For batch architecture simulations:

python -u   simulate.py config/scenario_x.json
python -u   generateBatchResults_eventMode_with_seed.py scenarios method nbRep horizon traffic seed

where:
    config/scenario_x.json could be any of the differents json files in config, storing the parameters of each scenario and policy.


The result is stored for each policy and scenario executed in different numpy files in the data directory where 
    cum_regrets - the cumulative regret of the simulation
    num_events_'stop_criteria' - the num of events of each simulation to reach the stop criteria.
    last_rewards_'stop_criteria'- The last reward of each arm using the stop criteria defined
    last_regrets_'stop_criteria'- The last regrets of each arm using the stop criteria defined
    last_means_'stop_criteria'- The empirical means of each arm a the end of the simulation.
    nbPulls_'stop_criteria' - the number of times each arm has been pulled in each simulation.


If you don't want simulated again and just analized the results just copy the simulated data from the UCB repository to the data directory and open  comparing results.ipynb notebook to see and play with  the analysis results.
