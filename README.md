# MLDLRL
<div id="top"></div>

<br />
<div align="center">
  <a href="https://github.com/wibox/MLDLRL">
    <img src="logo/logo.png" alt="Logo" width="300" height="300">
  </a>

  <h3 align="center">Reinforcement Learning for Machine Learning and Deep Learning course <br /> PoliTo 2021/2022</h3>

  <p align="center">
    In the following, the whole project is briefly presented in how it works and how the various scripts should be executed to reproduce our results. We worked with some basic RL algorithms to make our friendly hopper able to jump to infinity and beyond.
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li><a href="#getting-started">Getting Started</a></li>
    <li><a href="#udr">Uniform Domain Randomization</a></li>
    <li><a href="#adr">Adaptive Domain Randomization</a></li>
    <li><a href="#tests">Tests</a></li>
    <li><a href="#variant">Variant</a></li>
    <li><a href="#contact">Contacts</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project
As per the guidelines, this project implements some basic RL Algorithms to learn an efficient policy that makes the HOPPER able to jump forward, along the x axis, maximazing its horizontal velocity. The way the code is organized is very simple: a preliminary inspection of the main algorithms employed is done via the **interface.py** file, which makes use of the content of the following folders: agent/ commons/ env/ /models and /policies. Step3 and Step4 of the task are implemented in domain_randomization/ and adaptive_dr/ respectively, in a way that their content can be executed without relying  on other files. In the following more informations will be provided.

<!-- GETTING STARTED -->
## Getting Started
The algorithms we implemented are REINFORCE (with three different reward-systems implemented, i.e. baseline, reward to-go, and standard REINFORCE), ActorCritic, PPO and TRPO. REINFORCE and ActorCritic are implemented from scratch, and their content can be found in agents/ and policies/; since we tried to keep the structure simple and intuitive, in agents/ there are the main classes that perform the update operation of a given policy, while in policies/ there are two classes: fraNET.py and tibNET.py. For the purpose of solving the task as indicated in the guidelines, tibNET is the only policy approximation network employed in the whole project, while fraNET served just for debugging purposes, therefore its content shall not evaluated. One can interact with a default instantiation of these algorithms via **interface.py** through some simple commandline arguments:


  * --op: "train"/"test". default="train". The former tells the script to instantiate a specific agent, train it with a specific number of episodes/timesteps and save the resulting model in models/ as a .mdl file; the latter tests a given model (selected in models/ through --model) and renders the results.

  * --model: str default=None. Is just the name of the model you want to save after a training procedure or the name of the model you want to load when testing an agent. Be aware that no specific check is done to assert its compatibility with the specified --agent-type, therefore it should be coherent with said choice. Don't use --agent-type actorCritic and load a model for TRPO with --model trpo-model.mdl.

  * --device: "cpu"/"cuda". default="cpu". The whole project is developed in PyTorch [https://pytorch.org/], therefore there is the possibility to perform some operations using the GPU. At this specific state of the project such opportunity is not yet implemented but will for sure in next releases. Just leave it as default for now.

  * --render: bool. default=True. Is a boolean value used to render the environment during training or testing.


  * --episodes: int. default=10. Is the number of episodes/timesteps (for REINFORCE/ActorCritic and PPO/TRPO respectively) to train an agent or test it in a specified environemnt


  * --agent-type: "REINFORCE"/"actorCritic"/"PPO"/"TRPO". default="REINFORCE". Is a string used to select the specific agent to train/test.


  * --domain-type: "source"/"target". default="source". Is a string used to indicate the specific environment to build. Our task is a sim-to-sim job, therefore the source environemnt refers to the one which has a torso mass set to 1kg less than the target environment.


  * --batch-size: int. default=10. Is an integer used to indicate the dimension of the batch chuck used for ActorCritic policy updates.


  * --print-every: int. default=10. Is the number of episodes to skip before printing the episode's total reward during training/testing.



Just for clarity purposes, here follows an example on how to _train_ a _REINFORCE_ agent for 50000 _episodes_ in the _source domain_ and print its _total reward_ every 100 episodes.
```sh
python interface.py --op train --agent-type reinforce --episodes 50000 --domain-type source --print-every 100
```
And _testing_ it in the _target domain_ for _50 episodes_:
```sh
python interface.py --op test --agent-type reinforce --episodes 50 --domain-type target --render --model reinforce-model.mdl
```

<!-- UDR -->
## UDR

<!-- ADR -->
## ADR

<!-- TESTS -->
## TESTS
The folders step2_tests/ step3_tests/ and step4_tests/ perform the hyperparameters' tuning of the agents in different contexts.
* step2_tests/ content optimizes REINFORE, ActorCritic, PPO and TRPO parameters for environments in which no specific randomization techniques are employed.
* step3_tests/ does the same thing but for environments in which Uniform Domain Randomization is applied.
* step4_tests/ performs hyperparameters' tuning for a TRPO agent that makes use of BayRn.py.
The way each folder works is substantially the same: first config_setter.py creates a parameter grid object of the configurations we wanted to test and writes the resulting combinations as a json file which then each indivudal script uses. Please notice that every tests_loader_"model".py has its specific folder in which the combinations are stored and a "model"_evaluation.txt in which each combination's specific performance are recorded. Eventually, config_getter.py retrives the best one for each model and stores it inside best_config.txt. For each folder a run_hpt.sh file is present and its sufficient to run it to perform all the tests in that folder. !! Notice that all the results are already in this repository, so to save your time we strongly recommend not to look at run_hpt.sh, kindly !!
<!-- VARIANT -->
## VARIANT
variant/ contains the content of the alternative approach we tried to implement in order to best identify the domain's parameters to randomize. A better explanation of such procedure can be read in our work's report, therefore here we just specify the order in which variant/*.py files should be executed to read the automatically generated results:

<ol>
<li>dataset_generation.py to create a specific dataset suitable for our analysis.</li>
<li>dynamics_analysis.py in which a Random Forest Regressor performs its magic on the previously built dataset.</li>
<li>
variant_validation.py which trains a TRPO agent in the source environment where just 2 of the 3 masses are randomized according to our analysis results, and then tests it in the target domain.
</li>
</ol>

<!-- CONTACTS -->
## CONTACTS
Francesco Pagano  s299266@studenti.polito.it <br />
Francesco Capuano  s295366@studenti.polito.it <br />
Francesca Mangone  s303489@studenti.polito.it