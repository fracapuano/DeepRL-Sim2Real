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
As per the guidelines, this project implements some basic RL Algorithms to learn an efficient policy that makes the HOPPER able to jump forward, along the x axis, maximazing its horizontal velocity. The way the code is organized is very simple: a preliminary inspection of the main algorithms employed is done via the **interface.py** file, which makes use of the content of the following folders: agent/ commons/ env/ /models and /policies. Step3 and Step4 of the task are implemented in domain_randomization/ and adaptive_dr/ respectively, in a way that their content can be executed without relying  on other files. In the following, each script will be explained individually.

<!-- GETTING STARTED -->
## Getting Started
The algorithms we implemented are REINFORCE (with three different reward-systems implemented, i.e. baseline, reward to-go, and standard REINFORCE), ActorCritic, PPO and TRPO. REINFORCE and ActorCritic are implemented from scratch, and their content can be found in agents/ and policies/; since we tried to keep the structure simple and intuitive, in agents/ there are the main classes that perform the update operation of a given policy, while in policies/ there are two classes: fraNET.py and tibNET.py. For the purpose of solving the task as indicated in the guidelines, tibNET is the only policy approximation network employed in the whole project, while fraNET served just for debugging purposes, therefore its content shall not evaluated. One can interact with a default instantiation of these algorithms via **interface.py** through some simple commandline arguments:

<ol>
  <li>
--op: "train"/"test". default="train". The former tells the script to instantiate a specific agent, train it with a specific number of episodes/timesteps and
save the resulting model in models/ as a .mdl file; the latter tests a given model (selected in models/ through --model) and renders the results.
  </li>
  <li>
  --model:
  </li>
  <li>
  --device:
  </li>
  <li>
  --render:
  </li>
  <li>
  --episodes:
  </li>
  <li>
  --agent-type:
  </li>
  <li>
  --domain-type:
  </li>
  <li>
  --batch-size:
  </li>
  <li>
  --print-every:
  </li>
</ol>

Just for clarity purposes, here follows an example on how to _train_ a _REINFORCE_ agent for 50000 _episodes_ in the _source domain_ and print its _total reward_ every 100 episodes.
```sh
python interface.py --op train --agent-type reinforce --episodes 50000 --domain-type source --print-every 100
```
And _testing_ it in the _target domain_ for _50 episodes_:
```sh
python interace.py --op test --agent-type reinforce --episodes 50 --domain-type target --render --model reinforce-model.mdl
```

<!-- UDR -->
## UDR

<!-- ADR -->
## ADR

<!-- TESTS -->
## TESTS

<!-- VARIANT -->
## VARIANT

<!-- CONTACTS -->
## CONTACTS