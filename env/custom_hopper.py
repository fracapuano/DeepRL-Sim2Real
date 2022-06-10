"""Implementation of the Hopper environment supporting
domain randomization optimization."""

from copy import deepcopy
from multiprocessing.sharedctypes import Value

import numpy as np
import gym
from gym import utils
from .mujoco_env import MujocoEnv
import torch
from torch.distributions.uniform import Uniform
from torch.nn.init import trunc_normal_ as tn

class CustomHopper(MujocoEnv, utils.EzPickle):
    def __init__(self, domain=None):
        MujocoEnv.__init__(self, 4)
        utils.EzPickle.__init__(self)

        self.original_masses = np.copy(self.sim.model.body_mass[1:])    # Default link masses

        if domain == 'source':  # Source environment has an imprecise torso mass (1kg shift) with respect to the 
            self.sim.model.body_mass[1] -= 1.0
        
        self.random_masses = 3

    def set_uniform_parametrization(self, bounds):
        self.low1 = bounds[0]
        self.high1 = bounds[1]
        self.low2 = bounds[2]
        self.high2 = bounds[3]
        self.low3 = bounds[4]
        self.high3 = bounds[5]

    def set_normal_parametrization(self, low=.25, high=5, mean=0, std=1):
        self.low = low
        self.high = high
        self.mean = mean
        self.std = std

    def set_random_parameters(self):
        #Set random masses
        self.set_parameters(*np.hstack((np.array(self.sim.model.body_mass[1]), self.sample_parameters())))

    def sample_parameters(self, dist_type='uniform'):
        """
        Sample masses according to a domain randomization distribution
        -----
        This function samples masses for a Uniform Distribution. Sampling is done independently for each component of the
        vector. 
        """
        if dist_type=='uniform':
            self.MassDistribution1 = Uniform(low = torch.tensor([self.low1], dtype = float),
                                            high = torch.tensor([self.high1], dtype = float), 
                                            validate_args = False)
            self.MassDistribution2 = Uniform(low = torch.tensor([self.low2], dtype = float),
                                            high = torch.tensor([self.high2], dtype = float), 
                                            validate_args = False)
            self.MassDistribution3 = Uniform(low = torch.tensor([self.low3], dtype = float),
                                            high = torch.tensor([self.high3], dtype = float), 
                                            validate_args = False)

            randomized_masses = np.array([
            self.MassDistribution1.sample().detach().numpy(), 
            self.MassDistribution2.sample().detach().numpy(),
            self.MassDistribution3.sample().detach().numpy()
            ])

        elif dist_type=='normal':
            empty_tensor = torch.empty(1,1)
            self.MassDistribution1 = tn(empty_tensor, a=self.low, b=self.high, mean=self.mean, std=self.std)
            self.MassDistribution2 = tn(empty_tensor, a=self.low, b=self.high, mean=self.mean, std=self.std)
            self.MassDistribution3 = tn(empty_tensor, a=self.low, b=self.high, mean=self.mean, std=self.std)

            randomized_masses = np.array([
            self.MassDistribution1.detach().numpy(), 
            self.MassDistribution2.detach().numpy(),
            self.MassDistribution3.detach().numpy()
            ])

        else:
            raise ValueError("Parametric distribution not supported! Try normal or uniform.")

        return randomized_masses.reshape(-1,)

    def select_parameters_to_sample(self, masses=[], dist_type='uniform'):

        randomized_masses = [self.sim.model.body_mass[2], self.sim.model.body_mass[3], self.sim.model.body_mass[4]]

        masses_map = {
            'thigh':2,
            'leg':3,
            'foot':4
        }

        if not bool(masses):
            for mass in masses:
                if dist_type=='uniform':
                    randomized_masses[masses_map[mass]]= Uniform(
                        low = torch.tensor([self.low], dtype = float),
                        high = torch.tensor([self.high], dtype = float), 
                        validate_args = False
                        ).sample().detach().numpy()

                elif dist_type=='normal':
                    empty_tensor = torch.empty(1,1)
                    randomized_masses[masses_map[mass]] = tn(empty_tensor, a=self.low, b=self.high, mean=self.mean, std=self.std)
                else:
                    raise ValueError("Parametric distribution not supported! Try normal or uniform.")

        return np.array(randomized_masses).reshape(-1,)

    def get_parameters(self):
        """Get value of mass for each link"""
        masses = np.array( self.sim.model.body_mass[1:] )
        return masses

    def set_parameters(self, *task):
        """Set each hopper link's mass to a new value"""
        self.sim.model.body_mass[1:] = task

    def step(self, a):
        """Step the simulation to the next timestep

        Parameters
        ----------
        a : ndarray,
            action to be taken at the current timestep
        """
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and (height > .7) and (abs(ang) < .2))
        ob = self._get_obs()

        return ob, reward, done, {}

    def _get_obs(self):
        """Get current state"""
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat
        ])

    def reset_model(self):
        """Reset the environment to a random initial state"""
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20


    def set_mujoco_state(self, state):
        """Set the simulator to a specific state

        Parameters:
        ----------
        state: ndarray,
               desired state
        """
        mjstate = deepcopy(self.get_mujoco_state())

        mjstate.qpos[0] = 0.
        mjstate.qpos[1:] = state[:5]
        mjstate.qvel[:] = state[5:]

        self.set_sim_state(mjstate)


    def set_sim_state(self, mjstate):
        """Set internal mujoco state"""
        return self.sim.set_state(mjstate)

    def get_mujoco_state(self):
        """Returns current mjstate"""
        return self.sim.get_state()


"""
    Registered environments
"""
gym.envs.register(
        id="CustomHopper-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
)

gym.envs.register(
        id="CustomHopper-source-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
        kwargs={"domain": "source"}
)

gym.envs.register(
        id="CustomHopper-target-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
        kwargs={"domain": "target"}
)

