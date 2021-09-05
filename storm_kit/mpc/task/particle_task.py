import torch 
import yaml
import copy
import numpy as np
import torch.nn as nn

from ...mpc.task import BaseTask
from ...mpc.control import MPPI
from ...mpc.rollout.swiper_rollout import SwiperRollout
from ...mpc.utils.mpc_process_wrapper import ControlProcess
from ...util_file import join_path
from ...util_file import get_mpc_configs_path as mpc_configs_path
from ..utils.torch_utils import find_first_idx, find_last_idx

class ParticleTask(BaseTask):
    def __init__(self, exp_params, env=None, tensor_args={'device':"cpu", 'dtype':torch.float32}):
        super().__init__(tensor_args=tensor_args)
        self.env = env
        self.controller = self.init_mppi(exp_params)
        self.control_process = ControlProcess(self.controller)

    def get_rollout_fn(self, **kwargs):
        rollout_fn = SwiperRollout(**kwargs)
        return rollout_fn

    def update_params(self, **kwargs):
        self.controller.rollout_fn.update_params(**kwargs)
        return True

    def get_command(self, t_step, curr_state, control_dt=0.0, WAIT=False):
        # import ipdb; ipdb.set_trace()
        next_command, val, info, best_action = self.control_process.get_command_debug(t_step, curr_state, integrate_act=False, control_dt=control_dt)

        return next_command, self.controller.trajectories['actions'], self.controller.trajectories['costs']

    def init_mppi(self, exp_params):

        rollout_fn = self.get_rollout_fn(
            exp_params=exp_params, 
            env=self.env,
            tensor_args=self.tensor_args
        )

        mppi_params = exp_params['mppi']
        dynamics_model = rollout_fn.dynamics_model
        mppi_params['d_action'] = dynamics_model.d_action
        mppi_params['action_lows'] = -exp_params['model']['max_action'] * torch.ones(dynamics_model.d_action, **self.tensor_args)
        mppi_params['action_highs'] = exp_params['model']['max_action'] * torch.ones(dynamics_model.d_action, **self.tensor_args)
        init_action = torch.ones((mppi_params['horizon'], dynamics_model.d_action), **self.tensor_args)
        init_action[:,0] *= 0.
        init_action[:,1] *= 0.
        mppi_params['init_mean'] = init_action
        mppi_params['rollout_fn'] = rollout_fn
        mppi_params['tensor_args'] = self.tensor_args
        controller = MPPI(**mppi_params)
        self.exp_params = exp_params
        return controller