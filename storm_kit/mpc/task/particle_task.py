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
        self.control_process = None 

        # Use multi processing when not passing in gym environment
        if env == None:
            self.control_process = ControlProcess(self.controller)
        
        self.command = None
        self.MPC_dt = 0.0
        self.prev_mpc_tstep = 0.0
        self.command_tstep = copy.deepcopy(self.controller.rollout_fn.dynamics_model._traj_tstep.detach().cpu())

    def get_rollout_fn(self, **kwargs):
        rollout_fn = SwiperRollout(**kwargs)
        return rollout_fn

    def predict_next_state(self, t_step, curr_state):
        # predict next state
        # given current t_step, integrate to t_step+mpc_dt
        t1_idx = find_first_idx(self.command_tstep, t_step) - 1
        t2_idx = find_first_idx(self.command_tstep, t_step + self.MPC_dt) #- 1

        # integrate from t1->t2
        for i in range(t1_idx, t2_idx):
            command = self.command[0][i]
            curr_state = self.controller.rollout_fn.dynamics_model.get_next_state(curr_state, command, self.MPC_dt)
    
        return curr_state

    def base_get_command(self, t_step, curr_state, control_dt=0.03, WAIT=False):
        if(WAIT):
            next_command, val, info, best_action = self.control_process.get_command_debug(t_step, curr_state.cpu().numpy(), integrate_act=False, control_dt=control_dt)
        else:
            next_command, val, info, best_action = self.control_process.get_command(t_step, curr_state.cpu().numpy(), integrate_act=False, control_dt=control_dt)

        return next_command

    def get_command(self, t_step, curr_state, control_dt=0.0, WAIT=False):
        if self.control_process != None:
            return self.base_get_command(t_step, curr_state, control_dt, WAIT)

        if self.command is not None:
            print("next state")
            curr_state = self.predict_next_state(t_step, curr_state)

        shift_steps = find_first_idx(self.command_tstep, t_step + self.MPC_dt)
        if(shift_steps < 0): shift_steps = 0

        self.command = list(self.controller.optimize(curr_state, shift_steps=shift_steps))
        self.MPC_dt = t_step - self.prev_mpc_tstep
        self.prev_mpc_tstep = copy.deepcopy(t_step)

        return self.command

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
        print(mppi_params['action_lows'])
        print(mppi_params['action_highs'] )
        init_action = torch.zeros((mppi_params['horizon'], dynamics_model.d_action), **self.tensor_args)
        mppi_params['init_mean'] = init_action
        mppi_params['rollout_fn'] = rollout_fn
        mppi_params['tensor_args'] = self.tensor_args
        controller = MPPI(**mppi_params)
        self.exp_params = exp_params
        return controller