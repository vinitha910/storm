import torch
import numpy as np 
from chamferdist import ChamferDistance
import time 

from ...mpc.model.cont_gnn_dynamics_model import GNNDynamicsModel
from ...mpc.model.sim_dynamics_model import SimDynamicsModel
from ...mpc.rollout.rollout_base import RolloutBase

class SwiperRollout(RolloutBase):

    def __init__(self, exp_params, env=None, tensor_args={'device':'cpu','dtype':torch.float32}):
        print(tensor_args)
        self.tensor_args = tensor_args
        self.exp_params = exp_params
        mppi_params = exp_params['mppi']

        self.horizon = mppi_params['horizon'] # model_params['dt']
        self.batch_size = exp_params['model']['batch_size']

        self.dynamics_model = GNNDynamicsModel(
            config=self.exp_params,
            tensor_args=self.tensor_args
        )
        
        self.goal_pts = None

    def cost_fn(self, states):
        '''
            Returns cost of action sequence

            Args:
                states: torch.Tensor [batch_size, horizon, H, W]
        '''
        chamferDist = ChamferDistance()

        costs = []
        # B, T, _ = states.shape
        H = W = 128
        # states = states[:,:,:-3].reshape(B,T,H,W)
        # batch: [horizon, H, W]
        batches = []
        for t in range(self.horizon):
            batch = chamferDist(states[:,t,:,:].float(), self.goal_pts, bidirectional=True, reduction='none')
            batches.append(batch.unsqueeze(1))
        costs = torch.cat(batches, dim=1)

        return costs

        # source_pts = torch.stack(torch.where(state > 0.5)).T.to(**self.tensor_args)/(H-1)
        # if len(source_pts) == 0:
        #     cost = 10000
        # else: 
        # if state.min() < 0 or state.max() > 1 or state.max() < 0 or state.min() > 1:
        #     cost = 10000
        # else:

    def rollout_fn(self, start_state, act_seq):
        '''
            Returns a sequence of costs and states after simulating a
            batch of action sequences.

            Args:
                start_state: torch.Tensor [H, W]
                act_seq: torch.Tensor [num_rollouts, horizon, act_vec]
        '''
        rollout_time = time.time()
        out_states = self.dynamics_model.rollout_open_loop(start_state, act_seq)
        rollout_time = time.time() - rollout_time

        cost_seq = self.cost_fn(out_states)
        # print(cost_seq)
        sim_trajs = dict(
            actions=act_seq,
            costs=cost_seq,
            rollout_time=rollout_time,
            state_seq=out_states
        )

        return sim_trajs

    def update_params(self, goal_state=None, tool_pose=None):
        '''
            Updates the goal state

            Args:
                goal_state: torch.Tensor [H x W]
        '''
        if goal_state is not None:
            _, H, W = goal_state.shape
            goal_pts = torch.stack(torch.where(goal_state.squeeze() == 1.)).T.to(**self.tensor_args)/(H-1)
            self.goal_pts = goal_pts.unsqueeze(0).repeat(self.batch_size,1,1)

        # if tool_pose is not None:
        #     self.dynamics_model.set_tool_pose(tool_pose)

    def __call__(self, start_state, act_seq):
        return self.rollout_fn(start_state, act_seq)

    def current_cost(self, current_state):
        '''
            Returns that cost for the current state.

            Args:
                current_state: torch.Tensor [1, H, W]
        '''

        current_state = current_state.to(**self.tensor_args).unsqueeze(0)

        cost = self.cost_fn(current_state)
        return cost, current_state