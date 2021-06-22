import torch
import numpy as np 
from chamferdist import ChamferDistance

from ...mpc.model.gnn_dynamics_model import GNNDynamicsModel
from ...mpc.model.sim_dynamics_model import SimDynamicsModel
from ...mpc.rollout.rollout_base import RolloutBase

class SwiperRollout(RolloutBase):

    def __init__(self, exp_params, env=None, tensor_args={'device':'cpu','dtype':torch.float32}):
        self.tensor_args = tensor_args
        self.exp_params = exp_params
        mppi_params = exp_params['mppi']

        dynamics_horizon = mppi_params['horizon'] # model_params['dt']

        self.tensor_args['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if env != None:
            self.dynamics_model = SimDynamicsModel(
                config=self.exp_params, 
                vector_env=env,
                tensor_args=self.tensor_args
            )
        else:
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

        # batch: [horizon, H, W]
        for batch in states:
            # state: [H, W]
            seq_cost = []
            for state in batch:
                source_pts = torch.stack(torch.where(state > 0.75)).T.to(**self.tensor_args)
                cost = chamferDist(source_pts.unsqueeze(0), self.goal_pts.unsqueeze(0), bidirectional=True)
                seq_cost.append(cost.item())

            costs.append(seq_cost)

        return torch.tensor(costs).to(**self.tensor_args)

    def rollout_fn(self, start_state, act_seq):
        '''
            Returns a sequence of costs and states after simulating a
            batch of action sequences.

            Args:
                start_state: torch.Tensor [H, W]
                act_seq: torch.Tensor [num_rollouts, horizon, act_vec]
        '''
        out_states = self.dynamics_model.rollout_open_loop(start_state, act_seq)
        cost_seq = self.cost_fn(out_states)
        # print(cost_seq)
        sim_trajs = dict(
            actions=act_seq,
            costs=cost_seq,
            rollout_time=0.0,
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
            self.goal_pts = torch.stack(torch.where(goal_state.squeeze() == 1.)).T.to(**self.tensor_args)
        if tool_pose is not None:
            self.dynamics_model.set_tool_pose(tool_pose)

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