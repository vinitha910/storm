import torch
import numpy as np 
from chamferdist import ChamferDistance
import time 
import matplotlib.pyplot as plt
import point_cloud_utils as pcu

from ...mpc.model.cont_gnn_dynamics_model import GNNDynamicsModel
from ...mpc.model.conv_dynamics_model import ConvLSTMDynamicsModel
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
        # self.dynamics_model = ConvLSTMDynamicsModel(
        #     config=self.exp_params,
        #     tensor_args=self.tensor_args
        # )
        
        self.goal_pts = None

    def cost_fn(self, states):
        '''
            Returns cost of action sequence

            Args:
                states: torch.Tensor [batch_size, horizon, H, W]
        '''
        states, tool_poses = states
        
        chamferDist = ChamferDistance()

        costs = []
        # B, T, _ = states.shape
        H = W = 128
        # states = states[:,:,:-3].reshape(self.batch_size,self.horizon,H,W)
        goal_pts = self.goal_pts[0].detach().cpu().numpy()
        goal_pts = np.concatenate((goal_pts,np.zeros((goal_pts.shape[0],1))),axis=1)
        costs = np.zeros((self.batch_size, self.horizon))

        for b in range(self.batch_size):
            for t in range(self.horizon):
                pose = tool_poses[b][t][:2]
                if (pose > 0.15).any() or (pose < -0.15).any():
                    costs[b][t] = 10000.
                    continue

                state = states[b][t].float().detach().cpu().numpy()
                state = np.concatenate((state,np.zeros((state.shape[0],1))),axis=1)
                forward = pcu.chamfer_distance(state, goal_pts)
                backward = pcu.chamfer_distance(goal_pts, state)
                costs[b][t] = forward + backward

        return torch.tensor(costs*100.)

        # batches = []
        # for t in range(self.horizon):
        #     batch = chamferDist(states[:,t,:,:].float(), self.goal_pts, bidirectional=True, reduction='none')**2
        #     batches.append(batch.unsqueeze(1))
        # costs = torch.cat(batches, dim=1)

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
        # from line_profiler import LineProfiler
        # lp = LineProfiler()
        # lp_wrapper = lp(self.dynamics_model.rollout_open_loop)
        # lp_wrapper(start_state, act_seq)
        # lp.print_stats()
        # return

        rollout_time = time.time()
        out_states = self.dynamics_model.rollout_open_loop(start_state, act_seq)
        rollout_time = time.time() - rollout_time
        print("Rollout Time: " + str(rollout_time))
        cost_seq = self.cost_fn(out_states)
        # print(cost_seq)
        sim_trajs = dict(
            actions=act_seq,
            costs=cost_seq,
            rollout_time=rollout_time,
            state_seq=out_states
        )

        # for (cost, act, batch) in zip(cost_seq, act_seq, out_states):
        #     fig, axes = plt.subplots(1,5, figsize=(25, 5))
        #     for i in range(len(batch)):
        #         xs = batch[i][:,0].cpu().detach().numpy()
        #         ys = batch[i][:,1].cpu().detach().numpy()
        #         for x, y in zip(xs, ys):
        #             axes[i].add_artist(plt.Circle((y, x), 0.01, color='red', linewidth=0.5, alpha=0.5))
        #         for x, y in self.goal_pts[0]:
        #             axes[i].add_artist(plt.Circle((y, x), 0.01, color='blue', linewidth=0.5, alpha=0.5))
        #     print(cost)
        #     print(act)
        #     print()
        #     plt.show()
        #     plt.close()

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