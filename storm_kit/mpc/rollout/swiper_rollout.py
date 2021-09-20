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
from particle_dynamics_network.lit_utils import fps_samples
from wassdistance.layers import SinkhornDistance

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
        self.sampling_threshold = self.exp_params['model']['sampling_threshold']
    
    def cost_fn(self, start_state, states):
        '''
            Returns cost of action sequence

            Args:
                states: torch.Tensor [batch_size, horizon, H, W]
        '''
        chamferDist = ChamferDistance()
        sinkhorn = SinkhornDistance(eps=0.1, max_iter=100)
        
        start_state = start_state.repeat(self.batch_size,1,1)
        # start_cost = chamferDist(start_state.float(), self.goal_samples, bidirectional=True)
        start_cost, _, _ = sinkhorn(start_state.cpu(), self.goal_samples.cpu())
        # start_cost = start_cost.repeat(self.batch_size, 1)
        start_cost = start_cost.unsqueeze(1)*1000.

        costs = []
        # B, T, _ = states.shape
        H = W = 128
        # states = states[:,:,:-3].reshape(self.batch_size,self.horizon,H,W)
        # goal_pts = self.goal_pts[0].detach().cpu().numpy()
        # goal_pts = np.concatenate((goal_pts,np.zeros((goal_pts.shape[0],1))),axis=1)
        # costs = np.zeros((self.batch_size, self.horizon))

        # for b in range(self.batch_size):
        #     for t in range(self.horizon):
                # pose = tool_poses[b][t][:2]
                # if (pose > 0.15).any() or (pose < -0.15).any():
                #     costs[b][t] = 10000.
                #     continue

                # state = states[b][t].float().detach().cpu().numpy()
                # state = np.concatenate((state,np.zeros((state.shape[0],1))),axis=1)
                # forward = pcu.chamfer_distance(state, goal_pts)
                # backward = pcu.chamfer_distance(goal_pts, state)
                # costs[b][t] = forward + backward
        # return torch.tensor(costs*100.)
        batches = []
        for t in range(self.horizon):
            d, _, _ = sinkhorn(states[:,t,:,:].cpu(), self.goal_samples.cpu())
            batches.append(d.unsqueeze(1))
        emd_costs = torch.cat(batches, dim=1)*1000.
        # emd_costs[:,:-1] = 0.
        all_costs = torch.cat((start_cost, emd_costs), dim=1)
        delta = (all_costs[:,1:] - all_costs[:,:-1])
        # delta[torch.where(delta < 0)] = 0.
        # return emd_costs + delta

        batches = []
        for t in range(self.horizon):
            batch = chamferDist(
                states[:,t,:,:].float(), 
                self.goal_samples, 
                bidirectional=True, 
                reduction='none', 
                K=5
            )
            batches.append(batch.unsqueeze(1))
        cd_costs = torch.cat(batches, dim=1)

        # import IPython; IPython.embed()
        # all_costs = torch.cat((start_cost, cd_costs), dim=1)
        # return (all_costs[:,1:] - all_costs[:,:-1])
        # return (emd_costs + (all_costs[:,1:] - all_costs[:,:-1]).cpu())*100.
        return cd_costs*100.

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
        start_state, out_states, tool_features = self.dynamics_model.rollout_open_loop(start_state, act_seq)
        rollout_time = time.time() - rollout_time
        print("Rollout Time: " + str(rollout_time))
        cost_seq = self.cost_fn(start_state, out_states)
        # print(cost_seq)
        sim_trajs = dict(
            actions=act_seq,
            costs=cost_seq,
            rollout_time=rollout_time,
            state_seq=out_states,
            tool_seq=tool_features
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

            goal_samples = fps_samples(goal_pts*(H-1))
            self.goal_samples = goal_samples.unsqueeze(0).repeat(self.batch_size,1,1)/(H-1)

            # goal_pts = (goal_pts[torch.argmin(goal_pts)] + goal_pts[torch.argmax(goal_pts)])/2
            # self.goal_pts = goal_pts.unsqueeze(0).repeat(self.batch_size,1,1)
            # import IPython; IPython.embed()

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