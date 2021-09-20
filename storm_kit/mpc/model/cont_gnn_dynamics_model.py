import sys
sys.path.insert(1, '../')

import torch
import numpy as np
from math import fabs
from autolab_core import YamlConfig
from torch_geometric.data import Data, Batch

from ...mpc.model.model_base import DynamicsModelBase
from ...mpc.model.integration_utils import build_int_matrix

from particle_dynamics_network.lit_graph_network import LitGraphNetwork
from particle_dynamics_network.lit_utils import fps_samples, construct_graph, batch_construct_graph, seq_batch_kde
from particle_gym.utils import normalize_angle_rad, abs_diff
from particle_dynamics_network.heatmap import uneven_batch_kde
from particle_dynamics_network.lit_visualize import visualize_graph_rollouts, visualize_rollouts, visualize_graph, visualize_seq_rollouts, plot_grad_flow, visualize_model_rollouts

from particle_gym.rectangle import Rectangle 
from visualize import visualize_traj
from batch_tools import BatchTools

class GNNDynamicsModel(DynamicsModelBase):
    def __init__(self, config, tensor_args={'device':'cuda','dtype':torch.float32}):
        self.tensor_args = tensor_args
        
        # Camera properties
        self.wTc = torch.tensor(config['cam_params']['wTc']).float()
        self.cTi = torch.tensor(config['cam_params']['cTi']).float()
        self.cam_props_width = config['cam_params']['cam_props_width']

        self.max_speed = config['model']['max_speed']
        self.max_ang_vel = config['model']['max_ang_vel']
        
        # config = YamlConfig(config_path)
        self.dt = 1 #config['model']['dt']
        self.model_action_len = 3 # [slice, speed, angular velocity]
        self.n_dofs = 1
        self.d_action = 2 # [speed, angular velocity]
        self.batch_size = config['mppi']['num_particles']
        self.horizon = config['mppi']['horizon']
        self.num_traj_points = config['mppi']['horizon']
        self.threshold = config['model']['sampling_threshold']

        self._integrate_matrix = build_int_matrix(self.num_traj_points,device=self.tensor_args['device'],
                                                  dtype=self.tensor_args['dtype'])
        dt_array = [self.dt] * int(1.0 * self.num_traj_points) #+ [self.dt * 5.0] * int(0.3 * self.num_traj_points)
        self._dt_h = torch.tensor(dt_array, **self.tensor_args)
        self.traj_dt = self._dt_h
        self._traj_tstep = torch.matmul(self._integrate_matrix, self._dt_h)
        
        # 2D projection of tool used to calculate positions of particles after action is applied
        tool_config = config['sim_env_config']['tool_config']
        # self.tool = Rectangle(0., 0., tool_config['length'], tool_config['width'])
        self.tools = BatchTools(batch_size=self.batch_size, l_m=tool_config['length'], w_m=tool_config['width'])
        
        self.curr_tool_pose = None
        # Initialize dynamics model
        self.init_model(config['model']['model_dir'] + config['model']['model_name'])

    def init_model(self, chkpt_path):
        print("Loading model " + chkpt_path)
        self.model = LitGraphNetwork.load_from_checkpoint(chkpt_path)
        self.model.eval()
        self.model.to(self.tensor_args['device'])
        torch.set_grad_enabled(False)
        
    def world_to_pixel_coord(self, x, y, depth=None):
        if depth == None:
            depth = 0.01 # Top of the table

        X = torch.tensor([[x], [depth], [y], [1]]).float()
        # X = np.matrix([[x], [depth], [y], [1]]) 

        # Convert point from world frame to overhead camera frame
        p = self.wTc.matmul(X).float()
        # p = self.wTc * X
        p[2] = fabs(p[2])
        p = self.cTi.matmul(p[:3]) / fabs(p[2])
        # p = self.cTi * p[:3] / fabs(p[2])

        # Must subtract y from length of image
        return torch.tensor([int(p[0]), self.cam_props_width - int(p[1])])
        # return (int(p[0]), self.cam_props_width - int(p[1]))

    def get_fps_samples(self, curr_state):
        '''
            Args:
                curr_state: torch.Tensor [B, H, W]

            Return:
                batch_samples: [[N0, 2], [N1, 2],...,[NB, 2]]
        '''
        H, W = curr_state.squeeze().shape
        state = curr_state.squeeze()
        samples = torch.nonzero(state)
        if len(samples) != 0:
            samples = fps_samples(samples)/(H - 1)

        return samples

    def get_tool_particle_positions_px(self, particles=None):
        if particles == None:
            particles = self.tools.get_transformed_particles()

        _, M, _ = particles.shape
        x = particles[:,:,0].unsqueeze(1)
        y = particles[:,:,1].unsqueeze(1)
        depth = depth = torch.ones(self.batch_size,1,M).type_as(particles)*0.01
        ones = torch.ones(self.batch_size,1,M).type_as(particles)
        
        # B x 4 x M
        X = torch.cat((y, depth, x, ones),dim=1)
        # 4 x 4 * B x 4 x M
        p = self.wTc.type_as(X).matmul(X)
        # B x 3 x M 
        p[:,2,:] = p[:,2,:].abs()
        p = self.cTi.type_as(p).matmul(p[:,:3,:])/p[:,2,:].abs().unsqueeze(1)
        # B x M x 2
        p = p[:,:2,:]
        p = torch.swapaxes(p, 2, 1)

        p[:,:,1] = self.cam_props_width - p[:,:,1]

        # Normalize
        return p/(self.cam_props_width-1) 

    def get_next_state(self, curr_state, act, slice_dirt=False):
        """
        Args:
            curr_state: torch.Tensor [batch_size, H, W]
            act: torch.Tensor [d_act]
            dt: number of simulation frames

        Return:
            out_states: [batch_size, H, W]
        """

        # Assume dt is constant for now
        H = W = 128
        # import ipdb; ipdb.set_trace()
        
        tool_poses = curr_state[-3:].unsqueeze(0).repeat(self.batch_size, 1).to(**self.tensor_args)
        self.tools.update_poses(tool_poses)

        start_state = curr_state[:-3].reshape(1,H,W).to(**self.tensor_args)
        
        # [[N0, 2], [N1, 2],...,[NB, 2]]
        # Sample points on initial segmentation, if initial segmentation 
        # has no segmented pixels, return batch of blank segmentations
        samples = self.get_fps_samples(start_state).unsqueeze(0).float()

        # [slice, speed, angular velocity]
        action_edge_features = torch.zeros((1, self.model_action_len))
        if slice_dirt:
            action_edge_features[:,1] = 1    
        action_edge_features[:,1] = act[0]
        action_edge_features[:,2] = act[1]

        # [B, M, 2]
        tool_node_features = self.get_tool_particle_positions_px()[0].unsqueeze(0)
        M = tool_node_features.shape[1]

        gnn_in = batch_construct_graph(
            samples,
            tool_node_features, 
            action_edge_features.type_as(samples),
        )

        out_states = self.model.rollout(gnn_in, [M])

        exec_act = act.clone().cuda()
        exec_act[0] = exec_act[0]*self.max_ang_vel
        exec_act[1] = exec_act[1]*self.max_speed
        new_poses = self.tools.integrate_action_step(exec_act.unsqueeze(0))

        return out_states

    def rollout_open_loop(self, start_state, act_seq, vis=False):
        """
        Return sequence of costs and states encountered
        by simulating a batch of action sequences

        Args:
            start_state: torch.Tensor [1, H * W + 3]
            action_seq: torch.Tensor [batch_size, horizon, d_act]

        Return:
            out_states: torch.Tensor [batch_size, horizon, H * W + 3]
        """

        # Scale action sequence to true action range
        # act_seq *= np.pi 
        tool_poses = start_state[:,-3:].repeat(self.batch_size, 1)
        self.tools.update_poses(tool_poses)

        H = W = 128
        start_state = start_state[:,:-3].reshape(1,H,W)
        
        # Sample points on initial segmentation, if initial segmentation 
        # has no segmented pixels, return batch of blank segmentations
        start_samples = self.get_fps_samples(start_state)
        new_samples = start_samples.unsqueeze(0).repeat(self.batch_size, 1, 1)

        out_states = None

        action_edge_features = torch.zeros((
            self.batch_size, 
            self.horizon, 
            self.model_action_len
        ), device=self.tensor_args['device'])
        action_edge_features[:,:,1] = act_seq[:,:,0] # angular velocity
        action_edge_features[:,:,2] = act_seq[:,:,1]      # speed

        exec_act_seq = act_seq.clone()
        exec_act_seq[:,:,0] = exec_act_seq[:,:,0]*self.max_ang_vel
        exec_act_seq[:,:,1] = exec_act_seq[:,:,1]*self.max_speed
        # Keep track of tool positions at the BEGINNING of the timestep
        seq_tool_poses = []
        tool_features = []
        out_states = []

        graphs = []
        for t in range(self.horizon):
            # import ipdb; ipdb.set_trace()
            # Get the tool node features from the current tool poses
            tool_node_features = self.get_tool_particle_positions_px()
            tool_features.append(tool_node_features)

            gnn_in = batch_construct_graph(
                new_samples, 
                tool_node_features, 
                action_edge_features[:,t,:],
                edge_radius=0.06
                # vis=True
            )
            graphs.append(gnn_in[0].clone())

            M = tool_node_features.shape[1]
            output = self.model.model(gnn_in)
            out = output.x.reshape(self.batch_size, -1, 3)[:,:,:2]
            new_samples = out[:,M:,:]
            out_states.append(new_samples.clone().unsqueeze(1))

            # Integrate action step after applying action
            new_poses = self.tools.integrate_action_step(exec_act_seq[:,t,:])
            self.tools.update_poses(new_poses)
            seq_tool_poses.append(new_poses)
            # visualize_traj(start_state.squeeze(), gnn_in[0], new_samples[0])

        # B x T x N x 2
        out_states = torch.cat(out_states,dim=1) #, torch.stack(seq_tool_poses, dim=1)
        # visualize_model_rollouts(out_states, torch.stack(tool_features, dim=1))
        # visualize_graph_rollouts(graphs)
        # kde = seq_batch_kde(out_states, H, W)

        if vis:
            return out_states, graphs

        return start_samples, out_states, torch.stack(tool_features, dim=1)

        # from line_profiler import LineProfiler
        # lp = LineProfiler()
        # lp_wrapper = lp(self.rollout)
        # lp_wrapper(samples, act_seq[:,i,:])
        # lp.print_stats()
        # return
