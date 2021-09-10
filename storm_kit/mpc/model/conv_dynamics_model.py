import sys
sys.path.insert(1, '../')

import torch
import numpy as np
from math import fabs
from autolab_core import YamlConfig
from torch_geometric.data import Data, Batch

from ...mpc.model.model_base import DynamicsModelBase
from ...mpc.model.integration_utils import build_int_matrix

from physical_interaction_video_prediction_pytorch.networks import ConvLSTM, network
from physical_interaction_video_prediction_pytorch.options import Options

from particle_gym.rectangle import Rectangle 
from visualize import visualize_traj
from batch_tools import BatchTools

class ConvLSTMDynamicsModel(DynamicsModelBase):
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
        
        # Initialize dynamics model
        self.init_model(config['model']['model_dir'] + config['model']['model_name'])

    def init_model(self, chkpt_path):
        print("Loading model " + chkpt_path)
        opt = Options().parse()
        opt.pretrained_model = chkpt_path
        opt.sequence_length = self.horizon
        self.model = network(
            opt.channels, 
            opt.height,
            opt.width, 
            -1,
            opt.schedsamp_k,
            opt.use_state,
            opt.num_masks,
            opt.model=='STP',
            opt.model=='CDNA',
            opt.model=='DNA',
            opt.context_frames
        )
        self.model.load_state_dict(torch.load(opt.pretrained_model))
        self.model.eval()
        self.model.to(self.tensor_args['device'])
        torch.set_grad_enabled(False)

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
        action_edge_features[:,2] = abs(act[1])

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

        tool_poses = start_state[:,-3:].repeat(self.batch_size, 1)
        self.tools.update_poses(tool_poses)

        action_features = torch.zeros((
            self.batch_size, 
            self.horizon,
            self.model_action_len),
            device=self.tensor_args['device']
        )
        action_features[:,:,1] = act_seq[:,:,0] # angular velocity
        action_features[:,:,2] = act_seq[:,:,1]      # speed

        exec_act_seq = act_seq.clone()
        exec_act_seq[:,:,0] = exec_act_seq[:,:,0]*self.max_ang_vel
        exec_act_seq[:,:,1] = exec_act_seq[:,:,1]*self.max_speed
        
        seq_action_features = []
        for t in range(self.horizon):
            tool_features = self.get_tool_particle_positions_px().flatten(1,2)
            # B x 21 (18 tool node values, 3 action vals)
            seq_action_features.append(torch.cat((
                tool_features, 
                action_features[:,t,:]),dim=1)
            )

            new_tool_poses = self.tools.integrate_action_step(exec_act_seq[:,t,:])
            self.tools.update_poses(new_tool_poses)

        seq_action_features = torch.stack(seq_action_features)

        start_state = start_state[:,:-3].reshape(1,128,128).repeat(self.batch_size, 1, 1).unsqueeze(1)
        out_states = self.model.forward_from_single(start_state.double(), seq_action_features.double(), self.horizon)
        out_states = torch.cat(out_states, dim=1)
        out = torch.stack(torch.where(out_states > 0.5)).T
        out[:,2:] = out[:,2:]/127.

        _, counts = out[:,0].unique(dim=0, return_counts=True)
        batches = torch.split(out[:,1:], list(counts))
        sequences = []
        for batch in batches:
            _, counts = batch[:,0].unique(dim=0, return_counts=True)
            seq = torch.split(batch[:, 1:], list(counts))
            sequences.append(seq)

        return sequences

        # from line_profiler import LineProfiler
        # lp = LineProfiler()
        # lp_wrapper = lp(self.rollout)
        # lp_wrapper(samples, act_seq[:,i,:])
        # lp.print_stats()
        # return
