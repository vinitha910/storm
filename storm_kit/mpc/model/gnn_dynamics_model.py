import sys
sys.path.insert(1, '../')

import torch
import numpy as np
from math import fabs
from autolab_core import YamlConfig
from torch_geometric.data import Data, Batch

from ...mpc.model.model_base import DynamicsModelBase
from ...mpc.model.integration_utils import build_int_matrix

from particle_gym.utils import normalize_angle_rad, abs_diff
from particle_dynamics_network.network import DynamicsNetwork 
from particle_dynamics_network.heatmap import uneven_batch_kde
import particle_dynamics_network.graph_construction as gc 

from particle_gym.rectangle import Rectangle 
from visualize import visualize_traj

class GNNDynamicsModel(DynamicsModelBase):
    def __init__(self, config, tensor_args={'device':'cpu','dtype':torch.float32}):
        self.tensor_args = tensor_args
        
        # Camera properties
        self.wTc = torch.tensor(config['cam_params']['wTc']).float()
        self.cTi = torch.tensor(config['cam_params']['cTi']).float()
        self.cam_props_width = config['cam_params']['cam_props_width']

        self.fps = 60.
        self.speed = 0.001
        self.max_speed = config['model']['max_speed']
        self.max_ang_vel = config['model']['max_ang_vel']

        # config = YamlConfig(config_path)
        self.dt = config['model']['dt']
        self.rotate_offset = np.pi/8
        self.model_action_len = 3 # [slice, speed, angular velocity]
        self.n_dofs = 1
        self.d_action = 2 # [speed, angular velocity]
        self.batch_size = config['mppi']['num_particles']
        self.horizon = config['mppi']['horizon']
        self.num_traj_points = config['mppi']['horizon']
        self.threshold = config['model']['sampling_threshold']
        self.sigma = config['model']['sigma']

        self._integrate_matrix = build_int_matrix(self.num_traj_points,device=self.tensor_args['device'],
                                                  dtype=self.tensor_args['dtype'])
        dt_array = [self.dt] * int(1.0 * self.num_traj_points) #+ [self.dt * 5.0] * int(0.3 * self.num_traj_points)
        self._dt_h = torch.tensor(dt_array, **self.tensor_args)
        self.traj_dt = self._dt_h
        self._traj_tstep = torch.matmul(self._integrate_matrix, self._dt_h)
        
        # 2D projection of tool used to calculate positions of particles after action is applied
        tool_config = config['sim_env_config']['tool_config']
        self.tool = Rectangle(0., 0., tool_config['length'], tool_config['width'])
        self.curr_tool_pose = None
        # Initialize dynamics model
        self.init_model(config['model'])

    def init_model(self, config):
        print("initializing model")
        network_config = config['network_config']
        network_config['layer_config'] = \
            network_config['layer_config'] * network_config['num_layers']

        self.gnn_network = DynamicsNetwork(network_config)

        model_path = config['model_dir'] + config['model_name'] + '.pth'
        self.gnn_network.load(model_path)

        self.gnn_network.eval_mode()
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

    def calc_distances(self, p0, points):
        return ((p0 - points)**2).sum(axis=1)

    def fps_samples(self, pts):
        farthest_pts = torch.zeros(pts.shape).to(pts.device)
        farthest_pts[0] = pts[np.random.randint(pts.shape[0])]
        distances = self.calc_distances(farthest_pts[0], pts)
        sample_distances = torch.Tensor([10000000.]).to(pts.device)
        i = 0
        while torch.min(sample_distances) > self.threshold:
            i += 1
            farthest_pts[i] = pts[torch.argmax(distances)]
            distances = torch.minimum(distances, self.calc_distances(farthest_pts[i], pts))
            new_distances = self.calc_distances(farthest_pts[i], farthest_pts[:i])
            sample_distances = torch.minimum(sample_distances, new_distances[:i])
            sample_distances = torch.cat((sample_distances, torch.Tensor([new_distances[-1]]).to(pts.device)))
        
        return farthest_pts[:i].float()

    def get_batch_samples(self, curr_state):
        B, H, W = curr_state.squeeze().shape
        batch_samples = []
        for i in range(B):
            state = curr_state.squeeze()[i]
            samples = self.fps_samples(torch.nonzero(state))/(H - 1)
            batch_samples.append(samples.to(**self.tensor_args))

        return batch_samples

    def get_tool_particle_positions_px(self, particles=None):
        if particles == None:
            particles = torch.tensor(self.tool.get_particles())

        N, _ = particles.shape
        for i in range(N):
            x = particles[:,:2][i][1]
            y = particles[:,:2][i][0]
            particles[:,:2][i] = self.world_to_pixel_coord(x, y)
            
        return particles.to(**self.tensor_args)/(self.cam_props_width-1) 

    def get_next_state(self, curr_state, act, dt):
        """
        Args:
            curr_state: torch.Tensor [batch_size, H, W]
            act: torch.Tensor [d_act]
            dt: number of simulation frames

        Return:
            out_states: torch.Tensor [batch_size, H, W]
        """

        # Assume dt is constant for now
        B, H, W = curr_state.shape
        samples = self.get_batch_samples(torch.tensor(curr_state))

        # [slice, speed, angular velocity]
        action_edge_features = torch.zeros((B, self.model_action_len))
        action_edge_features[:,1] = abs(float(act[0]))
        action_edge_features[:,2] = float(act[1])

        x, y, theta = self.curr_tool_pose.cpu().numpy()
        self.tool.update(x, y, theta)
        tool_node_features = torch.stack([self.get_tool_particle_positions_px()]*B)
        M = tool_node_features.shape[1]
        
        gnn_in = gc.batch_construct_graph(
            samples, 
            tool_node_features, 
            action_edge_features.to(**self.tensor_args),
            device=self.tensor_args['device']
        )
        
        out_gnn = self.gnn_network.rollout(gnn_in)
        new_samples = self.gnn_network.get_nodes(out_gnn, [M]*B)
        return uneven_batch_kde(new_samples, H, W, self.sigma, self.tensor_args['device'])
       
    def set_tool_pose(self, pose):
        x, _, y, theta = pose
        theta = normalize_angle_rad(theta)
        self.curr_tool_pose = torch.tensor([x, y, theta]).to(**self.tensor_args)
        self.tool.update(x, y, theta)
     
    def rollout_open_loop(self, start_state, act_seq):
        """
        Return sequence of costs and states encountered
        by simulating a batch of action sequences

        Args:
            start_state: torch.Tensor [batch_size, H, W]
            action_seq: torch.Tensor [batch_size, horizon, d_act]

        Return:
            out_states: torch.Tensor [batch_size, horizon, H, W]
        """

        # Scale action sequence to true action range
        # act_seq *= np.pi 

        B, H, W = start_state.squeeze().shape
        out_states = []
        samples = self.get_batch_samples(start_state)

        # [slice, speed, angular velocity]
        action_edge_features = torch.zeros((B, self.horizon, self.model_action_len))
        action_edge_features[:,:,1] = abs(act_seq[:,:,0])
        action_edge_features[:,:,2] = act_seq[:,:,1]

        x, y, theta = self.curr_tool_pose.cpu().numpy()
        self.tool.update(x, y, theta)
        tool_node_features = torch.stack([self.get_tool_particle_positions_px()]*B)
        M = tool_node_features.shape[1]
        # import IPython; IPython.embed()

        out_states = []
        for t in range(self.horizon):
            if t == 0:
                # Construct batch of graphs at timestep t using previous timestep rollout
                opt_data = {}
                st = gc.batch_construct_graph(
                    samples, 
                    tool_node_features, 
                    action_edge_features[:,t,:],
                    device=self.tensor_args['device'],
                    opt_data=opt_data
                )
                num_action_edges = opt_data['num_action_edge_attr']
                x = torch.tensor([x]*B).to(**self.tensor_args)
                y = torch.tensor([y]*B).to(**self.tensor_args)
                theta = torch.tensor([theta]*B).to(**self.tensor_args)
                
            else:
                st, num_action_edges = gc.batch_update_graph(
                    out_gnn,
                    tool_node_features,
                    num_action_edges,
                    action_edge_features[:,t,:],
                    device=self.tensor_args['device']
                )

            out_gnn = self.gnn_network.rollout(st)
            new_samples = self.gnn_network.get_nodes(out_gnn, [M]*B)
            out_states.append(uneven_batch_kde(samples, H, W, self.sigma, self.tensor_args['device']))

            # import IPython; IPython.embed()
            visualize_traj(start_state.squeeze()[0], st[0], new_samples[0])
            arc_length = abs(act_seq[:,t,0])*self.max_speed
            arc_theta = act_seq[:,t,1]*self.max_ang_vel
            radius = arc_length/(arc_theta + 1e-10)
            # print(t, arc_length, arc_length)
            cx = x - radius*torch.cos(theta - np.pi/2)
            cy = y + radius*torch.sin(theta - np.pi/2)
            
            dx = radius * torch.cos(theta + arc_theta - np.pi/2)
            dy = radius * torch.sin(theta + arc_theta - np.pi/2)
            
            x = cx + dx
            y = cy - dy
            theta = arc_theta + theta

            tool_node_features = []
            for xi, yi, thi in zip(x, y, theta):
                self.tool.update(xi.item(), yi.item(), thi.item())
                tool_node_features.append(self.get_tool_particle_positions_px())

            tool_node_features = torch.stack(tool_node_features)

        return torch.stack(out_states, axis=1)
        
        # from line_profiler import LineProfiler
        # lp = LineProfiler()
        # lp_wrapper = lp(self.rollout)
        # lp_wrapper(samples, act_seq[:,i,:])
        # lp.print_stats()
        # return
