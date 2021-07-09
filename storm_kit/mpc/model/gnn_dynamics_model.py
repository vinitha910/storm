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

class GNNDynamicsModel(DynamicsModelBase):
    def __init__(self, config, tensor_args={'device':'cpu','dtype':torch.float32}):
        self.tensor_args = tensor_args
        
        # Camera properties
        self.wTc = torch.tensor(config['cam_params']['wTc']).float()
        self.cTi = torch.tensor(config['cam_params']['cTi']).float()
        self.cam_props_width = config['cam_params']['cam_props_width']

        self.fps = 60.
        self.speed = 0.001

        # config = YamlConfig(config_path)
        self.dt = config['model']['dt']
        self.rotate_offset = np.pi/8
        self.model_action_len = 4 # [slice, rotate_rad, theta_x, theta_y]
        self.n_dofs = 1
        self.d_action = 1 # [heading]
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
            batch_samples.append(samples)

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

    def construct_or_update_graph(self, samples, tool_node_features, action_edge_features, num_action_edge_attr_idx=0, opt_data={}):
        # Only construct graphs from scratch if not given graph, otherwise update
        # the tool node features and actions edges/features in the existing graph
        if isinstance(samples, torch.Tensor):
            # from line_profiler import LineProfiler
            # lp = LineProfiler()
            # lp_wrapper = lp(gc.construct_graph)
            # lp_wrapper(samples, 
            #     tool_node_features, 
            #     action_edge_features,
            #     opt_data=opt_data)
            # lp.print_stats()
            # return

            return gc.construct_graph(
                samples, 
                tool_node_features, 
                action_edge_features,
                opt_data=opt_data
            ), num_action_edge_attr_idx

        assert('num_action_edge_attr' in opt_data.keys())
        out, num_action_edges = gc.update_graph(
            samples,
            tool_node_features.to(**self.tensor_args),
            opt_data['num_action_edge_attr'][num_action_edge_attr_idx],
            action_edge_features.to(**self.tensor_args)
        )
        opt_data['num_action_edge_attr'][num_action_edge_attr_idx] = num_action_edges
        num_action_edge_attr_idx += 1
        return out, num_action_edge_attr_idx

    def batch_rotate_action_graphs(self, batch_samples, actions, directions, opt_data={}):
        """
        Constrct a single graph for a discrete rotate action
    
        Args:
            batch_samples: torch.Tensor [[N0 x 2], [N1 x 2],...,[NB x 2]]
            actions: [B x 1]; action params: [heading_rad]
            directions: [B x 1] The direction the tool is rotating
            tstep: float The current timestep
            opt_data: dict Optional data containing number of tool nodes and action edges

        Return:
            out_graph: torch.geometric.data.Data
        """

        graphs = []
        num_action_edge_attr_idx = 0
        for i in range(len(batch_samples)):

            # Set tool position at initial timestep
            x, y, theta = self.curr_tool_pose[i].cpu().numpy()
            self.tool.update(x, y, theta)

            # from line_profiler import LineProfiler
            # lp = LineProfiler()
            # lp_wrapper = lp(self.get_tool_particle_positions_px)
            # lp_wrapper()
            # lp.print_stats()
            # return

            # Tool node features at initial timestep
            tool_node_features = self.get_tool_particle_positions_px()

            # [0, rotate_theta_rad, 0, 0]
            action_edge_features = torch.zeros(self.model_action_len)
            action_edge_features[1] = directions[i]

            graph, num_action_edge_attr_idx = self.construct_or_update_graph(
                batch_samples[i].to(**self.tensor_args),
                tool_node_features,
                action_edge_features,
                num_action_edge_attr_idx=num_action_edge_attr_idx,
                opt_data=opt_data
            )
            # if i == 0:
            #     print(self.tool.theta)
            #     print(directions[i].item())
            #     print(actions[i].item())
            #     print()
            #     gc.visualize(graph)

            # Update tool heading
            self.tool.rotate(actions[i].item())
            self.curr_tool_pose[i] = torch.tensor([self.tool.cx, self.tool.cy, self.tool.theta])

            graphs.append(graph.to(self.tensor_args['device']))

        if torch.cuda.device_count() > 1:
            return graphs

        return Batch.from_data_list(graphs)

    def batch_push_action_graph(self, batch_samples, opt_data={}):
        """
        Constrct a batch of graphs for a push action
    
        Args:
            samples: list [N x 2]

        Return:
            out_graph: torch.geometric.data.Data
            M: float Number of tool nodes
        """
        graphs = []
        num_action_edge_attr_idx = 0
        for i in range(len(batch_samples)):

            # Set tool position at initial timestep
            x, y, theta = self.curr_tool_pose[i].cpu().numpy()
            self.tool.update(x, y, theta)

            # Update pushing heading
            # heading = self.tool.update_direction(directions[i])
            heading = self.tool.theta

            # Tool node features at initial timestep
            tool_node_features = self.get_tool_particle_positions_px()

            # [0, 0, theta_x, theta_y]
            theta = torch.tensor(heading)        
            heading_vec = \
                torch.tensor([torch.cos(theta), torch.sin(theta)])
            action_edge_features = torch.zeros(self.model_action_len)
            action_edge_features[-2:] = heading_vec

            graph, num_action_edge_attr_idx = self.construct_or_update_graph(
                batch_samples[i].to(**self.tensor_args),
                tool_node_features,
                action_edge_features,
                num_action_edge_attr_idx=num_action_edge_attr_idx,
                opt_data=opt_data
            )
            # if i == 0:
            #     print(heading)
            #     print()
            #     gc.visualize(graph)

            # Update tool position
            dist = (self.speed * self.fps) * (self.dt/self.fps)
            heading_vec = \
                np.array([np.cos(heading), np.sin(heading)])
            self.tool.translate_along_vector(heading_vec, dist)

            self.curr_tool_pose[i] = torch.tensor([self.tool.cx, self.tool.cy, self.tool.theta])

            graphs.append(graph.to(self.tensor_args['device']))

        if torch.cuda.device_count() > 1:
            return graphs

        return Batch.from_data_list(graphs).to(self.tensor_args['device'])

    def rollout(self, samples, act):
        '''
        Rollout a batch of states for a single timestep

        Args: 
            samples: list [[N0 x 2], [N1 x 2],...,[NB x 2]]
            act: torch.Tensor [B x d_act]

        Return:
            out_points: list [[N0 x 2], [N1 x 2],...,[NB x 2]]
        '''
        # torch.cuda.empty_cache()
        # act = torch.tensor([normalize_angle_rad(a.item()) for a in act]).unsqueeze(1).to(**self.tensor_args)

        # Calculate number of discrete rotations to reach desired heading for every
        # tool in the batch 
        curr_theta = self.curr_tool_pose[:,-1].clone().unsqueeze(1) # [B x d_act]
        # num_offsets = torch.round(abs_diff(act, curr_theta)/self.rotate_offset)
        # import IPython; IPython.embed()
        # directions = abs_diff(act, curr_theta+num_offsets*self.rotate_offset) < self.rotate_offset
    
        # Update counter-clockwise directions
        # rotate_directions = torch.ones(act.shape).to(**self.tensor_args)
        # rotate_directions[torch.stack(torch.where(directions.squeeze() == False)).T] = -1

        num_offsets = torch.round((act - curr_theta)/self.rotate_offset)
        rotate_directions = torch.tensor(np.sign(num_offsets.cpu().numpy())).to(**self.tensor_args)
        num_offsets = abs(num_offsets)

        graphs = samples.copy()
        opt_data = {}
        # Convert desired heading for each tool in discrete steps and rollout for each step
        while sum(num_offsets) != 0:
            # Find indices of tools that need to be rotated
            indices = torch.where(num_offsets > 0)[0]

            # Get the samples and actions for the graphs that need to be reconstructed 
            graphs_subset = [graphs[i.item()] for i in indices]
            act_subset = curr_theta[indices] + (self.rotate_offset*rotate_directions[indices])
            # from line_profiler import LineProfiler
            # lp = LineProfiler()
            # lp_wrapper = lp(self.batch_rotate_action_graphs)
            # lp_wrapper(graphs_subset, 
            #         act_subset,
            #         rotate_directions[indices],
            #         opt_data=opt_data)
            # lp.print_stats()
            # return

            # Reconstruct the graphs
            gnn_in = \
                self.batch_rotate_action_graphs(
                    graphs_subset, 
                    act_subset,
                    rotate_directions[indices],
                    opt_data=opt_data
                )

            gnn_out = self.gnn_network.rollout(gnn_in, opt_data['num_tool_nodes'])

            # Update the samples
            for i, j in zip(indices, range(len(indices))):
                graphs[i.item()] = gnn_out[j]   
            
            # Update the tools that have been rotated for the next iteration
            num_offsets[indices] -= 1
            curr_theta[indices] = act_subset
            self.curr_tool_pose[:,-1] = curr_theta.squeeze()

        # import IPython; IPython.embed()
        # Push tool along heading vector
        gnn_in = self.batch_push_action_graph(
            graphs, opt_data=opt_data
        )

        gnn_out = self.gnn_network.rollout(gnn_in, opt_data['num_tool_nodes'])
        return self.gnn_network.get_nodes(gnn_out, opt_data['num_tool_nodes'])

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
        gnn_out = self.rollout(samples, torch.tensor(act).repeat(B, 1).to(**self.tensor_args))

        return uneven_batch_kde(gnn_out, H, W, self.sigma, self.tensor_args['device'])

    def set_tool_pose(self, pose):
        x, _, y, theta = pose
        theta = normalize_angle_rad(theta)
        self.curr_tool_pose = torch.tensor([[x, y, theta]]*self.batch_size).to(**self.tensor_args)
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

        _, H, W = start_state.squeeze().shape
        out_states = []
        samples = self.get_batch_samples(start_state)

        for i in range(self.horizon):
            # from line_profiler import LineProfiler
            # lp = LineProfiler()
            # lp_wrapper = lp(self.rollout)
            # lp_wrapper(samples, act_seq[:,i,:])
            # lp.print_stats()
            # return
            samples = self.rollout(samples, act_seq[:,i,:])

            # [[B x H x W]]
            out_states.append(uneven_batch_kde(samples, H, W, self.sigma, self.tensor_args['device']))

        # B x T x H x W
        return torch.stack(out_states, axis=1)

