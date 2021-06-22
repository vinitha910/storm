import sys
sys.path.insert(1, '../')

import torch
import numpy as np
from autolab_core import YamlConfig

from ...mpc.model.model_base import DynamicsModelBase
from ...mpc.model.integration_utils import build_int_matrix

# from dynamics_network.network import DynamicsNetwork 
# import dynamics_network.graph_construction as gc 

# from fps import fps_samples 
# from particle_gym.vector_env import VectorEnv 

class SimDynamicsModel(DynamicsModelBase):
    def __init__(self, config, vector_env, tensor_args={'device':'cpu','dtype':torch.float32}):
        self.tensor_args = tensor_args

        # config = YamlConfig(config_path)
        self.dt = config['model']['dt']
        self.d_action = 3 # [x, y, theta]
        self.batch_size = config['mppi']['num_particles']
        self.horizon = config['mppi']['horizon']
        self.num_traj_points = config['mppi']['horizon']
        self.threshold = config['model']['threshold']
        self.vector_env = vector_env

        self._integrate_matrix = build_int_matrix(self.num_traj_points,device=self.tensor_args['device'],
                                                  dtype=self.tensor_args['dtype'])
        dt_array = [self.dt] * int(1.0 * self.num_traj_points) #+ [self.dt * 5.0] * int(0.3 * self.num_traj_points)
        self._dt_h = torch.tensor(dt_array, **self.tensor_args)
        self.traj_dt = self._dt_h
        self._traj_tstep = torch.matmul(self._integrate_matrix, self._dt_h)
        # self.vector_env = vector_env
        
    # def init_model(self, config_path, sim_model=True):
    #     if sim_model:
    #         self.vector_env = VectorEnv(config['env_config'])
    #         return 

        # gnn_config = config['model_config']
        # gnn_config['layer_config'] = \
        #     gnn_config['layer_config'] * gnn_config['num_layers']

        # self.gnn_network = DynamicsNetwork(gnn_config)

        # model_path = config['model_dir'] + config['model_name'] + '.pth'
        # self.gnn_network.load(model_path)

        # self.gnn_network.eval_mode()

    def get_next_state(self, curr_state, act, dt):
        """
        Args:
            curr_state: {
                particle_poses: torch.Tensor [batch_size, num_particles, 3] 
                states: torch.Tensor [batch_size, H, W]
            }
            act: [x_m, y_m, heading_rad]
            dt: number of simulation frames

        Return:
            out_states: {
                particle_poses: torch.Tensor [batch_size, num_particles, 3] 
                states: torch.Tensor [batch_size, H, W]
            }
        """
        # graph = gc.construct_graph(
        #   curr_state['particle_node_features'], 
        #   act['tool_node_features'], 
        #   act['action_edge_features']
        # )

        # Move tool to starting pose above particles
        self.vector_env.vector_set_particles(curr_state['particle_poses'])
        print('applying action')
        print(act)
        return self.apply_sim_action(act.repeat(self.batch_size, 1).float())

    def apply_sim_action(self, act):
        """
        Args:
            act: torch.Tensor [batch_size, d_act]

        Return:
            state: {
                particle_poses: torch.Tensor [batch_size, num_particles, 3] 
                states: torch.Tensor [batch_size, H, W]
            }
        """
        act[:,:2] *= 0.2
        act[:,-1] *= np.pi 

        for idx in range(self.batch_size):
            self.vector_env.envs[idx].init_step(act[idx].cpu().numpy(), world_coor=True)

        self.vector_env.simulate(render=False)

        # Move tool until tool touches the table
        tool_on_table = np.array([False]*self.batch_size)
        while not tool_on_table.all():
            for idx in range(self.batch_size):
                if not tool_on_table[idx]: 
                    tool_on_table[idx] = self.vector_env.envs[idx].move_swiper_down(0.001)
            self.vector_env.simulate(render=False)

        # Push the particles for dt frames
        for i in range(self.dt):
            for idx in range(self.batch_size):
                self.vector_env.envs[idx].move_swiper(0.001)
            self.vector_env.simulate(render=False)

        self.vector_env.simulate(render=True)

        return {
            'particle_poses': torch.stack([torch.tensor(env.get_particles_positions()) for env in self.vector_env.envs]),
            'states': torch.stack([torch.tensor(env.get_segmentation().squeeze()) for env in self.vector_env.envs])
        }

    def rollout_open_loop(self, start_state, act_seq):
        """
        Return sequence of costs and states encountered
        by simulating a batch of action sequences

        Args:
            start_state: {
                particle_poses: torch.Tensor [batch_size, num_particles, 3] 
                states: torch.Tensor [batch_size, H, W]
            }
            action_seq: torch.Tensor [batch_size, horizon, d_act]

        Return:
            out_states: torch.Tensor [batch_size, horizon, H, W]
        """

        self.vector_env.vector_set_particles(start_state['particle_poses'])

        out_states = torch.tensor([])
        for i in range(self.horizon):
            states = self.apply_sim_action(act_seq[:,i,:])['states']
            out_states = torch.cat((out_states, states.unsqueeze(1)), axis=1)
            print(i)

        return out_states

        # samples = fps_samples(torch.nonzero(start_state.squeeze()), self.threshold)/(H - 1)
