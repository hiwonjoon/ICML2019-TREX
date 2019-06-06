import gym
import numpy as np
from baselines.common.vec_env import VecEnvWrapper
from baselines.common.running_mean_std import RunningMeanStd
from baselines.common.trex_utils import preprocess
#import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F


class AtariNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(4, 16, 7, stride=3)
        self.conv2 = nn.Conv2d(16, 16, 5, stride=2)
        self.conv3 = nn.Conv2d(16, 16, 3, stride=1)
        self.conv4 = nn.Conv2d(16, 16, 3, stride=1)
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 1)


    def forward(self, traj):
        '''calculate cumulative return of trajectory'''
        x = traj.permute(0,3,1,2) #get into NCHW format
        #compute forward pass of reward network
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = x.view(-1, 784)
        x = F.leaky_relu(self.fc1(x))
        r = torch.sigmoid(self.fc2(x))
        return r

class VecPyTorchAtariReward(VecEnvWrapper):
    def __init__(self, venv, reward_net_path, env_name):
        VecEnvWrapper.__init__(self, venv)
        self.reward_net = AtariNet()
        self.reward_net.load_state_dict(torch.load(reward_net_path))
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.reward_net.to(self.device)

        self.rew_rms = RunningMeanStd(shape=())
        self.epsilon = 1e-8
        self.cliprew = 10.
        self.env_name = env_name

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
    
        #mask and normalize for input to network
        normed_obs = preprocess(obs, self.env_name)
    
        with torch.no_grad():
            rews_network = self.reward_net.forward(torch.from_numpy(np.array(normed_obs)).float().to(self.device)).cpu().numpy().squeeze()

        return obs, rews_network, news, infos

    def reset(self, **kwargs):
        obs = self.venv.reset()

    
        return obs


if __name__ == "__main__":
    pass
