
# coding: utf-8

# In[1]:


import pickle
import gym
import time
import numpy as np
import random
import torch
from run_test import *
import matplotlib.pylab as plt
import argparse

parser = argparse.ArgumentParser(description=None)
parser.add_argument('--env_name', default='', help='Select the environment name to run, i.e. pong')
parser.add_argument('--reward_net_path', default='', help="name and location for learned model params")
parser.add_argument('--seed', default=0, help="random seed for experiments")
parser.add_argument('--models_dir', default = ".", help="top directory where checkpoint models for demos are stored")
parser.add_argument('--save_fig_dir', help ="where to save visualizations")


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Assume that we are on a CUDA machine, then this should print a CUDA device:
print(device)


args = parser.parse_args()
env_name = args.env_name
save_fig_dir = args.save_fig_dir

if env_name == "spaceinvaders":
    env_id = "SpaceInvadersNoFrameskip-v4"
elif env_name == "mspacman":
    env_id = "MsPacmanNoFrameskip-v4"
elif env_name == "videopinball":
    env_id = "VideoPinballNoFrameskip-v4"
elif env_name == "beamrider":
    env_id = "BeamRiderNoFrameskip-v4"
else:
    env_id = env_name[0].upper() + env_name[1:] + "NoFrameskip-v4"
env_type = "atari"

seed = args.seed
torch.manual_seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)

print(env_id)

stochastic = True

reward_net_path = args.reward_net_path


env = make_vec_env(env_id, 'atari', 1, 0,
                   wrapper_kwargs={
                       'clip_rewards':False,
                       'episode_life':False,
                   })


env = VecFrameStack(env, 4)
agent = PPO2Agent(env, env_type, stochastic)



import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(4, 16, 7, stride=3)
        self.conv2 = nn.Conv2d(16, 16, 5, stride=2)
        self.conv3 = nn.Conv2d(16, 16, 3, stride=1)
        self.conv4 = nn.Conv2d(16, 16, 3, stride=1)
        self.fc1 = nn.Linear(784, 64)
        #self.fc1 = nn.Linear(1936,64)
        self.fc2 = nn.Linear(64, 1)



    def cum_return(self, traj):
        '''calculate cumulative return of trajectory'''
        sum_rewards = 0
        sum_abs_rewards = 0
        for x in traj:
            x = x.permute(0,3,1,2) #get into NCHW format
            x = F.leaky_relu(self.conv1(x))
            x = F.leaky_relu(self.conv2(x))
            x = F.leaky_relu(self.conv3(x))
            x = F.leaky_relu(self.conv4(x))
            x = x.view(-1, 784)
            x = F.leaky_relu(self.fc1(x))
            r = torch.sigmoid(self.fc2(x))
            sum_rewards += r
            sum_abs_rewards += torch.abs(r)
        return sum_rewards, sum_abs_rewards



    def forward(self, traj_i, traj_j):
        '''compute cumulative return for each trajectory and return logits'''
        cum_r_i, abs_r_i = self.cum_return(traj_i)
        cum_r_j, abs_r_j = self.cum_return(traj_j)
        return torch.cat([cum_r_i, cum_r_j]), abs_r_i + abs_r_j


reward = Net()
reward.load_state_dict(torch.load(reward_net_path))
reward.to(device)



#generate some trajectories for inspecting learned reward
checkpoint_min = 50
checkpoint_max = 600
checkpoint_step = 50

if env_name == "enduro":
    checkpoint_min = 3100
    checkpoint_max = 3650
elif env_name == "seaquest":
    checkpoint_min = 10
    checkpoint_max = 65
    checkpoint_step = 5

checkpoints_demos = []

for i in range(checkpoint_min, checkpoint_max + checkpoint_step, checkpoint_step):
        if i < 10:
            checkpoints_demos.append('0000' + str(i))
        elif i < 100:
            checkpoints_demos.append('000' + str(i))
        elif i < 1000:
            checkpoints_demos.append('00' + str(i))
        elif i < 10000:
            checkpoints_demos.append('0' + str(i))
print(checkpoints_demos)



#generate some trajectories for inspecting learned reward
checkpoint_min = 650
checkpoint_max = 1450
checkpoint_step = 50
if env_name == "enduro":
    checkpoint_min = 3625
    checkpoint_max = 4425
    checkpoint_step = 50
elif env_name == "seaquest":
    checkpoint_min = 10
    checkpoint_max = 65
    checkpoint_step = 5
elif env_name == "hero":
    checkpoint_min = 100
    checkpoint_max = 2400
    checkpoint_step = 100
checkpoints_extrapolate = []
for i in range(checkpoint_min, checkpoint_max + checkpoint_step, checkpoint_step):
        if i < 10:
            checkpoints_extrapolate.append('0000' + str(i))
        elif i < 100:
            checkpoints_extrapolate.append('000' + str(i))
        elif i < 1000:
            checkpoints_extrapolate.append('00' + str(i))
        elif i < 10000:
            checkpoints_extrapolate.append('0' + str(i))
print(checkpoints_extrapolate)



from baselines.common.trex_utils import preprocess
model_dir = args.models_dir
demonstrations = []
learning_returns_demos = []
pred_returns_demos = []
for checkpoint in checkpoints_demos:

    model_path = model_dir + "/models/" + env_name + "_25/" + checkpoint
    if env_name == "seaquest":
        model_path = model_dir + "/models/" + env_name + "_5/" + checkpoint

    agent.load(model_path)
    episode_count = 1
    for i in range(episode_count):
        done = False
        traj = []
        r = 0

        ob = env.reset()
        #traj.append(ob)
        #print(ob.shape)
        steps = 0
        acc_reward = 0
        while True:
            action = agent.act(ob, r, done)
            ob, r, done, _ = env.step(action)
            #print(ob.shape)
            traj.append(preprocess(ob, env_name))
            steps += 1
            acc_reward += r[0]
            if done:
                print("checkpoint: {}, steps: {}, return: {}".format(checkpoint, steps,acc_reward))
                break
        print("traj length", len(traj))
        print("demo length", len(demonstrations))

        demonstrations.append(traj)
        learning_returns_demos.append(acc_reward)
        pred_returns_demos.append(reward.cum_return(torch.from_numpy(np.array(traj)).float().to(device))[0].item())
        print("pred return", pred_returns_demos[-1])

learning_returns_extrapolate = []
pred_returns_extrapolate = []

for checkpoint in checkpoints_extrapolate:

    model_path = model_dir + "/models/" + env_name + "_25/" + checkpoint
    if env_name == "seaquest":
        model_path = model_dir + "/models/" + env_name + "_5/" + checkpoint

    agent.load(model_path)
    if env_name == "enduro":
        episode_count = 1
    else:
        episode_count = 3
    for i in range(episode_count):
        done = False
        traj = []
        r = 0

        ob = env.reset()
        #traj.append(ob)
        #print(ob.shape)
        steps = 0
        acc_reward = 0
        while True:
            action = agent.act(ob, r, done)
            ob, r, done, _ = env.step(action)
            #print(ob.shape)
            traj.append(preprocess(ob, env_name))
            steps += 1
            acc_reward += r[0]
            if done:
                print("checkpoint: {}, steps: {}, return: {}".format(checkpoint, steps,acc_reward))
                break
        print("traj length", len(traj))
        print("demo length", len(demonstrations))
        demonstrations.append(traj)
        learning_returns_extrapolate.append(acc_reward)
        pred_returns_extrapolate.append(reward.cum_return(torch.from_numpy(np.array(traj)).float().to(device))[0].item())
        print("pred return", pred_returns_extrapolate[-1])


env.close()






#plot extrapolation curves

def convert_range(x,minimum, maximum,a,b):
    return (x - minimum)/(maximum - minimum) * (b - a) + a


# In[12]:


buffer = 20
if env_name == "pong":
    buffer = 2
import matplotlib.pylab as pylab
params = {'legend.fontsize': 'xx-large',
         # 'figure.figsize': (6, 5),
         'axes.labelsize': 'xx-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'xx-large',
         'ytick.labelsize':'xx-large'}
pylab.rcParams.update(params)
learning_returns_all = learning_returns_demos + learning_returns_extrapolate
pred_returns_all = pred_returns_demos + pred_returns_extrapolate
print(pred_returns_all)
print(learning_returns_all)
plt.plot(learning_returns_extrapolate, [convert_range(p,max(pred_returns_all), min(pred_returns_all),max(learning_returns_all), min(learning_returns_all)) for p in pred_returns_extrapolate],'bo')
plt.plot(learning_returns_demos, [convert_range(p,max(pred_returns_all), min(pred_returns_all),max(learning_returns_all), min(learning_returns_all)) for p in pred_returns_demos],'ro')
plt.plot([min(0, min(learning_returns_all)-2),max(learning_returns_all) + buffer],[min(0, min(learning_returns_all)-2),max(learning_returns_all) + buffer],'g--')
plt.plot([min(0, min(learning_returns_all)-2),max(learning_returns_demos)],[min(0, min(learning_returns_all)-2),max(learning_returns_demos)],'k-', linewidth=2)
plt.axis([min(0, min(learning_returns_all)-2),max(learning_returns_all) + buffer,min(0, min(learning_returns_all)-2),max(learning_returns_all)+buffer])
plt.xlabel("Ground Truth Returns")
plt.ylabel("Predicted Returns (normalized)")
plt.tight_layout()
plt.savefig(save_fig_dir + "/" + env_name + "_gt_vs_pred_rewards.png")





#search for min and max predicted reward observations

min_reward = 100000
max_reward = -100000
cnt = 0
with torch.no_grad():
    for d in demonstrations:
        print(cnt)
        cnt += 1
        for i,s in enumerate(d[2:-1]):
            r = reward.cum_return(torch.from_numpy(np.array([s])).float().to(device))[0].item()
            if r < min_reward:
                min_reward = r
                min_frame = s
                min_frame_i = i+2
            elif r > max_reward:
                max_reward = r
                max_frame = s
                max_frame_i = i+2






def mask_coord(i,j,frames, mask_size, channel):
    #takes in i,j pixel and stacked frames to mask
    masked = frames.copy()
    masked[:,i:i+mask_size,j:j+mask_size,channel] = 0
    return masked

def gen_attention_maps(frames, mask_size):

    orig_frame = frames

    #okay so I want to vizualize what makes these better or worse.
    _,height,width,channels = orig_frame.shape

    #find reward without any masking once
    r_before = reward.cum_return(torch.from_numpy(np.array([orig_frame])).float().to(device))[0].item()
    heat_maps = []
    for c in range(4): #four stacked frame channels
        delta_heat = np.zeros((height, width))
        for i in range(height-mask_size):
            for j in range(width - mask_size):
                #get masked frames
                masked_ij = mask_coord(i,j,orig_frame, mask_size, c)
                r_after = r = reward.cum_return(torch.from_numpy(np.array([masked_ij])).float().to(device))[0].item()
                r_delta = abs(r_after - r_before)
                #save to heatmap
                delta_heat[i:i+mask_size, j:j+mask_size] += r_delta
        heat_maps.append(delta_heat)
    return heat_maps



#plot heatmap
mask_size = 3
delta_heat_max = gen_attention_maps(max_frame, mask_size)
delta_heat_min = gen_attention_maps(min_frame, mask_size)


# In[45]:


plt.figure(5)
for cnt in range(4):
    plt.subplot(1,4,cnt+1)
    plt.imshow(delta_heat_max[cnt],cmap='seismic', interpolation='nearest')
    plt.axis('off')
plt.tight_layout()
plt.savefig(save_fig_dir + "/" + env_name + "max_attention.png", bbox_inches='tight')


plt.figure(6)
print(max_frame_i)
print(max_reward)
for cnt in range(4):
    plt.subplot(1,4,cnt+1)
    plt.imshow(max_frame[0][:,:,cnt])
    plt.axis('off')
plt.tight_layout()
plt.savefig(save_fig_dir + "/" + env_name + "max_frames.png", bbox_inches='tight')


# In[46]:

plt.figure(7)
for cnt in range(4):
    plt.subplot(1,4,cnt+1)
    plt.imshow(delta_heat_min[cnt],cmap='seismic', interpolation='nearest')
    plt.axis('off')
plt.tight_layout()
plt.savefig(save_fig_dir + "/" + env_name + "min_attention.png", bbox_inches='tight')

print(min_frame_i)
print(min_reward)
plt.figure(8)
for cnt in range(4):
    plt.subplot(1,4,cnt+1)
    plt.imshow(min_frame[0][:,:,cnt])
    plt.axis('off')
plt.tight_layout()
plt.savefig(save_fig_dir + "/" + env_name + "min_frames.png", bbox_inches='tight')


#random frame heatmap
d_rand = np.random.randint(len(demonstrations))
f_rand = np.random.randint(len(demonstrations[d_rand]))
rand_frames = demonstrations[d_rand][f_rand]


# In[55]:

plt.figure(9)
for cnt in range(4):
    plt.subplot(1,4,cnt+1)
    plt.imshow(rand_frames[0][:,:,cnt])
    plt.axis('off')
plt.tight_layout()
plt.savefig(save_fig_dir + "/" + env_name + "random_frames.png", bbox_inches='tight')


delta_heat_rand = gen_attention_maps(rand_frames, mask_size)
plt.figure(10)
for cnt in range(4):
    plt.subplot(1,4,cnt+1)
    plt.imshow(delta_heat_rand[cnt],cmap='seismic', interpolation='nearest')
    plt.axis('off')
plt.tight_layout()
#plt.colorbar()
plt.savefig(save_fig_dir + "/" + env_name + "random_attention.png", bbox_inches='tight')

