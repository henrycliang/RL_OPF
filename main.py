import gym
import networkx as nx
import opendssdirect as dss
import numpy as np
import copy
import random
from collections import deque
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from TD3 import TD3

path='E:\CUHK Course Project\OPF_DeepRL\ieee_feeder\ieee37Lines.dss'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
initial_factor=0.3
project_factor=1.

algo_parser='TD3'


def Volatge(YVol, NodeNum):
    Vmag = [0] * NodeNum
    VSquare = [0] * NodeNum
    for i in range(NodeNum):
        VSquare[i] = YVol[2 * i] ** 2 + YVol[2 * i + 1] ** 2
        Vmag[i] = math.sqrt(VSquare[i])
    return VSquare, Vmag

NodeToLine={}
G=nx.DiGraph()
with open(path) as file:
    lines=file.readlines()
    # print(lines)
    # i=0
    for line in lines:
        if line[0]=='N':
            # i=i+1
            # print(line)
            temp=line.split(' ')
            temp=[item for item in temp if item!='']
            index=temp[3].find('.')
            index2=temp[4].find('.')
            line=temp[1].upper()
            NodeToLine[temp[4][5:index2].upper()]=line
            # lineType=temp[5][9:]
            # print(temp[3][5:index])
            # print(temp[4][5:index2])
            # we only consider the single-phase cases here
            G.add_edge(temp[3][5:index].upper(),temp[4][5:index2].upper(),Num_phase=1,z=np.zeros((1),dtype=complex))
file.close()

dss.run_command("Compile 'E:\CUHK Course Project\OPF_DeepRL\ieee_feeder\ieee37.dss'" )
circuit=dss.Circuit
AllNodeNames=circuit.YNodeOrder()
AllNodeNames=[str(node) for node in AllNodeNames]

node_number=len(AllNodeNames)
# Compute the initial voltage
VYol=circuit.YNodeVArray()
_,VMagReference=Volatge(VYol,node_number)

NodeNameToIndex={}
IndexToNodeName={}
i=0
for name in AllNodeNames:
    NodeNameToIndex[name[:-2]]=i
    IndexToNodeName[i] = name[:-2]
    i+=1

###### Compute Line z ##
Y_path='E:\CUHK Course Project\OPF_DeepRL\ieee_feeder\ieee37_SystemY.txt'
ZMatrix=np.zeros(shape=(node_number+1,node_number+1),dtype=complex)
with open(Y_path) as file:
    lines=file.readlines()
    for line in lines:
        if (line[0] == '['):
            index1 = line.find(',')
            row = int(line[1:index1])
            index2 = line.find(']')
            col = int(line[index1 + 1:index2])
            index1 = line.find('=')
            index2 = line.find('+')
            real = float(line[index1 + 1:index2])
            index1 = line.find('j')
            imag = float(line[index1 + 1:])
            Y_complex = complex(real, imag)
            ZMatrix[row - 1, col - 1] = Y_complex
            ZMatrix[col - 1, row - 1] = Y_complex
file.close()

# Update the line z matrix
for edge in G.edges:
    row=NodeNameToIndex[edge[0]]
    col=NodeNameToIndex[edge[1]]
    temp=[[ZMatrix[row][col]]]
    temp=-np.linalg.pinv(temp)
    temp=temp[0,0]
    G[edge[0]][edge[1]]['z']=temp
    # temp2='{:5f}'.format(temp)


loadFile='E:\CUHK Course Project\OPF_DeepRL\ieee_feeder\ieee37.dss'
Loads_main={}
LoadToNode={}

def whichCluster(LoadName,NodeName):
    if NodeName in list(G):
        Loads_main[LoadName]=NodeName

with open(loadFile) as file:
    lines=file.readlines()
    for line in lines:
        if line[0] == 'N' and line[5]=='o':
            temp=line.split(' ')
            temp=[item for item in temp if item!='']
            # print(temp)
            LoadName=temp[1][5:]
            NodeName=temp[2][5:]
            # print(loadName,NodeName)
            LoadToNode[LoadName]=NodeName
            whichCluster(LoadName,NodeName)

def BoundVol(AllNodeNames,circuit):
    length=len(AllNodeNames)
    VUpper=[0]*length
    VLower=[0]*length
    VBase=[0]*length
    for i in range(length):
        circuit.SetActiveBus(AllNodeNames[i])
        base=dss.Bus.kVBase()*1000
        VUpper[i]=base*1.05
        VLower[i]=base*0.95
        VBase[i]=base
    return VUpper,VLower,VBase

LoadNameToIndex={} # Correspond the LoadName to the Index in action
IndexToLoadName={}

def reset_initial():
    dss_path = 'E:\CUHK Course Project\OPF_DeepRL\ieee_feeder\ieee37.dss'
    with open(dss_path) as file:
        lines = file.readlines()
        for line in lines:
            if line[0] == 'N' and line[5] == 'o':
                temp = line.split(' ')
                temp = [item for item in temp if item != '']
                loadname = temp[1][5:]
                # print(loadname)
                if loadname in Loads_main.keys():
                    loadKW = '? Load.' + loadname.upper() + '.kw'
                    loadKVar = '? Load.' + loadname.upper() + '.kvar'
                    p = dss.run_command(loadKW)
                    q = dss.run_command(loadKVar)
                    p_new = float(p) * initial_factor
                    q_new = float(q) * initial_factor
                    # q_ref+=q_new
                    command = "edit load." + loadname + " kw=" + str(p_new) + " kvar=" + str(q_new)
                    print(command)
                    dss.run_command(command)
                else:
                    print('Load {} is not controllable'.format(loadname))
    file.close()
    dss.run_command("Solve")
reset_initial()
dss.run_command("show voltages LN Nodes")

def project(num_p,num_q,ini_p,ini_q,coff=project_factor):
    if (ini_p>0):
        p=min(max(num_p,0.1*ini_p),coff*ini_p)
        q=min(max(num_q,0.1*ini_q),coff*ini_q)
    else:
        p=max(min(num_p,0.1*ini_p),coff*ini_p)
        q=max(min(num_q,0.1*ini_q),coff*ini_q)
    return p,q

def getInitialPV2(Loads_Set):
    P_initial={}
    Q_initial={}
    i=0
    for LoadName in Loads_Set.keys():
        loadKW = '? Load.' + LoadName.upper() + '.kw'
        loadKVar = '? Load.' + LoadName.upper() + '.kvar'
        p = dss.run_command(loadKW)
        q = dss.run_command(loadKVar)
        LoadNameToIndex[LoadName]=i
        IndexToLoadName[i]=LoadName
        i+=1
        P_initial[LoadName] = float(p) * 1000
        Q_initial[LoadName] = float(q) * 1000
    return P_initial, Q_initial

def getInitialPV(Loads_Set):
    P_initial={}
    Q_initial={}
    for LoadName in Loads_Set.keys():
        loadKW='? Load.' + LoadName.upper() + '.kw'
        loadKVar='? Load.' + LoadName.upper() + '.kvar'
        p=dss.run_command(loadKW)
        q=dss.run_command(loadKVar)
        P_initial[LoadName]=float(p)*1000*1./initial_factor
        Q_initial[LoadName]=float(q)*1000*1./initial_factor
    return P_initial,Q_initial


P_main,Q_main=getInitialPV2(Loads_main)
P_main_reset,Q_main_reset=getInitialPV2(Loads_main)
P_main_restore,Q_main_restore=getInitialPV(Loads_main)
VUpper,VLower,VBase=BoundVol(AllNodeNames,circuit)

print('first step')
def unpack_state(state):
    P_main_=state['P_main']
    Q_main_=state['Q_main']
    PQ_main_array=[]
    for key,value in P_main_.items():
        PQ_main_array.append(value)
    for key,value in Q_main_.items():
        PQ_main_array.append(value)
    return np.array(PQ_main_array)


# save the reference voltage
unit_voltage_reference=[]
for node in range(node_number):
    unit_voltage_reference.append(VMagReference[node]/VBase[node])
np.save(f'E:\CUHK Course Project\OPF_DeepRL/training_voltage_reference_{algo_parser}.npy',unit_voltage_reference)
######################################################################################################

def to_tensor(x):
    if isinstance(x,np.ndarray):
        x=torch.from_numpy(x).type(torch.float32)
    assert isinstance(x,torch.Tensor)
    if x.dim()==3 or x.dim()==1:
        x=x.unsqueeze(0)
    assert x.dim()==2 or x.dim()==4, x.shape
    return x

# class ReplayMemory:
#     def __init__(self,capacity):
#         self.memory=deque(maxlen=capacity)
#
#     def push(self,transition):
#         self.memory.append(transition)
#     def sample(self,batch_size):
#         return random.sample(self.memory,batch_size)
#     def __len__(self):
#         return len(self.memory)


class PowerEnv(gym.Env):
    def __init__(self,Loads_main,P_main,Q_main,P_main_reset,Q_main_reset,node_number):
        super(PowerEnv,self).__init__()
        self.P_main=P_main
        self.Q_main=Q_main
        self.P_main_reset=P_main_reset
        self.Q_main_reset=Q_main_reset
        self.P_main_restore=P_main_restore
        self.Q_main_restore=Q_main_restore
        self.Loads_main=Loads_main
        self.node_number=node_number
        self.max_action=1.
        self.action_space=gym.spaces.Box(np.float32(-self.max_action),np.float32(self.max_action),shape=(len(P_main)+len(Q_main),))
        self.observation_space=gym.spaces.Box(np.float32(-1),np.float32(1),shape=(25*2,))
        # print(self.action_space)

    def Volatge(self,YVol,NodeNum):
        Vmag=[0]*NodeNum
        VSquare=[0]*NodeNum
        for i in range(NodeNum):
            VSquare[i]=YVol[2*i]**2+YVol[2*i+1]**2
            Vmag[i]=math.sqrt(VSquare[i])
        return VSquare,Vmag


    def reward(self,VSquare,cost):
        Reward_vol=0
        reward_injection=0
        Reward_cost = 178 - 1e-8 * cost  #原来画曲线时这个设置是178
        violation_bool=False
        # Reward_cost = 10
        for i in range(len(VSquare)):
            if VSquare[i]-VUpper[i]**2>0:
                # Reward_vol+=-1e-3*(VSquare[i]-VUpper[i]**2)
                # Reward_vol =-1e2
                violation_bool=True
                # Reward_cost = 0
            elif VLower[i]**2-VSquare[i]>0:
                # Reward_vol+=-1e-3*(VLower[i]**2-VSquare[i])
                # Reward_vol = -1e2
                violation_bool = True
                # print('voltage violation')
                # Reward_cost = 0
            else:
                Reward_vol+=0.
        total_reward = Reward_vol + Reward_cost
        # total_reward=Reward_cost
        for LoadName, NodeName in Loads_main.items():
            if self.P_main[LoadName]>self.P_main_reset[LoadName]*1./initial_factor:
                # reward_injection =-1e2
                violation_bool = True
                # print('injection violation')
            if self.Q_main[LoadName]>self.Q_main_reset[LoadName]*1./initial_factor:
                # reward_injection =-1e2
                violation_bool = True
                # print('injection violation')
            # if self.P_main[LoadName] < self.P_main_reset[LoadName] * 1. / initial_factor*0.1:
            #     reward_injection+=-1e-3
            #     violation_bool = True
            # if self.Q_main[LoadName] < self.Q_main_reset[LoadName] * 1. / initial_factor*0.1:
            #     reward_injection+=-1e-3
            #     violation_bool = True

        # if Reward_cost ==0:
        #     print('Voltage violation Occurs!!!!!!!!!!!!!!!!!!!!!')

        # if reward_injection<0:
        #     total_reward=reward_injection
        total_reward+=reward_injection
        return np.array(total_reward),violation_bool

    def reset(self):
        self.P_main=self.P_main_reset.copy()  #之所以用copy是因为不用copy时，直接的等号self.P_main变化会导致self.P_main_reset也变化
        self.Q_main=self.Q_main_reset.copy()
        # reset_initial()
        # YVol=circuit.YNodeVArray()
        # VSquare=self.Volatge(YVol,self.node_number)
        obs_dict=dict(P_main=self.P_main,
                      Q_main=self.Q_main)
        obs=unpack_state(obs_dict)
        # print(obs)
        return obs

    def step(self,action):
        VSquare,VMag,cost=self.next_PQ(action)
        reward,violation_bool=self.reward(VSquare,cost)
        obs_dict=dict(P_main=self.P_main,
                         Q_main=self.Q_main,
                         VSquare=VSquare)
        observation=unpack_state(obs_dict)
        if violation_bool or reward>0:
            done=True
        else:
            done=False

        P_main_=self.P_main.copy()
        Q_main_=self.Q_main.copy()
        # 这里将voltage也当作info返回，用来画图
        info={'cost':cost/10e9,
              'VMag':VMag,
              'P_main':P_main_,
              'Q_main':Q_main_}

        return observation,reward,done,info


    def next_PQ(self,action):
        # Here the action is equal to the gradient,
        # unormalize the step
        action=action*200
        cost=0
        P_action=action[:25]
        Q_action=action[25:]
        for LoadName,NodeName in Loads_main.items():
            self.P_main[LoadName]=self.P_main[LoadName]-P_action[LoadNameToIndex[LoadName]]
            self.Q_main[LoadName]=self.Q_main[LoadName]-Q_action[LoadNameToIndex[LoadName]]

            # self.P_main[LoadName],self.Q_main[LoadName]=project(self.P_main[LoadName],self.Q_main[LoadName],self.P_main_restore[LoadName],self.Q_main_restore[LoadName])

            command="edit load."+LoadName+" kw="+str(self.P_main[LoadName]/1000)+" kvar="+str(self.Q_main[LoadName]/1000)
            dss.run_command(command)

            cost+=(self.P_main[LoadName]-self.P_main_restore[LoadName])**2+(self.Q_main[LoadName]-self.Q_main_restore[LoadName])**2
        dss.run_command("Solve")
        # We assume that the self.next_PQ function return the next state by taking the action (the new_P and new_Q),
        YVol=circuit.YNodeVArray()
        VSquare,VMag=self.Volatge(YVol,node_number)
        cost=cost
        return VSquare,VMag,cost


class Actor(nn.Module):
    def __init__(self,state_dim,num_actions,max_action):
        super(Actor,self).__init__()
        self.l1=nn.Linear(state_dim,512)
        self.l2=nn.Linear(512,512)
        self.l3=nn.Linear(512,num_actions)
        self.max_action=max_action

    def forward(self, obs):
        a=F.relu(self.l1(obs))
        a=F.relu(self.l2(a))
        a=self.l3(a)
        a=self.max_action*torch.tanh(a)
        return a

class Critic(nn.Module):
    def __init__(self,state_dim,action_dim):
        super(Critic,self).__init__()

        self.l1=nn.Linear(state_dim+action_dim,512)
        self.l2=nn.Linear(512,512)
        self.l3=nn.Linear(512,1)

    def forward(self,state,action):
        q=F.relu(self.l1(torch.cat([state,action], 1)))
        q=F.relu(self.l2(q))
        return self.l3(q)



class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )


class DDPG(object):
    def __init__(self, state_dim, action_dim, max_action, discount=0.99, tau=0.001):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), weight_decay=1e-2)

        self.discount = discount
        self.tau = tau

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=64):
        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        # Compute the target Q value
        target_Q = self.critic_target(next_state, self.actor_target(next_state))
        target_Q = reward + (not_done * self.discount * target_Q).detach()

        # Get current Q estimate
        current_Q = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute actor loss
        actor_loss = -self.critic(state, self.actor(state)).mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)

# test the built environment
# env=PowerEnv(Loads_main,P_main,Q_main,P_main_reset,Q_main_reset,node_number)
# # observation=env.reset()
# for t in range(100):
#     print('reset, ',t)
#     env.reset()
#     action=env.action_space.sample()
#     print(env.action_space.shape)
#     obs,reward,done,_=env.step(action)
#     print(obs)
#     print(reward)

def state_normalize(obs):
    new_obs=obs/10000
    return new_obs

def state_unormalize(obs):
    new_obs=obs*10000
    return new_obs
### Here to begin the RL algorithm
# run the train
config=dict(
    max_timestep=1000000,
    batch_size=32,
    learning_rate=0.01,
    clip_norm=10.0,
    memory_size=50000,
    learn_start=1e4,
    eps=0.1,
    max_episode_length=2000,
    gamma=0.99,
    eval_frequency=10,
    save_freq=50000
)

env=PowerEnv(Loads_main,P_main,Q_main,P_main_reset,Q_main_reset,node_number)

state_dim=env.observation_space.shape[0]
action_dim=env.action_space.shape[0]
max_action=float(env.action_space.high[0])

# policy=DDPG(state_dim, action_dim, max_action)
policy=TD3(state_dim,action_dim,max_action)


replay_buffer=ReplayBuffer(state_dim,action_dim)

state,done=env.reset(),False
state=state_normalize(state)  #normalize the state to input to TD3

episode_reward=0
episode_timesteps=0
episode_num=0
reward_recorder=deque(maxlen=100)
cost_recorder=deque(maxlen=100)
reward_mean_list=[]
cost_mean_list=[]

reward_max_list=[]
cost_max_list=[]

reward_min_list=[]
cost_min_list=[]


unit_voltage_list=[]
P_main_list=[]
Q_main_list=[]
for t in range(config["max_timestep"]):
    # print('reset', t)
    # trainer.train(i) #这李不用Trainer里面的train方法了

    episode_timesteps+=1
    if t<config['learn_start']:
        action=env.action_space.sample()
    else:
        action=policy.select_action(np.array(state))\
               +np.random.normal(0,max_action*0.1,size=action_dim
                                 ).clip(-max_action,max_action)


    # Perform action

    next_state,reward,done,info=env.step(action)
    next_state=state_normalize(next_state)
    if episode_timesteps>=config['max_episode_length']:
        done=True

    done_bool=float(done)

    replay_buffer.add(state,action,next_state,reward,done_bool)

    state=next_state
    episode_reward+=reward

    if t>config['learn_start']:
        # print('start learning')
        policy.train(replay_buffer,batch_size=256)

    # Record the training process
    if done and t > config['learn_start']:
        print(f"Record reward at timestep: {t + 1}")
        reward_recorder.append(reward)
        cost = info['cost'] * 10e9
        cost_recorder.append(cost)
        reward_mean = np.mean(reward_recorder)
        cost_mean = np.mean(cost_recorder)

        reward_max=np.max(reward_recorder)
        cost_max=np.max(cost_recorder)

        reward_min=np.min(reward_recorder)
        cost_min=np.min(cost_recorder)

        # append to the list
        reward_mean_list.append(reward_mean)
        cost_mean_list.append(cost_mean)

        reward_max_list.append(reward_max)
        cost_max_list.append(cost_max)

        reward_min_list.append(reward_min)
        cost_min_list.append(cost_min)

        # Compute the voltage
        VMag_episode=info['VMag']
        unit_VMag_episode=[]
        for node in range(len(VMag_episode)):
            unit_VMag_episode.append(VMag_episode[node]/VBase[node])
        unit_voltage_list.append(unit_VMag_episode)

        # Compute the P_main and Q_main
        P_main_=info['P_main']
        P_main_episode=[]
        for key,value in P_main_.items():
            P_main_episode.append(value/1000.)
        P_main_list.append(P_main_episode)

        Q_main_=info['Q_main']
        Q_main_episode=[]
        for key,value in Q_main_.items():
            Q_main_episode.append(value/1000.)
        Q_main_list.append(Q_main_episode)


        # save the training progress
        np.save(f'E:\CUHK Course Project\OPF_DeepRL/training_reward_{algo_parser}_mean.npy', reward_mean_list)
        np.save(f'E:\CUHK Course Project\OPF_DeepRL/training_cost_{algo_parser}_mean.npy', cost_mean_list)
        np.save(f'E:\CUHK Course Project\OPF_DeepRL/training_reward_{algo_parser}_max.npy',reward_max_list)
        np.save(f'E:\CUHK Course Project\OPF_DeepRL/training_cost_{algo_parser}_max.npy',cost_max_list)
        np.save(f'E:\CUHK Course Project\OPF_DeepRL/training_reward_{algo_parser}_min.npy', reward_min_list)
        np.save(f'E:\CUHK Course Project\OPF_DeepRL/training_cost_{algo_parser}_min.npy', cost_min_list)

        # Save the voltage
        np.save(f'E:\CUHK Course Project\OPF_DeepRL/training_voltage_{algo_parser}.npy',unit_voltage_list)

        # Save the P_main and Q_main
        np.save(f'E:\CUHK Course Project\OPF_DeepRL/training_P_main_{algo_parser}.npy',P_main_list)
        np.save(f'E:\CUHK Course Project\OPF_DeepRL/training_Q_main_{algo_parser}.npy',Q_main_list)

    if done:
        print(
            f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Final Reward: {reward:.3f}  Cost: {info['cost']*100}")
        # Reset environment
        # print(action)
        state, done = env.reset(), False
        # print('reset state',state)
        episode_reward = 0
        episode_timesteps = 0
        episode_num += 1

    if (t + 1) % config['save_freq'] == 0:
        print(f"Save model at: {t + 1} at:/models/default")
        # policy.save(f"E:\CUHK Course Project\OPF_DeepRL\models/default_{t+1}")


