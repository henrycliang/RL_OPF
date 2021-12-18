import matplotlib.pyplot as plt
import numpy as np
import time
from math import *
import pandas as pd
import seaborn as sns

# cost_list=10e9*np.load('Cost_list_0.npy')
# timeLine=list(range(len(cost_list)))
# plt.figure(1)
# plot_RL=plt.plot(timeLine,cost_list,'r')
# plt.xlabel('Transition Step')
# plt.ylabel('The Cost Function')
# plt.xlim((-1,len(timeLine)))
# plt.grid(True)
# plt.legend((plot_RL[0],),('DDPG',))
# plt.savefig('The_cost_convergence.pdf')
# plt.show()


training_reward=np.load('training_reward.npy')
training_cost=np.load('training_cost.npy')
dfs=[]
# training_reward=training_reward[5:]
# training_cost=training_cost[5:]

timestep=np.arange(len(training_reward))
curve_reward=np.vstack((timestep,training_reward))
curve_reward=pd.DataFrame(curve_reward,index=["traing_episode","episode_reward_mean"])
curve_reward=curve_reward.T
# curve_reward["label"]="Episode Reward Mean"

plt.figure(dpi=300)
sns.set("notebook", "darkgrid")
ax = sns.lineplot(
    data=curve_reward,
    x="traing_episode",
    y="episode_reward_mean"
)
ax.set_title("TD3 training in OPF Environment")
ax.set_ylabel("Episode Reward Mean")
ax.set_xlabel("Sample Steps (Episode Steps)")
# plt.savefig('TD3 training reward.png')
plt.show()

curve_cost=np.vstack((timestep,training_cost))
curve_cost=pd.DataFrame(curve_cost,index=["traing_episode","episode_cost_mean"])
curve_cost=curve_cost.T

plt.figure(dpi=300)
sns.set("notebook", "darkgrid")
ax = sns.lineplot(
    data=curve_cost,
    x="traing_episode",
    y="episode_cost_mean"
)
ax.set_title("TD3 training in OPF Environment")
ax.set_ylabel("Cost Function Mean")
ax.set_xlabel("Sample Steps (Episode Steps)")
# plt.savefig('TD3 training cost.png')
plt.show()

# fig=plt.figure()
#
# ax1=fig.add_subplot(111)
# ax1.plot(timestep,training_reward,'green')
# ax1.set_ylabel("Episode Reward Mean")
# ax1.set_title("TD3 training in OPF Environment")
#
# ax2=ax1.twinx()
# ax2.plot(timestep,training_cost,'r')
# ax2.set_ylabel("Cost Function Mean")
# ax2.set_xlabel("Sample Steps (Episode Steps)")
# ax.legend(fontsize=8, loc="center right")
# plt.show()

# training_reward_DDPG=np.load('training_reward_DDPG_mean.npy')
# dfs_DDPG=[]
#
# timestep_DDPG=np.arange(len(training_reward_DDPG))
# curve_reward_DDPG=np.vstack((timestep_DDPG,training_reward_DDPG))
# curve_reward_DDPG=pd.DataFrame(curve_reward_DDPG,index=["traing_episode","episode_reward_mean"])
# curve_reward_DDPG=curve_reward_DDPG.T
#
# plt.figure(dpi=300)
# sns.set("notebook", "darkgrid")
# ax = sns.lineplot(
#     data=curve_reward_DDPG,
#     x="traing_episode",
#     y="episode_reward_mean"
# )
# ax.set_title("DDPG training in OPF Environment")
# ax.set_ylabel("Episode Reward Mean")
# ax.set_xlabel("Sample Steps (Episode Steps)")
# # plt.show()


plt.figure(dpi=300)
## Plot the scatter voltage
voltage=np.load('training_voltage_TD3.npy')
voltage_reference=np.load('training_voltage_reference_TD3.npy')
voltage_traditional=np.load('voltage_traditional.npy')
unit_voltage=voltage[-4]
timeLine=list(range(len(unit_voltage)))
plot_v1=plt.scatter(timeLine,unit_voltage,s=30,c='red')
plot_v2=plt.scatter(timeLine,voltage_reference,s=30,c='green')
plot_v3=plt.scatter(timeLine,voltage_traditional,s=30,c='orange')
plt.xlabel('Node Index')
plt.ylabel('Voltage Mag p.u.')
plt.title('Voltages: RL control v.s. Traditional control')
plt.grid(True)
plt.legend((plot_v1,plot_v2,plot_v3),('RL control','No Control','Tradi Control'))
# plt.savefig('Volatges under RL control.png')
plt.show()


plt.figure(dpi=300)
P_main=np.load('training_P_main_TD3.npy')
P_main_tradi=np.load('traditional_P.npy',allow_pickle=True)
P_main_episode=P_main[-1]
timeLine_P=list(range(len(P_main_episode)))

plot_v1=plt.scatter(timeLine_P,P_main_episode,s=30,c='red')
plot_v2=plt.scatter(timeLine_P,P_main_tradi,s=30,c='orange')
plt.xlabel('Node Index')
plt.ylabel('Active Power Injection (kW)')
plt.title('Active Power Injection: RL control v.s. Traditional control')
plt.grid(True)
plt.legend((plot_v1,plot_v2),('RL control','Tradi Control'))
plt.savefig('Active Power Injection.png')
plt.show()


# plot the Q_main
plt.figure(dpi=300)
Q_main=np.load('training_Q_main_TD3.npy')
Q_main_tradi=np.load('traditional_Q.npy')
Q_main_episode=Q_main[-1]

plot_v1=plt.scatter(timeLine_P,Q_main_episode,s=30,c='red')
plot_v2=plt.scatter(timeLine_P,Q_main_tradi,s=30,c='orange')
plt.xlabel('Node Index')
plt.ylabel('Reactive Power Injection (kVAR)')
plt.title('Reactive Power Injection: RL control v.s. Traditional control')
plt.grid(True)
plt.legend((plot_v1,plot_v2),('RL control','Tradi Control'))
plt.savefig('Reactive Power Injection')
plt.show()