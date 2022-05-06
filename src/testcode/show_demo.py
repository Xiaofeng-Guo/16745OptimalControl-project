import numpy as np

import argparse
import copy

import d4rl
import gym
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import numpy as np
import kinematics
import time
import pybullet as p
import pybullet_data
def pid_control (env,target_states, init_action):

    action = init_action
    N = len(target_states)
    N = 100
    P = 1
    D = 0.01
    I = 0.1
    old_state_error = 0
    sum_state_error = 0
    time_step = 0
    observation, reward, done, info = env.step(action)
    env.render()
    current_state = observation[0:9]
    error_threthold = 0.04
    X = [np.zeros(9) for k in range(0, N*3+100)]
    Xref = [np.zeros(9) for k in range(0, N * 3 + 100)]

    while True:

        target_state = target_states[min(time_step//2,N-1),:9]
        Xref[time_step] = target_state
        state_error = target_state-current_state
        sum_state_error += state_error
        delta_state_error = state_error-old_state_error
        # if np.linalg.norm(delta_state_error)<0.01:
        #     print("using delta state error\n")
        #     break
        old_state_error = state_error
        action = P*state_error + D*delta_state_error + I*sum_state_error
        # action[7] = 1
        # action[8] = 1
        if target_state[7] <0.01:
            action[7] = -1
        if target_state[8]<0.01:
            action[8]= -1
        observation, reward, done, info = env.step(action)
        # env.render()
        current_state = observation[0:9]
        X[time_step] = observation[:9]
        time_step += 1
        if (time_step//2 > (N-1)  and (np.linalg.norm(current_state-target_state)<error_threthold)):
            break
        if time_step%100 ==0:
            print(target_state,'\n',current_state,'\n',action,'\n',np.linalg.norm(current_state-target_state),np.linalg.norm(delta_state_error),'\n')
            error_threthold += 0.01

    X = np.asarray(X)
    Xref = np.asarray(Xref)
    #
    # for i in range(0,7):
    #     plt.plot(X[:2*N, i],label = 'tracked tajectory')
    #     plt.plot(Xref[:2*N,i],label = 'reference tajectory')
    #     plt.legend()
    #     plt.xlabel("time step")
    #     plt.ylabel("joint "+str(i))
    #     plt.show()

    return action


def direct_pid_control (env,target_state, init_action):

    action = init_action
    print(target_state)
    P = 1
    D = 0.01
    I = 0.1
    old_state_error = 0
    sum_state_error = 0
    time_step = 0
    observation, reward, done, info = env.step(action)
    env.render()
    current_state = observation[0:9]
    error_threthold = 0.03
    xref = []
    uref = []
    while True:
        state_error = target_state-current_state
        sum_state_error += state_error
        delta_state_error = state_error-old_state_error
        # if np.linalg.norm(delta_state_error)<0.01:
        #     print("using delta state error\n")
        #     break
        old_state_error = state_error
        action = P*state_error + D*delta_state_error + I*sum_state_error
        # action[7] = 1
        # action[8] = 1
        if target_state[7] <0.01:
            action[7] = -1
        if target_state[8]<0.01:
            action[8]= -1
        observation, reward, done, info = env.step(action)
        env.render()
        current_state = observation[0:9]
        xref.append(observation[0:18])
        uref.append(action)
        time_step += 1
        if (np.linalg.norm(current_state-target_state)<error_threthold):
            break
        if time_step%100 ==0:
            print(target_state,'\n',current_state,'\n',action,'\n',np.linalg.norm(current_state-target_state),np.linalg.norm(delta_state_error),'\n')
            error_threthold += 0.01
    return xref,uref



def init_env(env):
    env.reset()
    action = np.zeros(9)
    observation, reward, done, info = env.step(action)
    init_joint = observation[0:7]
    init_state = observation[0:9]

    return init_state




def calculate_uref():
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='kitchen-complete-v0')
    args = parser.parse_args()

    env = gym.make(args.env_name)
    dataset = env.get_dataset()
    np.set_printoptions(threshold=np.inf)

    rewards = dataset['rewards']
    actions = dataset['actions']
    observations = dataset['observations']

    kine = kinematics.Kinematics()
    init_state = init_env(env)
    init_joint = init_state[0:7]

    templete_obs = np.load("data/trial3/obs.npy")
    print(templete_obs.shape)
    templete_obs[20:29,7] = 0.04
    templete_obs[20:29,8] = 0.04
    templete_obs[29:45,7] = 0.002
    templete_obs[29:45, 8] = 0.002
    templete_obs[45:60,7] = 0.04
    templete_obs[45:60,8] = 0.04
    templete_obs[65:75, 7] = 0.002
    templete_obs[65:75, 8] = 0.002

    init_action = np.zeros(9)

    for i in range(0,100):
        env.render()

    N = len(templete_obs)
    pid_control(env,templete_obs,init_action)

    while True:
        env.render()
