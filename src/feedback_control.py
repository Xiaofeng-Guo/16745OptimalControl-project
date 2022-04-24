import argparse
import d4rl
import gym
import numpy as np
import matplotlib.pyplot as plt

from frankx import Affine, Kinematics, NullSpaceHandling
import numpy as np


def pid_control (env,target_state, init_action):

    action = init_action
    P = 1
    D = 0.01
    I = 0.1
    old_state_error = 0
    sum_state_error = 0
    while True:
        observation, reward, done, info = env.step(action)
        env.render()
        current_state = observation[0:9]
        state_error = target_state-current_state
        sum_state_error += state_error
        delta_state_error = state_error-old_state_error
        old_state_error = state_error
        action = P*state_error + D*delta_state_error + I*sum_state_error
        if (np.linalg.norm(current_state-target_state)<0.03):
            break
        print(action,np.linalg.norm(current_state-target_state))
    print("completed!")
    return action


def init_env(env):
    env.reset()
    action = np.zeros(9)
    observation, reward, done, info = env.step(action)
    init_joint = observation[0:7]
    init_state = observation[0:9]
    init_pos = Affine(Kinematics.forward(init_joint))
    init_pos = np.array(str(init_pos).strip('][').split(', '), np.float64)
    return init_pos,init_state


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

    init_pos,init_state = init_env(env)
    target_state = init_state
    init_action = np.zeros(9)
    pid_control (env,target_state, init_action)


