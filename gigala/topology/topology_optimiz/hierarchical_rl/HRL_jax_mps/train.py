import torch
import gym
import numpy as np
from HAC import HAC
import asset


# device = torch.device("mps:0" if torch.mps.is_available() else "cpu")
device = torch.device("cpu")
print(device)

def train():
    #################### Hyperparameters ####################
    env_name ="T0-h-v1"

    save_episode = 100             # keep saving every n episodes
    max_episodes = 100_000         # max num of training episodes
    random_seed = 1
    render = False
    
    env = gym.make(env_name)
    env.layer_dim= 12
    env.n_layers= 14
    env.optimizer='RMSprop'
    state_dim = env.observation_space.shape[0]
    action_dim = env.N_DISCRETE_ACTIONS

    """
     Actions (both primitive and subgoal) are implemented as follows:
       action = ( network output (Tanh) * bounds ) + offset
       clip_high and clip_low bound the exploration noise
    """
    
    # primitive action bounds and offset
    action_bounds = env.action_space.high[0]
    # action_offset = np.array([0.5 for x in range(env.N_DISCRETE_ACTIONS)])
    action_offset = np.array([0.0 for x in range(env.N_DISCRETE_ACTIONS)])

    action_offset = torch.FloatTensor(action_offset.reshape(1, -1)).to(device)
    action_clip_low = np.array([0 for x in range(env.N_DISCRETE_ACTIONS)])
    action_clip_high = np.array([1 for x in range(env.N_DISCRETE_ACTIONS)])
    
    # state bounds and offset 
    # state_bounds_np = np.array([0.5, 0.5e7])
    # state_bounds_np = np.array([1, 1e7])
    # state_bounds = torch.FloatTensor(state_bounds_np.reshape(1, -1)).to(device)
    state_bounds = env.observation_space.high[0]
    # state_offset =  np.array([0.5, 0.5e7])
    state_offset = np.array([0 for x in range(env.N_DISCRETE_ACTIONS)])
    state_offset = torch.FloatTensor(state_offset.reshape(1, -1)).to(device)
    state_clip_low = np.array([0 for x in range(env.N_DISCRETE_ACTIONS)])
    state_clip_high = np.array([1 for x in range(env.N_DISCRETE_ACTIONS)])

    exploration_action_noise = np.array([0.5629988256824885])
    exploration_state_noise = np.array([0.1313567686964759])

    goal_ = [0.68, 30]
    goal_state = np.array(goal_ + [0] * (env.N_DISCRETE_ACTIONS - len(goal_)))
    threshold = [0.05, 3]

    # HAC parameters:
    k_level = 2               # num of levels in hierarchy
    H = 9       # time horizon to achieve subgoal
    lamda = 0.3453605248576358     # subgoal testing parameter
    
    # DDPG parameters:
    gamma = 0.9777965340075817 # discount factor for future rewards
    n_iter = 223     # update policy n_iter times in one DDPG update
    batch_size = 340  # num of transitions sampled from replay buffer
    lr = 0.04471490153909566
    
    # save trained models
    directory = "./preTrained/{}/{}level/".format(env_name, k_level) 
    filename = "HAC_{}".format(env_name)
    #########################################################

    if random_seed:
        print("Random Seed: {}".format(random_seed))
        env.seed(random_seed)
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
    
    # creating HAC agent and setting parameters
    agent = HAC(k_level, H, state_dim, action_dim, render, threshold, 
                action_bounds, action_offset, state_bounds, state_offset, lr, env.optimizer, env.layer_dim, env.n_layers)
    
    agent.set_parameters(lamda, gamma, action_clip_low, action_clip_high, 
                       state_clip_low, state_clip_high, exploration_action_noise, exploration_state_noise)
    
    # logging file:
    log_f = open("log.txt","w+")
    
    # training procedure 
    R=0
    for i_episode in range(1, max_episodes+1):
        agent.reward = 0
        agent.timestep = 0
       
        state = env.reset()
        # collecting experience in environment
        last_state, done = agent.run_HAC(env, k_level-1, state, goal_state, False)
        
        if agent.check_goal(last_state, goal_state, threshold, env):
            print("################ Solved! ################ ")
            name = filename + '_solved'
            agent.save(directory, name)
        
        # update all levels
        agent.update(n_iter, batch_size, env)
        
        # logging updates:
        log_f.write('{},{}\n'.format(i_episode, agent.reward))
        log_f.flush()
        R += agent.reward
        if i_episode % save_episode == 0:
            agent.save(directory, filename)
            print('SAVING ################# SAVING ################## SAVING:', R/save_episode)
            R = 0
        
        print("Episode: {}\t Reward: {}".format(i_episode, agent.reward))
        
    
if __name__ == '__main__':
    train()
 