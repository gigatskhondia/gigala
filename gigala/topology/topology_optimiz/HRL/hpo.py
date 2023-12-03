import torch
import gym
import numpy as np
from HAC import HAC
import optuna
from asset.topology_optimization import CantileverEnv


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Check for HPO:
# https://towardsdatascience.com/hyperparameter-tuning-of-neural-networks-with-optuna-and-pytorch-22e179efc837

def train(params):
    
    #################### Hyperparameters ####################
    env_name ="T0-h-v1"

    save_episode = 20               # keep saving every n episodes
    # max_episodes = params['max_episodes']        # max num of training episodes
    max_episodes = 1_000 
    # random_seed = params['random_seed']
    random_seed=False
    render = False
    
    env = gym.make(env_name)
    env.layer_dim=params['layer_dim']
    # env.layer_dim=3
    env.n_layers=params['n_layers']
    # env.n_layers=6
    env.optimizer=params['optimizer']
    # env.optimizer='SGD'

    state_dim = env.observation_space.shape[0]
    action_dim = env.N_DISCRETE_ACTIONS
    
    """
     Actions (both primitive and subgoal) are implemented as follows:
       action = ( network output (Tanh) * bounds ) + offset
       clip_high and clip_low bound the exploration noise
    """
    
    # primitive action bounds and offset
    action_bounds = env.action_space.high[0]
    action_offset = np.array([0.5 for x in range(env.N_DISCRETE_ACTIONS)])
    action_offset = torch.FloatTensor(action_offset.reshape(1, -1)).to(device)
    action_clip_low = np.array([0 for x in range(env.N_DISCRETE_ACTIONS)])
    action_clip_high = np.array([1 for x in range(env.N_DISCRETE_ACTIONS)])
    
    # state bounds and offset 
    state_bounds_np = np.array([0.5, 0.5e7])
    state_bounds = torch.FloatTensor(state_bounds_np.reshape(1, -1)).to(device)
    state_offset =  np.array([0.5, 0.5e7])
    state_offset = torch.FloatTensor(state_offset.reshape(1, -1)).to(device)
    state_clip_low = np.array([0, 0])
    state_clip_high = np.array([1, 1e7])

    exploration_action_noise = np.array([params['action_noise']])        
    exploration_state_noise = np.array([params['state_noise_1'], params['state_noise_2']])

    goal_state=np.array([0.68, 20])
    threshold=[0.05, 5]
    
    # HAC parameters:
    k_level = 2               # num of levels in hierarchy
    H = params['H']               # time horizon to achieve subgoal
    # H = 11
    lamda = params['lamda']               # subgoal testing parameter
    # lamda = 0.9453109199655714
    
    # DDPG parameters:
    gamma = params['gamma']                # discount factor for future rewards
    # gamma = 0.992256316386673 
    n_iter = params['n_iter']              # update policy n_iter times in one DDPG update
    # n_iter = 186 
    batch_size = params['batch_size']         # num of transitions sampled from replay buffer
    # batch_size =256
    lr = params['lr']
    # lr= 0.0032967527995782626
    
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
    
  
    # training procedure 
    my_res=[]
    for i_episode in range(1, max_episodes+1):
        agent.reward = 0
        agent.timestep = 0
       
        state = env.reset()
        # collecting experience in environment
        last_state, done = agent.run_HAC(env, k_level-1, state, goal_state, False)
        
        agent.update(n_iter, batch_size, env)
        
        my_res.append(agent.reward)
            
    return np.mean(my_res)

def objective(trial):

    params = {
            #   'max_episodes':trial.suggest_int("max_episodes", 1000, 1500),
            #    'random_seed': trial.suggest_int("random_seed", 0, 5),
               'layer_dim':trial.suggest_int("layer_dim", 2, 16),
              'n_layers':trial.suggest_int("n_layers", 2, 16),
              'optimizer': trial.suggest_categorical("optimizer", ["Adam", 
                                                                   "RMSprop",
                                                                   "SGD"
                                                                   ]),
              'action_noise':trial.suggest_loguniform('action_noise', 0.01, 1),
              'state_noise_1': trial.suggest_loguniform('state_noise_1', 0.01, 1),
              'state_noise_2': trial.suggest_loguniform('state_noise_2', 1000, 1e7),
              'H':  trial.suggest_int("H", 3, 16),
              'lamda': trial.suggest_uniform('lamda', 0.3, 1),
              'gamma': trial.suggest_uniform('gamma', 0.95, 0.999),
              'n_iter': trial.suggest_int('n_iter', 50, 350),
               'batch_size': trial.suggest_int('batch_size', 50, 350),
                'lr': trial.suggest_loguniform('lr', 1e-5, 1)

              }
    
   
    
    rev = train(params)

    return rev



study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
study.optimize(objective, n_trials=100)


best_trial = study.best_trial

print()

for key, value in best_trial.params.items():
    print("{}: {}".format(key, value))
    