import matplotlib.pyplot as plt  
from utils import *
from utils_ import *
import gym
from gym import spaces
import random
import numpy as np
import autograd.numpy as anp  
from gym.utils import seeding


class Model:
    def __init__(self, x):
        self.flag_ = True
        self.n, self.m = x.shape
        self.actions_dic = {}
    
        k = 0
        for i in range(self.n):
            for j in range(self.m):
                self.actions_dic[k] = (i, j)
                k += 1
        
    def action_space_(self, action, x_cap):
        x, y = self.actions_dic[action]
        x_cap[x][y] = 1

    @staticmethod
    def draw(x_cap):
        plt.figure(dpi=50) 
        print('\nFinal Cantilever rl_beam design:')
        plt.imshow(x_cap)
        plt.show(block=False)
        plt.pause(3)
        plt.close('all')


class CantileverEnv(gym.Env):
    
    metadata = {"render.modes": ["human"],
                # 'video.frames_per_second' : 30
                }

    def __init__(self):
        super().__init__()

        self.rd = -1
        self.args = get_args(*mbb_beam(rd=self.rd))
        
        dim_cap = self.args.nelx*self.args.nely
        self.N_DISCRETE_ACTIONS = self.args.nelx*self.args.nely

        self.action_space = spaces.Box(low=0, high=1,
                                       shape=(self.N_DISCRETE_ACTIONS,), dtype=np.float64)

        self.observation_space = spaces.Box(low=np.array([-1e10 for x in range(dim_cap)]),
                                            high=np.array([1e10 for y in range(dim_cap)]),
                                            shape=(dim_cap,),
                                            dtype=np.float64)

        self.x = anp.ones((self.args.nely, self.args.nelx))*self.args.density 
    
        self.M = Model(self.x)
        
        self.reward = 0
        self.step_ = 0
        self.needs_reset = True
        self.layer_dim = 4
        self.n_layers = 2
        self.optimizer = 'Adam'
        self.seed()
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]   
    
    def step(self, action):

        self.args = get_args(*mbb_beam(rd=self.rd))

        act=np.argmax(action)

        self.M.action_space_(act, self.x)

        self.tmp, self.const = fast_stopt(self.args, self.x)
        self.step_+=1
        
        self.reward = (1/self.tmp)**0.5

        done = False
            
        if self.const > 0.68:
            done = True
                 
        if self.step_ > self.M.n*self.M.m:
            done = True

        if self.needs_reset:
            raise RuntimeError("Tried to step environment that needs reset")

        if done:
            self.needs_reset = True
            
        return self.x.reshape(self.x.shape[0]*self.x.shape[1]), self.reward, done, dict()
    
    def reset(self):
        
        if not self.M.flag_:
            self.rd = random.choice([0,2,-2])
        else:
            self.rd = -1
           
        self.x = anp.ones((self.args.nely, self.args.nelx))*self.args.density 

        self.reward = 0
        self.needs_reset = False
        self.step_ = 0

        return self.x.reshape(self.x.shape[0]*self.x.shape[1])

    def render(self, mode="human"):
        self.M.draw(self.x)    

    def close(self):
        pass
