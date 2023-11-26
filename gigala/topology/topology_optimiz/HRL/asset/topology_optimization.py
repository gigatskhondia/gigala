import matplotlib.pyplot as plt  
from utils import get_args, objective, mean_density, mbb_beam                               
import gym
from gym import spaces
import random
import numpy as np
import autograd, autograd.core, autograd.extend, autograd.tracer  
import autograd.numpy as anp  
import math
from gym.utils import seeding

class Model:
    def __init__(self, x):
        self.flag_ = True
        # self.flag_ = False
        self.n, self.m = x.shape
        self.actions_dic={} 
    
        k=0
        for i in range(self.n):
            for j in range(self.m):
                self.actions_dic[k]=(i,j)
                k+=1
        
    def action_space_(self, action, X):
        x,y=self.actions_dic[action]
        penalty=(X[x][y]==1)
        X[x][y]=1
        if penalty:
            return 1e-7
        return 0
        
    def draw(self,X):  
        plt.figure(dpi=50) 
        print('\nFinal Cantilever beam design:')
        plt.imshow(X) 
        plt.show(block=False)
        plt.pause(1.1)
        plt.close('all')
        
def fast_stopt(args, x):

    reshape = lambda x: x.reshape(args.nely, args.nelx)
    objective_fn = lambda x: objective(reshape(x), args)
    # constraint = lambda params: mean_density(reshape(params), args) - args.density
    constraint = lambda params: mean_density(reshape(params), args) 
    value = objective_fn(x)
    const = constraint(x)
    return value, const


class CantileverEnv(gym.Env):
    
    metadata = {"render.modes": ["human"],
                # 'video.frames_per_second' : 30
                }

    def __init__(self):
        super().__init__()
        
        
        self.rd=-1
        self.args = get_args(*mbb_beam(rd=self.rd))
        
        DIM=self.args.nelx*self.args.nely+(self.args.nelx+1)*(self.args.nely+1)*2
        self.N_DISCRETE_ACTIONS=self.args.nelx*self.args.nely
       
        # self.action_space = spaces.Discrete(self.N_DISCRETE_ACTIONS)
        
        self.action_space = spaces.Box(low=0, high=1,
                                       shape=(self.N_DISCRETE_ACTIONS,), dtype=np.float64)
       
        self.low_state=np.array([0, 0])
        self.high_state=np.array([1, 1e7])
           
        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state,
                                            dtype=np.float64)
        
 
        self.x = anp.ones((self.args.nely, self.args.nelx))*self.args.density 
    
        self.M=Model(self.x)
        
        self.reward=0
        self.step_=0
        self.needs_reset = True
        self.y=np.array([1e-4, 1e7])
        self.seed()
        self.layer_dim=4
        self.n_layers=2
        self.optimizer='Adam'
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]   
    
    def step(self, action):
        self.args = get_args(*mbb_beam(rd=self.rd))
        
        act=np.argmax(action)
            
        penalty=self.M.action_space_(act, self.x)
        
        self.tmp, self.const = fast_stopt(self.args, self.x)
        self.step_+=1
        
        self.reward = (1/self.tmp)**2
        # self.reward=(1/self.tmp)**0.5
        # self.reward += (1/self.tmp)**2
        # self.reward =(1/self.tmp)**2 - penalty
        # self.reward =-(self.tmp)**0.1*1e-4 + self.const*1e-2 if self.const<0.75 else -(self.tmp)**0.1*1e-4 - self.const*1e-2
                
        done=False
            
        if self.const>0.65:
#             self.reward-=1
            done=True
            
        # if self.const>0.65 and 100<self.tmp<300:
        #     self.reward+=1
        #     done=True  
                 
        if self.step_>self.M.n*self.M.m:
            done=True    
            
            
        if self.needs_reset:
            raise RuntimeError("Tried to step environment that needs reset")
            
        
        if done:
            self.needs_reset = True
            
        return  np.array([self.const,self.tmp]), self.reward, done, dict()
    
    def reset(self):
        
        if not self.M.flag_:
            self.rd=random.choice([0,2,-2])
        else:
            self.rd=-1
           
        self.x = anp.ones((self.args.nely, self.args.nelx))*self.args.density 

        self.reward=0
        self.needs_reset = False
        self.step_=0
        
        self.y=np.array([1e-4, 1e7])
        return self.y
       

    def render(self, mode="human"):
        self.M.draw(self.x)    

    def close(self):
        pass
        