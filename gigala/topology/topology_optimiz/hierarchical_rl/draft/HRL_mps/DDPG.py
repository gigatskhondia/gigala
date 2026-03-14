import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

device = torch.device("mps:0" if torch.mps.is_available() else "cpu")
print(device)


class extract_tensor(nn.Module):
    def forward(self,x):
        # Output shape (batch, features, hidden)
        tensor, _ = x
        # Reshape shape (batch, hidden)
        return tensor
    

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_bounds, offset, layer_dim, n_layers):
        super(Actor, self).__init__()
        
        # actor
        in_features = state_dim + state_dim
     
        # layers = [nn.LSTM(input_size=in_features, hidden_size=layer_dim, num_layers=n_layers, batch_first=True)]
        # layers.append(extract_tensor())
        
        layers=[]
        # out_features = layer_dim
        for i in range(n_layers):
            
            # Suggest the number of units in each layer
            out_features = layer_dim
            
            # layers.append(nn.Linear(out_features, out_features))
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.2))

            in_features = out_features

        # in_features = out_features
        
        layers.append(nn.Linear(in_features, action_dim))
        # layers.append(nn.Tanh())
        layers.append(nn.Softmax(dim=1))
        self.actor = nn.Sequential(*layers)
        
        # max value of actions
        self.action_bounds = action_bounds
        self.offset = offset
        
    def forward(self, state, goal):
        return (self.actor(torch.cat([state, goal], 1)) * self.action_bounds) + self.offset
        
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, H, layer_dim, n_layers):
        super(Critic, self).__init__()
        # UVFA critic
        layers = []
        
        
        in_features = state_dim + action_dim + state_dim
    
        for i in range(n_layers):
            
            # Suggest the number of units in each layer
            out_features = layer_dim
            
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.2))

            in_features = out_features
        
        layers.append(nn.Linear(in_features, 1))
        layers.append(nn.Sigmoid())
        self.critic = nn.Sequential(*layers)
        
        self.H = H
        
    def forward(self, state, action, goal):
        # rewards are in range [-H, 0]
        return -self.critic(torch.cat([state, action, goal], 1))* self.H 

    
class DDPG:
    def __init__(self, state_dim, action_dim, action_bounds, offset, lr, H, optimizer,layer_dim,n_layers):
        
        self.actor = Actor(state_dim, action_dim, action_bounds, offset, layer_dim,n_layers).type(torch.float32).to(device)
        self.actor_optimizer=getattr(optim, optimizer)(self.actor.parameters(), lr= lr)
        self.critic = Critic(state_dim, action_dim, H, layer_dim, n_layers).type(torch.float32).to(device)
        self.critic_optimizer=getattr(optim, optimizer)(self.critic.parameters(), lr= lr)
        
        self.mseLoss = torch.nn.MSELoss()
    
    def select_action(self, state, goal):
        state = torch.FloatTensor(state.reshape(1, -1)).type(torch.float32).to(device)
        goal = torch.FloatTensor(goal.reshape(1, -1)).type(torch.float32).to(device)
        return self.actor(state, goal).detach().cpu().data.numpy().flatten()
    
    def update(self, buffer, n_iter, batch_size,env):
    
           
        for i in range(n_iter):
            # Sample a batch of transitions from replay buffer:
            state, action, reward, next_state, goal, gamma, done = buffer.sample(batch_size)
            
            # convert np arrays into tensors
            state = torch.FloatTensor(state).type(torch.float32).to(device)
            action = torch.FloatTensor(action).type(torch.float32).to(device)
            reward = torch.FloatTensor(reward).reshape((batch_size,1)).type(torch.float32).to(device)
            next_state = torch.FloatTensor(next_state).type(torch.float32).to(device)
            goal = torch.FloatTensor(goal).type(torch.float32).to(device)
            gamma = torch.FloatTensor(gamma).reshape((batch_size,1)).type(torch.float32).to(device)
            done = torch.FloatTensor(done).reshape((batch_size,1)).type(torch.float32).to(device)
            
            # select next action
            next_action = self.actor(next_state, goal).detach()
            
            # Compute target Q-value:
            target_Q = self.critic(next_state, next_action, goal).detach()
            target_Q = reward + ((1-done) * gamma * target_Q)
            
               
            critic_loss = self.mseLoss(self.critic(state, action, goal), target_Q)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            
            # Compute actor loss:
            actor_loss = -self.critic(state, self.actor(state, goal), goal).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
                
                
    def save(self, directory, name):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, name))
        torch.save(self.critic.state_dict(), '%s/%s_crtic.pth' % (directory, name))
        
    def load(self, directory, name):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, name), map_location='cpu'))
        self.critic.load_state_dict(torch.load('%s/%s_crtic.pth' % (directory, name), map_location='cpu'))  
        