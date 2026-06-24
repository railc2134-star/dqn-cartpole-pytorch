import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import time
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)
class Replaybuffer:
    def __init__(self,capacity):
        self.buffer=deque(maxlen=capacity)
    def push(self,state,action,reward,next_state,done):
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    def __len__(self):
        return len(self.buffer)
env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0]  
action_dim = env.action_space.n              
policy_net=DQN(state_dim,action_dim)
target_net=DQN(state_dim,action_dim)
target_net.load_state_dict(policy_net.state_dict())
buffer=Replaybuffer(10000)
optimizer =torch.optim.Adam(policy_net.parameters(),lr=0.00001 )
gamma=0.99
epsilon=1.0
epsilon_decay=0.995
epsilon_min=0.01
batch_size=64
target_update=50
loss=nn.MSELoss()
losss = None
steps_per_episode = 0
K=input("enter what action 1/loed or 2/train")
if K==2:
    for episode in range(10000):
        current_state,_=env.reset()
        done = False
        steps_per_episode = 0
        while not done:
            steps_per_episode += 1
            if epsilon > np.random.random():
                action = np.random.randint(0, 2)
            else:
                state_tensor = torch.FloatTensor(current_state)
                action = torch.argmax(policy_net(state_tensor)).item()
            next_state, reward, done, truncated, _ = env.step(action)
            done = done or truncated
            reward = reward if not done else -1
            buffer.push(current_state, action, reward, next_state, done)
            if len(buffer) >= batch_size:
                exp=buffer.sample(batch_size)
                states_b, actions_b, rewards_b, next_states_b, dones_b=zip(*exp)
                states=torch.FloatTensor(states_b)
                actions=torch.LongTensor(actions_b)
                rewards=torch.FloatTensor(rewards_b)
                next_states=torch.FloatTensor(next_states_b)
                dones=torch.FloatTensor(dones_b)
                target=rewards+target_net(next_states).max(1)[0].detach()*gamma*(1-dones)
                predictions=policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
                optimizer.zero_grad()
                losss=loss(predictions,target)
                losss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                optimizer.step()
            current_state=next_state
        if episode %100==0:
            print(f"episode {episode} || loss {losss} || steps {steps_per_episode}") 
        epsilon=max(epsilon*epsilon_decay,epsilon_min)
        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
        if steps_per_episode==500:
            break
    torch.save(policy_net.state_dict(), 'cartpole_dqn.pth')
else:
    policy_net = DQN(state_dim, action_dim)
    policy_net.load_state_dict(torch.load('cartpole_dqn.pth'))
    policy_net.eval()
    env_render = gym.make("CartPole-v1", render_mode="human")
    state, _ = env_render.reset()
    done = False
    while not done:
        state_tensor = torch.FloatTensor(state)
        action = torch.argmax(policy_net(state_tensor)).item()
        state, reward, done, truncated, _ = env_render.step(action)
        done = done or truncated
        time.sleep(0.05)
    env_render.close()