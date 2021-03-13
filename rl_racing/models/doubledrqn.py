import math, random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F

import matplotlib.pyplot as plt

import numpy as np
import time
from collections import deque
from tqdm import trange
import cv2
from models import AbstractModel
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

class DoubleDrqnModel(AbstractModel):
    def __init__(self, env, epsilon_start = 1, epsilon_final = 0.01, epsilon_decay = 100000, gamma = 0.95, batch_size = 32, buffer_size = 7000):
        super().__init__(env)
        self.training_episodes = 0
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.device = torch.device("cuda:0")
        self.reset()        
        pass
    
    def reset(self):
        self.current_model = CnnLstmDQN((1, *self.env.input_shape), len(self.env.action_space))
        self.target_model = CnnLstmDQN((1, *self.env.input_shape), len(self.env.action_space))
        self.optimizer = optim.Adam(self.current_model.parameters())
        self.update_target()
        self.set_gpu()

    def set_gpu(self):
        self.current_model.to(self.device)
        self.target_model.to(self.device)

    def load(self, filename):
        """ Load model from file. """
        path = dir_path + '/save/doubledqn/' + filename
        checkpoint = torch.load(path)
        self.current_model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.training_episodes = checkpoint['episodes']
        self.update_target()
        self.set_gpu()

    def save(self, filename):
        """ Save model to file. """
        path = dir_path + '/save/doubledqn/' + filename
        state = {
            'episodes': self.training_episodes,
            'state_dict': self.current_model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(state, path)
        #torch.save(self.current_model.state_dict(), path)

    def train(self, episodes = 150000):
        eps_by_episode = lambda episode: self.epsilon_final + (self.epsilon_start - self.epsilon_final) * math.exp(-1. * episode / self.epsilon_decay)
        losses = []
        all_rewards = []
        episode_reward = 0
        """plt.plot([eps_by_episode(i) for i in range(episodes)])
        plt.show()"""
        state = self.env.reset()
        refresh_fps_time = 0.3
        last_time = time.time()
        cumulative = refresh_fps_time
        tot_reward = 0
        save_episodes_count = 0
        tr = trange(episodes+1, desc='Agent training', leave=True)
        for episode in tr:
            save_episodes_count += 1
            self.training_episodes += 1
            tr.set_description("Agent training (episode{}) Avg Reward {}".format(episode+1,tot_reward/(episode+1)))
            tr.refresh() 
            loop_time = time.time()-last_time
            cumulative -= loop_time
            if cumulative <= 0:
                cumulative = refresh_fps_time
                print('{}fps'.format(round(1/(loop_time))))
            last_time = time.time()
            epsilon = eps_by_episode(episode)
            """cv2.imshow('rl_image', np.squeeze(state))
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break"""
            action = self.current_model.act(state, epsilon, self.env.action_space)
            #print(self.env.action_space[action])
            next_state, reward, done, _ = self.env.step(action)
            self.replay_buffer.push(state, action, reward, next_state, done)

            tot_reward += reward
            
            state = next_state
            episode_reward += reward
            if episode_reward <= -50:
                done = True

            if done:
                self.env.clean_up()
                if save_episodes_count >=5000:
                    self.save(str(self.training_episodes)+'.pt')
                    save_episodes_count = 0
                    #plot(episode, all_rewards, losses)
                print('episode_reward:{}'.format(episode_reward))
                print('epsilon:{}'.format(epsilon))
                all_rewards.append(episode_reward)
                episode_reward = 0  
                state = self.env.reset()
                
            if len(self.replay_buffer) > self.batch_size:
                loss = self.compute_td_loss(self.batch_size)
                losses.append(loss.item())
                
            #if episode % 1000 == 0:
                #plot(episode, all_rewards, losses)

            if episode % 3000 == 0:
                self.update_target()
        #save the model when the training ends
        self.save(str(self.training_episodes)+'.pt')
        plot(episode, all_rewards, losses)
        #plt.plot([eps_by_episode(i) for i in range(10000)])
        #plt.show()

    def compute_td_loss(self, batch_size):
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

        state      = autograd.Variable(torch.cuda.FloatTensor(np.float32(state)))
        next_state = autograd.Variable(torch.cuda.FloatTensor(np.float32(next_state)), volatile=True)
        action     = autograd.Variable(torch.cuda.LongTensor(action))
        reward     = autograd.Variable(torch.cuda.FloatTensor(reward))
        done       = autograd.Variable(torch.cuda.FloatTensor(done))

        q_values      = self.current_model(state)
        next_q_values = self.current_model(next_state)
        next_q_state_values = self.target_model(next_state) 

        q_value          = q_values.gather(1, action.unsqueeze(1)).squeeze(1) 
        next_q_value     = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
        expected_q_value = reward + self.gamma * next_q_value * (1 - done)
        
        loss = (q_value - autograd.Variable(expected_q_value.data)).pow(2).mean()
            
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def update_target(self):
        self.target_model.load_state_dict(self.current_model.state_dict())

    def q(self, state):
        """ Return Q value for all action for a certain state.
            :return np.ndarray: Q values
        """
        return np.array([0, 0, 0, 0])
    
    def predict(self, **kwargs):
        """ Randomly choose the next action.
            :return int: selected action
        """
        return random.choice(self.env.action_space)

class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        state      = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
            
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done
    
    def __len__(self):
        return len(self.buffer)

class CnnLstmDQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(CnnLstmDQN, self).__init__()
        
        self.input_shape = input_shape
        self.num_actions = num_actions
        
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 16, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.lstm = nn.LSTM(self.feature_size(), 16, batch_first=True)
        
        self.fc = nn.Sequential(
            nn.Linear(16, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )
        
    def forward(self, x):
        x = self.features(x)
        # lstm input: [batch_size, seq_len, input_size]
        #print(x.shape)
        #print(x.size(0))
        x = x.view(x.size(0), 1, -1)
        x, _ = self.lstm(x)
        x = self.fc(x[:,-1,:])
        return x
    
    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)
    
    def act(self, state, epsilon, action_space):
        if random.random() > epsilon:
            state   = autograd.Variable(torch.cuda.FloatTensor(np.float32(state)).unsqueeze(0), volatile=True)
            q_value = self.forward(state)
            action  = q_value.max(1)[1].data[0]
        else:
            #print('random')
            action = random.randrange(len(action_space))
        return action

def plot(episode, rewards, losses):
    #clear_output(True)
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('episode %s. reward: %s' % (episode, np.mean(rewards[-10:])))
    plt.plot(rewards)
    plt.subplot(132)
    plt.title('loss')
    plt.plot(losses)   
    plt.show() 








