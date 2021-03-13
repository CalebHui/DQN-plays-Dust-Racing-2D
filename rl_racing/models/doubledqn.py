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

class DoubleDqnModel(AbstractModel):
    def __init__(self, env, epsilon_start = 1, epsilon_final = 0.01, epsilon_decay = 50000, gamma = 0.95, batch_size = 32, buffer_size = 10000, tau=4):
        super().__init__(env)
        self.training_episodes = 0
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.tau = tau
        self.state_buffer = deque(maxlen=tau)
        self.losses = []
        self.all_rewards = []
        self.tot_reward = 0
        self.frame_skipping = 9
        self.device = torch.device("cuda:0")
        self.reset()        
        pass
    
    def reset(self):
        self.current_model = CnnDQN((self.tau, *self.env.input_shape), len(self.env.action_space))
        self.target_model = CnnDQN((self.tau, *self.env.input_shape), len(self.env.action_space))
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
        self.all_rewards = checkpoint['all_rewards']
        self.losses = checkpoint['losses']
        self.replay_buffer = checkpoint['replay_buffer']
        self.tot_reward = checkpoint['tot_reward']
        self.update_target()
        self.set_gpu()

    def save(self, filename):
        """ Save model to file. """
        path = dir_path + '/save/doubledqn/' + filename
        state = {
            'episodes': self.training_episodes,
            'state_dict': self.current_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'all_rewards': self.all_rewards,
            'losses': self.losses,
            'replay_buffer': self.replay_buffer,
            'tot_reward': self.tot_reward
        }
        torch.save(state, path)
        #torch.save(self.current_model.state_dict(), path)

    def init_state_buffer(self):
        [self.state_buffer.append(np.zeros(self.env.input_shape)) for i in range(self.tau)]

    def train(self, episodes = 100000):
        eps_by_episode = lambda episode: self.epsilon_final + (self.epsilon_start - self.epsilon_final) * math.exp(-1. * episode / self.epsilon_decay)
        
        episode_reward = 0
        """plt.plot([eps_by_episode(i) for i in range(episodes)])
        plt.show()"""
        self.init_state_buffer()
        frame = self.env.reset()
        self.state_buffer.append(frame)
        refresh_fps_time = 0.3
        last_time = time.time()
        cumulative = refresh_fps_time
        save_episodes_count = 0
        tr = trange(episodes+1, desc='Agent training', leave=True)
        frame_skipping_count = 0
        done = False
        for episode in tr:
            save_episodes_count += 1
            self.training_episodes += 1
            tr.set_description("Agent training (episode{}) Avg Reward {}".format(self.training_episodes,self.tot_reward/(self.training_episodes)))
            tr.refresh() 
            loop_time = time.time()-last_time
            cumulative -= loop_time
            if cumulative <= 0:
                cumulative = refresh_fps_time
                print('{}fps'.format(round(1/(loop_time))))
                #print('{}loop_time'.format(loop_time))
            last_time = time.time()
            epsilon = eps_by_episode(self.training_episodes)
            """cv2.imshow('rl_image', np.squeeze(frame))
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break"""
            # one action every frame_skipping frames
            if frame_skipping_count % self.frame_skipping == 0:
                state = np.stack(self.state_buffer)
                action = self.current_model.act(state, epsilon, self.env.action_space)
                #print(self.env.action_space[action])
                next_frame, reward, done, _ = self.env.step(action)
                frame = next_frame
                self.state_buffer.append(frame)
                next_state = np.stack(self.state_buffer)
                self.replay_buffer.push(state, action, reward, next_state, done)
                self.tot_reward += reward
                episode_reward += reward

            if episode_reward <= -100:
                done = True

            frame_skipping_count += 1
            if done:
                frame_skipping_count = 0
                self.env.clean_up()
                if save_episodes_count >=50000:
                    self.save(str(self.training_episodes)+'.pt')
                    save_episodes_count = 0
                    #plot(episode, all_rewards, losses)
                print('episode_reward:{}'.format(episode_reward))
                print('epsilon:{}'.format(epsilon))
                self.all_rewards.append(episode_reward)
                episode_reward = 0
                self.init_state_buffer()
                frame = self.env.reset()
                self.state_buffer.append(frame)
            if len(self.replay_buffer) > self.batch_size:
                loss = self.compute_td_loss(self.batch_size)
                self.losses.append(loss.item())
                
            """if episode % 100 == 0:
                plot(self.training_episodes, self.all_rewards, self.losses)"""

            if self.training_episodes % 3000 == 0:
                self.update_target()
        #save the model when the training ends
        self.save(str(self.training_episodes)+'.pt')
        plot(self.training_episodes, self.all_rewards, self.losses)
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

class CnnDQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(CnnDQN, self).__init__()
        
        self.input_shape = input_shape
        self.num_actions = num_actions
        
        """self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 16, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )"""

        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)
    
    def act(self, state, epsilon, action_space):
        if random.random() > epsilon:
            state   = autograd.Variable(torch.cuda.FloatTensor(np.float32(state)).unsqueeze(0), volatile=True)
            q_value = self.forward(state)
            action  = q_value.max(1)[1].data[0]
            #print(action_space[action])
        else:
            action = random.randrange(len(action_space))
            #print('random')
            #print(action_space[action])
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








