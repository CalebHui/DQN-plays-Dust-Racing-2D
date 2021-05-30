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
from statistics import mean 
import collections, itertools
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
from .prioritized_buffer import Memory

class PrioritizedDqnModel(AbstractModel):
    def __init__(self, env, network_type='CnnDQN', epsilon_start = 1, epsilon_final = 0.01, epsilon_decay = 100000, gamma = 0.99, batch_size = 32, buffer_size = 30000, tau=4):
        super().__init__(env)
        self.training_episodes = 0
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.replay_buffer = Memory(buffer_size)
        self.tau = tau
        self.state_buffer = deque(maxlen=tau)
        self.losses = []
        self.all_rewards = []
        self.tot_reward = 0
        self.frame_skipping = 5
        self.network_type = network_type
        self.device = torch.device("cuda:0")
        self.reset()        
        pass
    
    def reset(self):
        if self.network_type == 'CnnDQN':
            self.current_model = CnnDQN((self.tau, *self.env.input_shape), len(self.env.action_space))
            self.target_model = CnnDQN((self.tau, *self.env.input_shape), len(self.env.action_space))
        if self.network_type == 'DDQN':
            self.current_model = DDQN((self.tau, *self.env.input_shape), len(self.env.action_space))
            self.target_model = DDQN((self.tau, *self.env.input_shape), len(self.env.action_space))
        self.optimizer = optim.Adam(self.current_model.parameters(), lr=0.0001)
        self.update_target()
        self.set_gpu()

    def set_gpu(self):
        self.current_model.to(self.device)
        self.target_model.to(self.device)

    def load(self, filename):
        """ Load model from file. """
        path = '{}/save/doubledqn/{}-{}'.format(dir_path, self.network_type, filename)
        checkpoint = torch.load(path)
        self.current_model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.training_episodes = checkpoint['episodes']
        self.all_rewards = checkpoint['all_rewards']
        self.losses = checkpoint['losses']
        # disable this when training
        self.replay_buffer.tree.data = checkpoint['replay_buffer']
        self.tot_reward = checkpoint['tot_reward']
        self.update_target()
        self.set_gpu()

    def save(self, filename):
        """ Save model to file. """
        path = '{}/save/doubledqn/{}-{}'.format(dir_path, self.network_type, filename)
        saved_buffer_len = 1000
        buffer = self.replay_buffer.tree.data
        if len(self.replay_buffer) >= saved_buffer_len:
            buffer = self.replay_buffer.tree.data[0:saved_buffer_len]
        state = {
            'episodes': self.training_episodes,
            'state_dict': self.current_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'all_rewards': self.all_rewards,
            'losses': self.losses,
            'replay_buffer': buffer,
            'tot_reward': self.tot_reward
        }
        torch.save(state, path)
        #torch.save(self.current_model.state_dict(), path)

    def init_state_buffer(self):
        self.state_buffer.clear()
        #[self.state_buffer.append(np.zeros(self.env.input_shape)) for i in range(self.tau)]

    def train(self, episodes = 600000):
        eps_by_episode = lambda episode: self.epsilon_final + (self.epsilon_start - self.epsilon_final) * math.exp(-1. * episode / self.epsilon_decay)
        
        episode_reward = 0
        plot(self.training_episodes, self.all_rewards, self.losses)
        plt.plot([eps_by_episode(i) for i in range(episodes)])
        plt.show()
        self.init_state_buffer()
        frame = self.env.reset()
        for i in range(self.tau):
            self.state_buffer.append(frame)
        refresh_fps_time = 0.3
        last_time = time.time()
        cumulative = refresh_fps_time
        save_episodes_count = 0
        trained_episodes_count = 0
        tr = trange(episodes+1, desc='Agent training', leave=True)
        done = False
        action = None
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

            # one action every frame_skipping frames
            state = np.stack(self.state_buffer)
            action = self.current_model.act(state, epsilon, self.env.action_space)
            rewards = []
            last_frame = None
            secondlast_frame = None
            for i in range(self.frame_skipping):
                frame, reward, done, _ = self.env.step(action)
                #secondlast_frame = last_frame
                last_frame = frame
                rewards.append(reward)
                if len(self.replay_buffer) > self.batch_size:
                    loss = self.compute_td_loss(self.batch_size)
                    self.losses.append(loss.item())
                if done:
                    break
            if secondlast_frame is not None:
                frame = np.maximum(secondlast_frame, last_frame)
            else:
                frame = last_frame
            """cv2.imshow('rl_image', np.squeeze(frame))
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break"""
            # if done => use last reward(lose/finsih)
            if not done:
                reward = mean(rewards)
            self.tot_reward += reward
            episode_reward += reward
            self.state_buffer.append(frame)
            next_state = np.stack(self.state_buffer)
            self.replay_buffer.store(state, action, reward, next_state, done)

            #penalty threshold
            if episode_reward <= -0.5:
                done = True
            if done:
                trained_episodes_count += 1
                self.env.clean_up()
                if save_episodes_count >= 250000:
                    self.save(str(self.training_episodes)+'.pt')
                    save_episodes_count = 0
                    #plot(episode, all_rewards, losses)
                print('episode_reward:{}'.format(episode_reward))
                print('epsilon:{}'.format(epsilon))
                print('episode:{}'.format(episode))
                self.all_rewards.append(episode_reward)
                if trained_episodes_count % 20 == 0:
                    game_reward, finish_lap, finish_game = self.play()
                    if finish_lap or finish_game:
                        self.save('success-{}.pt'.format(self.training_episodes))
                episode_reward = 0
                self.init_state_buffer()
                frame = self.env.reset()
                for i in range(self.tau):
                    self.state_buffer.append(frame)
            
            """if len(self.replay_buffer) > self.batch_size:
                loss = self.compute_td_loss(self.batch_size)
                self.losses.append(loss.item())"""
            """if episode % 100 == 0:
                plot(self.training_episodes, self.all_rewards, self.losses)"""

            if self.training_episodes % 4000 == 0:
                self.update_target()
        #save the model when the training ends
        self.save(str(self.training_episodes)+'.pt')
        plot(self.training_episodes, self.all_rewards, self.losses)
        #plt.plot([eps_by_episode(i) for i in range(10000)])
        #plt.show()

    def play(self, episodes = 1, plot_stat=False, random_model=False):
        #modify frame_skipping to adjust the fps
        #In my experiments, the fps is around 12-13 fps
        frame_skipping = 35
        episode_reward = 0
        tot_reward = 0
        all_rewards = []
        self.init_state_buffer()
        frame = self.env.reset()
        for i in range(self.tau):
            self.state_buffer.append(frame)
        refresh_fps_time = 0.3
        last_time = time.time()
        cumulative = refresh_fps_time
        done = False
        action = None
        epsilon = 0 #0.01
        if random_model:
            epsilon = 1
        episode = 0
        finish_lap = False
        finish_game = False
        lap_nums = []
        lap_num = 0
        while episode < episodes:
            loop_time = time.time()-last_time
            cumulative -= loop_time
            if cumulative <= 0:
                cumulative = refresh_fps_time
                print('{}fps'.format(round(1/(loop_time))))
                #print('{}loop_time'.format(loop_time))
            last_time = time.time()
            
            # one action every frame_skipping frames
            state = np.stack(self.state_buffer)
            action = self.current_model.act(state, epsilon, self.env.action_space)
            rewards = []
            last_frame = None
            secondlast_frame = None
            for i in range(frame_skipping):
                frame, reward, done, info = self.env.step(action)
                #secondlast_frame = last_frame
                last_frame = frame
                rewards.append(reward)
                if info == 'FINISH_LAP':
                    finish_lap = True
                    lap_num += 1
                if info == 'FINISH_GAME':
                    finish_game = True
                if done:
                    break
            if secondlast_frame is not None:
                frame = np.maximum(secondlast_frame, last_frame)
            else:
                frame = last_frame
            cv2.imshow('rl_image', np.squeeze(frame))
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
            # if done => use last reward(lose/finsih)
            if not done:
                reward = mean(rewards)
            tot_reward += reward
            episode_reward += reward
            self.state_buffer.append(frame)
            next_state = np.stack(self.state_buffer)

            #penalty threshold
            if episode_reward <= -0.5:
                done = True
            if done:
                episode += 1
                self.env.clean_up()
                print('episode_reward:{}'.format(episode_reward))
                print('epsilon:{}'.format(epsilon))
                print('episode:{}'.format(episode))
                lap_nums.append(lap_num)
                lap_num = 0
                all_rewards.append(episode_reward)
                episode_reward = 0
                self.init_state_buffer()
                frame = self.env.reset()
                for i in range(self.tau):
                    self.state_buffer.append(frame)
        if plot_stat:
            plt.figure(figsize=(20,5))
            plt.subplot(131)
            plt.title('episode %s. mean reward: %s' % (episode, np.mean(all_rewards)))
            plt.plot(all_rewards)
            plt.subplot(132)
            plt.title('mean laps finished: %s' % (np.mean(lap_nums)))
            plt.plot(lap_nums)
            plt.show() 
        return tot_reward, finish_lap, finish_game

    def compute_td_loss(self, batch_size):
        state, action, reward, next_state, done, indices, weights = self.replay_buffer.sample(batch_size)

        state      = autograd.Variable(torch.cuda.FloatTensor(np.float32(state)))
        next_state = autograd.Variable(torch.cuda.FloatTensor(np.float32(next_state)), volatile=True)
        action     = autograd.Variable(torch.cuda.LongTensor(action))
        reward     = autograd.Variable(torch.cuda.FloatTensor(reward))
        done       = autograd.Variable(torch.cuda.FloatTensor(done))
        weights    = autograd.Variable(torch.cuda.FloatTensor(weights))

        q_values      = self.current_model(state)
        next_q_values = self.current_model(next_state)
        next_q_state_values = self.target_model(next_state) 
        q_value          = q_values.gather(1, action.unsqueeze(1)).squeeze(1) 
        next_q_value     = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
        expected_q_value = reward + self.gamma * next_q_value * (1 - done)
        loss = (q_value - autograd.Variable(expected_q_value.data)).pow(2) * weights
        self.replay_buffer.batch_update(indices, loss.data.cpu().numpy())
        loss = loss.mean()
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.current_model.parameters():
            param.grad.data.clamp_(-1, 1)
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

    def debug_buffer(self, index):
        #plot(self.training_episodes, self.all_rewards, self.losses)
        print('episode in buffer:{}/{}'.format(index+1, len(self.replay_buffer.tree.data)))
        state, action, reward, next_state, done = self.replay_buffer.peek(index)
        for p in self.current_model.parameters():
            print('===========\ngradient:{}\n----------\n'.format(torch.max(p)))
        print(state.nbytes)
        print('action:{}'.format(self.env.action_space[action]))
        print('reward:{}'.format(reward))
        print('done:{}'.format(done))
        print('(A==B).all():{}'.format((state==next_state).all()))
        state_tensor      = autograd.Variable(torch.cuda.FloatTensor(np.float32(state)))
        next_state_tensor = autograd.Variable(torch.cuda.FloatTensor(np.float32(next_state)), volatile=True)
        if self.network_type == 'CnnDQN':
            self.print_CNN(state_tensor, next_state_tensor)
        if self.network_type == 'DDQN':
            self.print_DDQN(state_tensor, next_state_tensor)
        print('**********************************************************')
        state = np.squeeze(state)
        next_state = np.squeeze(next_state)
        state = np.hstack((state[0], state[1], state[2], state[3]))
        next_state = np.hstack((next_state[0], next_state[1], next_state[2], next_state[3]))
        image = np.vstack((state, next_state))
        cv2.imshow('state and next state', image)
        cv2.waitKey(0)

    def print_CNN(self, state_tensor, next_state_tensor):
        q_values = self.current_model(state_tensor)
        next_q_values = self.current_model(next_state_tensor)
        next_q_state_values = self.target_model(next_state_tensor)
        print('q_values:{}'.format(q_values))
        q_action  = q_values.max(1)[1].data[0]
        print('current model best action:{}'.format(self.env.action_space[q_action]))
        print('next_q_values:{}'.format(next_q_values))

    def print_DDQN(self, state_tensor, next_state_tensor):
        advantage, value = self.current_model.get_advantage_n_value(state_tensor)
        print('advantage:{}'.format(advantage))
        print('value:{}'.format(value))
        q_values = self.current_model(state_tensor)
        print('q_values:{}'.format(q_values))
        q_action  = q_values.max(1)[1].data[0]
        print('current model best action:{}'.format(self.env.action_space[q_action]))
        next_advantage, next_value = self.current_model.get_advantage_n_value(next_state_tensor)
        print('next advantage:{}'.format(next_advantage))
        print('next value:{}'.format(next_value))
        next_q_values = self.current_model(next_state_tensor)
        next_q_state_values = self.target_model(next_state_tensor)
        print('next_q_values:{}'.format(next_q_values))

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

    def peek(self, index):
        return self.buffer[index]

    def load(self, buffer):
        self.buffer = buffer
    
    def __len__(self):
        return len(self.buffer)

class DDQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DDQN, self).__init__()     

        self.input_shape = input_shape
        self.num_actions = num_actions

        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.fc_input_dim = self.feature_size()
        
        self.advantage = nn.Sequential(
            nn.Linear(self.fc_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_actions)
        )
        
        self.value = nn.Sequential(
            nn.Linear(self.fc_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, x):
        advantage, value = self.get_advantage_n_value(x)
        return value + advantage  - advantage.mean()
    
    def get_advantage_n_value(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        advantage = self.advantage(x)
        value     = self.value(x)
        return advantage, value

    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)

    def act(self, state, epsilon, action_space):
        if random.random() > epsilon:
            state   = autograd.Variable(torch.cuda.FloatTensor(np.float32(state)).unsqueeze(0), volatile=True)
            q_value = self.forward(state)
            action  = q_value.max(1)[1].item()
        else:
            print('random')
            #print(action_space[action])
            action = random.randrange(len(action_space))
        return action

class CnnDQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(CnnDQN, self).__init__()
        
        self.input_shape = input_shape
        self.num_actions = num_actions

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
            #print(q_value)
            action  = q_value.max(1)[1].data[0]
            #print(action_space[action])
        else:
            action = random.randrange(len(action_space))
            print('random')
            #print(action_space[action])
        return action

def plot(frames, rewards, losses):
    #clear_output(True)
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('frames %s. reward: %s' % (frames, np.mean(rewards[-10:])))
    plt.plot(rewards)
    plt.subplot(132)
    plt.title('loss')
    plt.plot(losses)   
    plt.show() 