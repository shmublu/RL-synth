#visible environment: angle, speed, <discrete distance number from randomly placed reward>(0,4)
#hidden environment: visible environment + pos_x_reward, pos_y_reward, pos of agent
#Action: steering(0, 2*pi), brake(-0.3, 0.0)
#Reward: +10 for every reward you come within 2 distance of, -0.5/second
#Constants:
#Reward constants:
#Less than 3 meters away: 0
#Between 3-5 meters away: 1
#Between 5-8 meters away: 2
#Between 8-12 meters away: 3
#Between 12-20 meters away: 4
#20+ meters away

import pickle
import gym
from gym import spaces
import numpy as np
import random
import math
import argparse

import torch as T  
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

MAX_ANGLE = 6.2831
MAX_SPEED = 5
EPOCH_LEN = 200

STEER = 0
SPEED = 1
BRAKE = 2
TIME_CONSTANT = 1
rewards = [0]
reward_seq = []

VERBOSE = 0
FORMAL_CONSTRAIN = False

t = 0

no_synth = 3000
network_size = 256


#constraint_array = ["is_x1", "is_x2", "is_greater", "is_less", "constant"]
constraint_array = [0, 1, 1, 0, 1]



MAX_SPEED=0.25

N_ACTIONS = 2
#brake, speed

N_OBS = 6
#speed, x_dist, y_dist : constraint_array
is_manual = False
    
def square_rooted(x):
   return round(math.sqrt(sum([a*a for a in x])),3)
  
def is_violated(constraint, action):
    global is_manual
    if action[SPEED] < 0:
        return True
    if action[STEER] < 0 or action[STEER] > MAX_ANGLE:
        return True
    constrained_variable = -1
    if constraint[0] == 1:
        constrained_variable = STEER
    else:
        constrained_variable = SPEED
    if constraint[2] == 1:
        if action[constrained_variable] < constraint[4]:
            return True
    else:
        if action[constrained_variable] > constraint[4]:
            return True
    return False
def randomize_constraint(constraint):
    constraint[0] = (constraint[0] + 1) % 2
    constraint[1] = (constraint[1] + 1) % 2
    constraint[2] = (constraint[2] + 1) % 2
    constraint[3] = (constraint[3] + 1) % 2
    constraint[4] = random.uniform(0.001, MAX_ANGLE)

def pick_constraint(constraint):
    print("Steer(0) or speed(1)?")
    i = int(input())
    constraint[0] = (i + 1) % 2
    constraint[1] = (i) % 2
    print(">(0) or <(1)?")
    j = int(input())
    constraint[2] = (j + 1) % 2
    constraint[3] = (j) % 2
    print("Constant?")
    constraint[4] = float(input())
class OurCustomEnv(gym.Env):

    def __init__(self):
        self.observation_space = spaces.Box(low = np.array([-1000, -1000, -1, -1]), high = np.array([1000, 1000, 1, 1]),
                                                                     dtype = np.float32)
        self.action_space = spaces.Box(low = np.array([-1.0,-1.0]), high = np.array([1.0, 1.0]),
                                                        dtype = np.float32)
        self.user_x = 0
        self.user_y = 0
        self.rewards = 0
        self.reward_y = 0
        self.reward_x = 0
        self.place_new_reward(True,5.5)
        self.speed = MAX_SPEED
        self.time = 0
        
    def seed(self, seed=None):
        random.seed(seed)
        
    def step(self, action):
        global VERBOSE
        global FORMAL_CONSTRAIN
        global constraint_array
        global is_manual
        self.time += 1
        if FORMAL_CONSTRAIN:
            if action[STEER] < self.min_angle:
                if VERBOSE > 2:
                    print('Model output has been constrained up!', action[STEER], self.min_angle)
                action[STEER] = self.min_angle
            elif action[STEER] > self.max_angle:
                if VERBOSE > 2:
                    print('Model output has been constrained down!', action[STEER], self.max_angle)
                action[STEER] = self.max_angle
        steer = action[STEER] % MAX_ANGLE
        self.speed = action[SPEED] % MAX_SPEED
        steer_x = math.cos(steer)
        steer_y = math.sin(steer)
        steer_dir = np.array([steer_x,steer_y])
        dist_x = self.reward_x -self.user_x
        dist_y = self.reward_y -self.user_y
        old_dist = (dist_x ** 2 + dist_y **2 )
        self.user_x += self.speed * steer_dir[0]
        self.user_y += self.speed * steer_dir[1]
        
        dist_x = self.reward_x -self.user_x
        dist_y = self.reward_y -self.user_y
        dist = (dist_x ** 2 + dist_y **2 ) #we aren't going to square root for speed sake
        if dist <= 4:
            reward = 50.0
            self.place_new_reward()
        elif old_dist < dist:
            reward = -4 * (dist - old_dist)
        else:
            reward = 1 * (old_dist - dist)
        if is_violated(constraint_array, action):
            reward -= 200
            if VERBOSE > 3 or is_manual:
                print('Model has been penalized for operating outside of constraints')
        done = False if self.time < EPOCH_LEN else True
        info = {}
        angle_away = np.arctan2(dist_x,dist_y)
        if self.time % 97 == 0 and VERBOSE > 2:
            print(np.array([angle_away] + constraint_array))
            print("Speed: ", action[SPEED], "Steer: ", action[STEER], "Dist from Reward: ", dist_x, dist_y, "Constraint: ",constraint_array)
        elif VERBOSE > 3:
            print("Speed: ", action[SPEED],  "Steer: ", action[STEER], "Dist from Reward: ", dist_x, dist_y,  "Constraint: ",constraint_array)

        if not is_manual:
            randomize_constraint(constraint_array)
        else:
            pick_constraint(constraint_array)
        state = np.array([angle_away] + constraint_array)
        return state, reward, done, info 
    def reset(self):
        global reward_seq
        global t
        global constraint_array
        t += 1
        reward_seq.append(self.rewards)
        self.rewards = 0
        self.user_x = 0
        self.user_y = 0
        self.reward_y = 0
        self.reward_x = 0
        self.place_new_reward(True, 5.5)
        dist_x = self.user_x - self.reward_x
        dist_y = self.user_y - self.reward_y
        
        self.min_angle = 0
        self.max_angle = MAX_ANGLE
        
        self.speed = MAX_SPEED
        self.time = 0
        angle_away = np.arctan2(dist_x,dist_y)
        state = np.array([angle_away] + constraint_array)
        return state
    def place_new_reward(self, first = False, d = 6.5, r = rewards):
        dist_x = self.user_x - self.reward_x
        dist_y = self.user_y - self.reward_y
        dist = (dist_x ** 2 + dist_y **2 ) ** 0.5
        if not first:
            if VERBOSE > 2:
                print("reward was gotten!", self.time, dist)
            self.rewards += 1
        while dist < 4.4:
            self.reward_y = random.uniform(-1.0,1.0) * d + self.user_y
            self.reward_x = random.uniform(-1.0,1.0) * d + self.user_x
            dist_x = self.user_x - self.reward_x
            dist_y = self.user_y - self.reward_y
            dist = (dist_x ** 2 + dist_y **2 )
        
class ReplayBuffer():

    def __init__(self, max_size, input_shape=N_OBS, n_actions=N_ACTIONS):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.new_state_memory = np.zeros((self.mem_size, input_shape))     
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)   
    
    def store_transition(self, state, action, reward, state_, done):

        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = done
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)   
        batch = np.random.choice(max_mem, batch_size)
        states = self.state_memory[batch]        
        states_ = self.new_state_memory[batch] 
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]    

        return states, actions, rewards, states_, dones
    
    
class CriticNetwork(nn.Module):
    def __init__(self, beta):
        global network_size
        super(CriticNetwork, self).__init__()
        self.input_dims = N_OBS    #fb, insta
        self.fc1_dims = network_size * 3 // 4    #hidden layers
        self.fc2_dims = network_size * 3 // 4    #hidden layers   #hidden layers
        self.n_actions = N_ACTIONS     #fb, insta
        self.fc1 = nn.Linear(self.input_dims+ self.n_actions, self.fc1_dims) #state + action
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q1 = nn.Linear(self.fc2_dims, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        q1_action_value = self.fc1(T.cat([state, action], dim=1 )) 
        q1_action_value = F.leaky_relu(q1_action_value) 
        q1_action_value = self.fc2(q1_action_value) 
        q1_action_value = F.leaky_relu(q1_action_value)
        q1 = self.q1(q1_action_value)
        return q1

class ActorNetwork(nn.Module):
    def __init__(self, alpha):
        super(ActorNetwork, self).__init__()
        self.input_dims = N_OBS
        self.fc1_dims = network_size
        self.fc2_dims = network_size
        self.n_actions = N_ACTIONS
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, state):
        prob = self.fc1(state)
        prob = F.leaky_relu(prob)
        prob = self.fc2(prob)
        prob = F.leaky_relu(prob)
        #fixing each agent between 0 and 1 and transforming each action in env
        mu = self.mu(prob)
        return mu
    
    
class Agent(object):
    def __init__(self, alpha, beta, input_dims=N_OBS, tau=0, env=None, gamma=0.99,
                 n_actions=N_ACTIONS, max_size=1000000,  batch_size=64):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size)
        self.batch_size = batch_size
        self.actor = ActorNetwork(alpha)
        self.critic = CriticNetwork(beta)
        self.target_actor = ActorNetwork(alpha)
        self.target_critic = CriticNetwork(beta)
        
        self.scale = 1.0
        self.noise = np.random.normal(scale=self.scale,size=(n_actions))
        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        self.actor.eval()
        observation = T.tensor(observation, dtype=T.float).to(self.actor.device)
        mu = self.actor.forward(observation).to(self.actor.device)
        mu_prime = mu #+ T.tensor(self.noise, dtype=T.float).to(self.actor.device)
        self.actor.train()
        return mu_prime.cpu().detach().numpy()

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)
        
        reward = T.tensor(reward, dtype=T.float).to(self.critic.device)
        done = T.tensor(done).to(self.critic.device)
        new_state = T.tensor(new_state, dtype=T.float).to(self.critic.device)
        action = T.tensor(action, dtype=T.float).to(self.critic.device)
        state = T.tensor(state, dtype=T.float).to(self.critic.device)

        self.target_actor.eval()
        self.target_critic.eval()
        self.critic.eval()
        target_actions = self.target_actor.forward(new_state)
        critic_value_ = self.target_critic.forward(new_state, target_actions)
        critic_value = self.critic.forward(state, action)

        target = []
        for j in range(self.batch_size):
            target.append(reward[j] + self.gamma*critic_value_[j]*done[j])
        target = T.tensor(target).to(self.critic.device)
        target = target.view(self.batch_size, 1)

        self.critic.train()
        #   nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        #nn.utils.clip_grad_value_(self.critic.parameters(), clip_value=1.0)
        self.critic.optimizer.step()

        self.critic.eval()
        self.actor.optimizer.zero_grad()
        mu = self.actor.forward(state)
        self.actor.train()
        actor_loss = -self.critic.forward(state, mu)
        actor_loss = T.mean(actor_loss)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=2.0, norm_type=2)
        self.actor.optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_dict = dict(target_critic_params)
        target_actor_dict = dict(target_actor_params)

        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + (1-tau)*target_critic_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)

        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + (1-tau)*target_actor_dict[name].clone()
        self.target_actor.load_state_dict(actor_state_dict)
    
    
def rl_manual_test(training_runs= 1, train_no_synth = 0):
    global no_synth
    global VERBOSE
    global is_manual
    file = open("agent.pickle",'rb')
    agent = pickle.load(file)
    file.close()
    VERBOSE = 5
    no_synth = train_no_synth
    env = OurCustomEnv()
    score_history = []
    is_manual = True
    max_score = -50000
    for i in range(training_runs):
        obs = env.reset()
        done = False
        score = 0
        while not done:
            act = agent.choose_action(obs)
            new_state, reward, done, info = env.step(act)
            score += reward
            obs = new_state
        if max_score < score:
            max_score = score
            if VERBOSE > 0:
                print("New Max: " +str(max_score))
        if VERBOSE > 1:
            print(i,score)
        score_history.append(score)
    obs = env.reset()
    print('Score History: ', score_history)
    print('Reward History: ',reward_seq[1:])

    
def rl_test(training_runs= 1, train_no_synth = 0):
    global no_synth
    global VERBOSE
    file = open("agent.pickle",'rb')
    agent = pickle.load(file)
    file.close()
    VERBOSE += 1
    no_synth = train_no_synth
    env = OurCustomEnv()
    score_history = []
    max_score = -50000
    for i in range(training_runs):
        obs = env.reset()
        done = False
        score = 0
        while not done:
            act = agent.choose_action(obs)
            new_state, reward, done, info = env.step(act)
            score += reward
            obs = new_state
        if max_score < score:
            max_score = score
            if VERBOSE > 0:
                print("New Max: " +str(max_score))
        if VERBOSE > 1:
            print(i,score)
        score_history.append(score)
    obs = env.reset()
    print('Score History: ', score_history)
    print('Reward History: ',reward_seq[1:])

    
def load_rl_train(training_runs=5000, train_no_synth = 3000, net_size=256):
    global t
    global VERBOSE
    global rewards
    global reward_seq
    global no_synth
    global network_size
    env = OurCustomEnv()
    no_synth = train_no_synth
    network_size = net_size
    file = open("agent.pickle",'rb')
    agent = pickle.load(file)
    file.close()
    score_history = []
    max_score = -50000
    for i in range(training_runs):
        t=i
        obs = env.reset()
        done = False
        score = 0
        while not done:
            act = agent.choose_action(obs)
            new_state, reward, done, info = env.step(act)
            agent.remember(obs, act, reward, new_state, int(done))
            agent.learn()
            score += reward
            obs = new_state
        if max_score < score:
            max_score = score
            if VERBOSE > 0:
                print("New Max: " +str(max_score))
        if VERBOSE > 1:
            print(i,score)
        score_history.append(score)
        if len(score_history) > 51 and VERBOSE > 1 and i % 10 == 0:
            print("Average score(last 50): " + str(sum(score_history[-50:])/ len(score_history[-50:])))
    pickle.dump(agent, file = open("agent.pickle", "wb"))
    obs = env.reset()
    if VERBOSE > 0:    
        print('Score History: ', score_history)
        print('Reward History: ',reward_seq[1:])

def rl_train(training_runs=5000, train_no_synth = 3000, net_size=256):
    global t
    global VERBOSE
    global rewards
    global reward_seq
    global no_synth
    global network_size
    env = OurCustomEnv()
    no_synth = train_no_synth
    network_size = net_size
    agent = Agent(alpha=0.000025, beta=0.00025, tau=0.001, env=env, batch_size=64, n_actions=N_ACTIONS)
    score_history = []
    max_score = -50000
    for i in range(training_runs):
        t=i
        obs = env.reset()
        done = False
        score = 0
        while not done:
            act = agent.choose_action(obs)
            new_state, reward, done, info = env.step(act)
            agent.remember(obs, act, reward, new_state, int(done))
            agent.learn()
            score += reward
            obs = new_state
        if max_score < score:
            max_score = score
            if VERBOSE > 0:
                print("New Max: " +str(max_score))
        if VERBOSE > 1:
            print(i,score)
        score_history.append(score)
        if len(score_history) > 51 and VERBOSE > 1 and i % 10 == 0:
            print("Average score(last 50): " + str(sum(score_history[-50:])/ len(score_history[-50:])))
    pickle.dump(agent, file = open("agent.pickle", "wb"))
    obs = env.reset()
    if VERBOSE > 0:    
        print('Score History: ', score_history)
        print('Reward History: ',reward_seq[1:])
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("training_runs", help="How many times to train in total", type=int)
    parser.add_argument("training_runs_no_synth", help="How many times to train without synthesis", type=int)
    parser.add_argument("network_size", help="Size of Actor NN, and Critic NN is 0.75 times this(recommended: 128+)", type=int)
    parser.add_argument("-fc", help="Formal Constraint Mode",action="store_true")
    group2 = parser.add_mutually_exclusive_group()
    group1 = parser.add_mutually_exclusive_group()
    group2.add_argument("-nt", help="Non-training Mode (load the model from agent.pickle and test it out)",action="store_true")
    group2.add_argument("-mi", help="Manual Input Mode (load the model from agent.pickle and test it out- you get to choose the limits)",action="store_true")
    group2.add_argument("-tp", help="Train Pickled Model Mode (load the model from agent.pickle and retrain it)",action="store_true")
    group1.add_argument("-v", help="Verbose Mode",action="store_true")
    group1.add_argument("-vv", help="Super Verbose Mode",action="store_true")
    group1.add_argument("-vvv", help="Super Duper Verbose Mode",action="store_true")
    args = parser.parse_args()
    if args.training_runs < args.training_runs_no_synth:
        exit('ERROR: Invalid number of no-synth runs!')
    if args.vvv:
        VERBOSE = 3
    elif args.vv:
        VERBOSE = 2
    elif args.v:
        VERBOSE = 1
    if args.fc:
        FORMAL_CONSTRAIN = True
    if args.nt:
        rl_test(args.training_runs,args.training_runs_no_synth)
    elif args.mi:
        rl_manual_test(args.training_runs,args.training_runs_no_synth)
    elif args.tp:
        load_rl_train(args.training_runs, args.training_runs_no_synth, args.network_size)
    else:
        rl_train(args.training_runs, args.training_runs_no_synth, args.network_size)
    


#https://www.analyticsvidhya.com/blog/2021/08/creating-continuous-action-bot-using-deep-reinforcement-learning/
#https://github.com/openai/gym/issues/1482
#https://stackoverflow.com/questions/44404281/openai-gym-understanding-action-space-notation-spaces-box
