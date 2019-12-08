import keras
from keras.models import Sequential
from keras.utils import np_utils
from keras.models import load_model
from keras.layers import Dense
from keras.optimizers import Adam
from qiskit.quantum_info import random_state
import numpy as np
import random
import matplotlib
import math
import random
from functions import *
import tensorflow as tf
import os
import objgraph
import sys
import csv


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto(gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3))
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


class Environment:

    def reset(self):
        state = random_state(2)
        new_state = state_norm(state)
        new_state = np.reshape(new_state.flatten(), (1,1,2))
        return new_state
class Agent:
    def __init__(self, state_size, new_state, is_eval=False, model_name=""):
        self.state_size = state_size
        self.action_size = 6 # measurement, CNOT, bit-flip
#         self.memory = deque(maxlen=1000)
        self.inventory = []
        self.model_name = model_name
        self.value = new_state
        self.is_eval = is_eval
        self.done = False
        self.final_state = [1/math.sqrt(2),1/math.sqrt(2)]

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.01
        self.model = self.QC_model()

#         self.model = load_model("models/" + model_name) if is_eval else self._model()

    def QC_model(self):
        model = Sequential()
        model.add(Dense(units=16, input_shape=self.state_size, activation="relu", name='layer1'))
        model.add(Dense(units=32, activation="relu", name='layer2'))
        model.add(Dense(units=8, activation="relu", name='layer3'))
        model.add(Dense(self.action_size, activation="softmax"))
        model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=0.003))
        return model


    def act(self, state):
#        action = 0
        if random.random() <= self.epsilon:
            options = self.model.predict(state)
            action =  random.randrange(self.action_size)
        else:
            options = self.model.predict(state)
            action = options[0]
        options = np.squeeze(options)
        return action, options

    def train(self):
        column_name = ['episode','time_step','reward', 'projection']
        with open('reward.csv','a') as fd:
            write_outfile = csv.writer(fd)
            write_outfile.writerow(column_name)
        batch_size = 10
        t = 0                   #increment
        states, prob_actions, dlogps, drs, proj_data, reward_data =[], [], [], [], [], []
        tr_x, tr_y = [],[]
        avg_reward = []
        reward_sum = 0
        ep_number = 0
        prev_state = None
        new_state = self.value

        while ep_number<10000:
            print("episode number: ",ep_number)
            prev_state = new_state
            states.append(new_state)
            action, probs = self.act(new_state)
            print("Action: ", action)
            print("Old State: ", new_state)
            prob_actions.append(probs)
            y = np.zeros([self.action_size])
            y[action] = 1
            if(action==4):
                bit = eval(command[action])
            else:
                new_state = eval(command[action])
            new_state = np.reshape(new_state, (1,1,2))
            print("State: ", new_state)
            proj = projection(new_state, self.final_state)
            print("projection: ", proj)
            proj_data.append(proj)
            if(t==0):
                rw = reward(proj,0)
                drs.append(rw)
                reward_sum+=rw
            else:
                rw = reward(proj_data[t], proj_data[t-1])
                drs.append(rw)
                print("present reward: ", rw)
                reward_sum+=rw
            reward_data = [ep_number, t, reward_sum, proj]
            with open('reward.csv','a') as fd:
                write_outfile = csv.writer(fd)
                write_outfile.writerow(reward_data)
#            with open('reward.csv','a') as fd:
#                write_outfile = csv.writer(fd)
#                write_outfile.writerow((reward_sum))
            print("reward till now: ",reward_sum)
#            print("probs: ", probs.shape)
            dlogps.append(np.array(y).astype('float32') * probs)
            print("dlogps before time step: ", len(dlogps))
            del(probs, action)
            print("time step: ",t)
            t+=1
            if(t==100):                         #### Done State
                ep_number+=1
                ep_x = np.vstack(states)
                print("length of states: ", len(ep_x))
                ep_dlogp = np.vstack(dlogps)
                print("dlogps: ", len(dlogps))
                ep_reward = np.vstack(drs)
                disc_rw = discounted_reward(ep_reward,self.gamma)
                print("disc_rw: ", len(disc_rw))
                disc_rw = disc_rw.astype('float32')
                disc_rw -= np.mean(disc_rw)
                disc_rw /= np.std(disc_rw)

                tr_y_len = len(ep_dlogp)
                print("ep_dlogp: ", len(ep_dlogp))
                ep_dlogp*=disc_rw
#                states, drs =[], []
                if ep_number % batch_size == 0:

                    input_tr_y = prob_actions - self.learning_rate * ep_dlogp
#                    input_tr_y = ep_dlogp
                    input_tr_y = np.reshape(input_tr_y, (tr_y_len,1,6))

                    self.model.train_on_batch(ep_x, input_tr_y)
                    dlogps, drs, states, prob_actions,proj_data, reward_data = [],[],[],[],[],[]
                avg_reward.append(float(reward_sum))
                if(ep_number%1000==0):
                    self.model.save("model_ep{:}.h5".format(ep_number))
                if len(avg_reward)>100: avg_reward.pop(0)
                print('Episode {:} reward {:.2f}, Last 30ep Avg. rewards {:.2f}.'.format(
                    ep_number,reward_sum,np.mean(avg_reward)))
                env = Environment()
                state = env.reset()
                t=0
                objgraph.show_most_common_types()

        return avg_reward

command = ['bit_flip_X(new_state)',
           'bit_flip_Y(new_state)',
           'hadamard_X(new_state)',
           'hadamard_Y(new_state)',
           'measurement(round(abs(new_state[0][0][1])**2),\
                    round(abs(new_state[0][0][1])**2))',
           'nothing(new_state)']


env = Environment()
state = env.reset()

agent = Agent(state[0].shape, state)
reward = agent.train()
