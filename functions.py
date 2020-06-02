from qiskit.quantum_info import random_state
import numpy as np
import math
import decimal

def state_norm(state):
    norm = 0.
    for i in range(state.shape[0]):
        norm += abs(state[i])**2
    norm = math.sqrt(norm)
    state = state/(norm)
    return state


def bit_flip_Y(x):
    matrix = np.array([[0, -1j],[1j,0]])
    x = np.squeeze(x)
    probs = np.matmul(matrix,x)
#     prob_1 = probs[0]
#     prob_2 = probs[1]
#     return prob_1, prob_2
    return probs

# def measurement(x,y):
#     p1 = abs(x)**2
#     p2 = abs(y)**2
#     diff = 1 - (p1+p2)
#     p1 = p1+diff
#     return np.random.choice(2, 1, p=[p1,p2])
def measurement(x,y):
    p1 = abs(x)**2
    p2 = abs(y)**2
    diff = 1 - (p1+p2)
    p1 = p1+diff
    bit =  np.random.choice(2, 1, p=[p1,p2])
    if bit==1:
        state = np.array([0+0j,1+0j])
    elif bit==0:
        state = np.array([1+0j,0+0j])
    return state

def bit_flip_X(x):
    matrix = np.array([[0, 1],[1,0]])
    x = np.squeeze(x)
    print(x)
    print(matrix)
    probs = np.matmul(matrix,x)
#     prob_1 = probs[0]
#     prob_2 = probs[1]
#     return prob_1, prob_2
    return probs

def hadamard_X(x):
    matrix = np.array([[1, 1],[1,-1]])
    x = np.squeeze(x)
    matrix = matrix/math.sqrt(2)
    print(x)
    print(matrix)
#     state = np.array([x,y])
    probs = np.matmul(matrix,x)
#     prob_1 = probs[0]
#     prob_2 = probs[1]
#     return prob_1, prob_2
    return probs

def hadamard_Y(x):
    matrix = np.array([[1, -1j],[1j,-1]])
    x = np.squeeze(x)
    matrix = matrix/math.sqrt(2)
    print(x)
    print(matrix)
#     state = np.array([x,y])
    probs = np.matmul(matrix,x)
#     prob_1 = probs[0]
#     prob_2 = probs[1]
#     return prob_1, prob_2
    return probs

def nothing(x):
    return x


# def reward(error_t, error_t1):
#     if(error_t> error_t1):
#         reward = 1
#     elif(error_t==error_t1):
#         reward = 0
#     elif(error_t<error_t1):
#         reward = -1
#     return reward

def reward(state, final_state):
    if np.allclose(state,final_state):
        reward = 1
    else:
        reward = 0
    return reward


def projection(state, final_state):
#     e = (abs(state)**2 - abs(final_state)**2)
    e = np.dot(state,final_state)
    e = abs(e)
    e = e**2
    e = np.float32(np.squeeze(e))
    return e


def discounted_reward(r, gamma):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for i in reversed(range(0, r.size)):
        running_add = running_add*gamma + r[i]
        discounted_r[i] = running_add
    return discounted_r
