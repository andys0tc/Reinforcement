import numpy as np
import gym
import random
import intervals as I
import random
import time
from IPython.display import clear_output

env = gym.make('LunarLander-v2')

#state_space_size = env.observation_space.shape[0]
state_space_size = 864
action_space_size = env.action_space.n  #sample of actions
q_table = np.zeros((state_space_size, action_space_size)) # q-table in zeros

num_episodes = 100
max_steps_per_episode = 100




#parameters
learning_rate = 0.1
discount_rate = 0.99
#epsilon greedy trade off
exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.001
exploration_decay_rate = 0.001


def selectIndex(state):
    x_pos = discretize_space_position(state[0])
    y_pos = discretize_space_position(state[1])
    angle = discretize_space_position(state[4])
    legs = dicretize_space_legs(state[7],state[6])
    index = (legs*216)+(angle*36)+(x_pos*6)+y_pos
    return index

def discretize_space_position(pos):
    if pos in I.openclosed(-3,-1):
        return 0
    elif pos in I.openclosed(-1,-0.5):
        return 1
    elif pos in I.openclosed(-0.5,0):
        return 2
    elif pos in I.openclosed(0,0.5):
        return 3
    elif pos in I.openclosed(0.5,1):
        return 4
    elif pos in I.openclosed(1,1.6):
        return 5
    else:
        print("Outliers de posición: %0.03f" %pos)
        return random.randint(0,5)

def discretize_space(angle):
    if angle in I.openclosed(-0.5,0):
        return 0
    elif angle in I.openclosed(0,0.5):
        return 1
    elif angle in I.openclosed(0.5,1):
        return 2
    elif angle in I.openclosed(1,1.5):
        return 3
    else:
        print("Outliers de angulo: %0.03f" % angle)
        return random.randint(0, 3)

def dicretize_space_legs(left,right):
    if (left==0 and right==0):
        return 0
    elif(left==0 and right==1):
        return 1
    elif(left==1 and right==0):
        return 2
    else:
        return 3




rewards_all_episodes = []
state = env.reset()
action = env.action_space.sample()




# Q-learning algorithm
for episode in range(num_episodes):  # cada episodio
    # initialize new episode params
    state = env.reset()             # se resetea el ambiente a valores iniciales
    index_state = selectIndex(state)
    done = False                    # True si el episodio ha terminado
    rewards_current_episode = 0     # recompesa del episodio

    for step in range(max_steps_per_episode):   #Cada tiempo/paso del episodio
        # Exploration-exploitation trade-off
        exploration_rate_threshold = random.uniform(0, 1)
        if exploration_rate_threshold > exploration_rate:  #Explota
            action = np.argmax(q_table[index_state, :])
        else:
            action = env.action_space.sample()              #Explora
        # Take new action
        env.render()
        new_state, reward, done, info = env.step(action)
        new_index_state = selectIndex(new_state)

        # Update Q-table
        q_table[index_state, action] = q_table[index_state, action] * (1 - learning_rate) + learning_rate * (reward + discount_rate * np.max(q_table[new_index_state, :]))
        # Set new state
        # Add new reward
        state = new_state
        rewards_current_episode += reward

        if done ==True:
            break
    print("fin de episodio")
    print(state[0])
    print(state[1])
    print("Reward Final %.03f" %rewards_current_episode)

    # Exploration rate decay
    exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate*episode)
    rewards_all_episodes.append(rewards_current_episode)


# Calculate and print the average reward per thousand episodes
rewards_per_hundred_episodes = np.split(np.array(rewards_all_episodes),num_episodes/10)
count = 100

np.savetxt('qtable.txt',q_table )
print("********Average reward per thousand episodes********\n")
for r in rewards_per_hundred_episodes:
    print(count, ": ", str(sum(r)/10))
    #print(count, ": ", str(sum(r/100)))
    count += 100



