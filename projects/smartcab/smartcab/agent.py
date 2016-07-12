import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

from collections import namedtuple
#import time
from matplotlib import pyplot as plt
import numpy as np

State = namedtuple('State', ['light', 'oncoming', 'left', 'right', 'next'])

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here

        # a trial counter
        self.num_trial = 1

        # hold the reward of every step
        self.rewards = []

        # hold the reward of every step in one trial
        self.trial_rewards = []

        # hold all the states the agent has encountered;
        # will generate on the fly
        # self.states = set()

        # q_table is a mapping of (state, action)-->q_value;
        # will generate on the fly
        self.q_table = {}
        self.initial_q_value = 1


#    def update_q_table(state, action, reward, s_prime, a_prime, alpha, gamma):

#        self.q_table[(state, action)] = (1-alpha)*self.q_table[(state, action)] + alpha*(reward + gamma* self.q_table[(s_prime, a_prime)])



#    def make_q_table(self):
#        q_table = {}
#        actions = Environment.valid_actions
#        states = self.get_total_states()
#        for s in states:
#            for act in actions:
#                q_table[(s, act)] = 99

#        return q_table


#    def make_q_table(self):
#        q_table = {}
#        actions = Environment.valid_actions
#        states = self.get_total_states()
#        for s in states:
#            if s.light == 'red':
#                for act in (None, 'right'):
#                    q_table[(s, act)] = 1
#            else:
#                for act in actions:
#                    q_table[(s, act)] = 1
#        q_table['other'] = -9999999
        
#        return q_table

#    def make_q_table(self):
#        q_table = {}
#        actions = Environment.valid_actions
#        states = self.get_total_states()
#        for s in states:
#            if s.light == 'green':
#                for act in actions:
#                    q_table[(s, act)] = 1
#            else:
#                for act in actions:
#                    if act == 'right' or act == None:
#                        q_table[(s, act)] = 1
#                    else:
#                        q_table[(s, act)] = -9999999

#        return q_table


    def get_states(self):
        states = map(lambda t: t[0], self.q_table.iterkeys())

        return set(states)
        

    def get_action(self, state):
        """
        according to current state, find and return the respective action
        """
        states = self.get_states()
        if state in states:
            # row a die
            epsilon = random.random()

            # if the agent "remembers" current state, 80% of the time we use our policy: q_table to choose an action
            if epsilon-0.8 < 0:  # "exploiting"

                # find all (action, q_value) pairs of the current state
                act_q = filter(lambda (k,v): k[0]==state, self.q_table.iteritems())
                act_q = map(lambda t: (t[0][1], t[1]), act_q)

                # sorted by q values
                act_q.sort(key=lambda t: t[1], reverse=True)

                # after sorted the list, the first pair has the maximum q value
                max_q = act_q[0][1]

                # get all the actions with the same q value
                actions = map(lambda t: t[0], filter(lambda t: t[1] == max_q, act_q))
                action = random.choice(actions)

            # other 20% of the time, we randomly choose an action
            else:  # "exploring"
                action = random.choice(Environment.valid_actions)
                try:
                    self.q_table[(state, action)]
                except KeyError:
                    self.q_table[(state, action)] = self.initial_q_value

        # the agent never encountered current state
        else:
            # we randomly choose an action
            action = random.choice(Environment.valid_actions)

            # put the (state, action) pair into the q_table, so it remember the state next time
            self.q_table[(state, action)] = self.initial_q_value
        
        return action


    def get_a_prime(self, s_prime):
        """
        according to s_prime, find and return the respective a_prime which has the biggest q value.
        """
        states = self.get_states()
        if s_prime in states:
            # find all (action, q_value) pairs of s_prime
            act_q = filter(lambda (k,v): k[0]==s_prime, self.q_table.iteritems())
            act_q = map(lambda t: (t[0][1], t[1]), act_q)

            # sorted by q values
            act_q.sort(key=lambda t: t[1], reverse=True)

            # after sorted the list, the first pair has the maximum q value
            max_q = act_q[0][1]

            # get all the actions with the same q value
            actions = map(lambda t: t[0], filter(lambda t: t[1] == max_q, act_q))
            a_prime = random.choice(actions)

        # s_prime may not in the set of states
        else:
            a_prime = random.choice(Environment.valid_actions)
            self.q_table[(s_prime, a_prime)] = self.initial_q_value

        return a_prime


    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.num_trial += 1

        self.rewards.append(self.trial_rewards)
        self.trial_rewards = []

            
    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        self.state = State(inputs['light'], inputs['oncoming'], inputs['left'], inputs['right'], self.next_waypoint)
            
        # TODO: Select action according to your policy
        action = self.get_action(self.state)

        # Execute action and get reward
        reward = self.env.act(self, action)
        self.trial_rewards.append(reward)
        
        # to get s' and a'
        inputs = self.env.sense(self)
        next_waypoint = self.planner.next_waypoint()
        s_prime = State(inputs['light'], inputs['oncoming'], inputs['left'], inputs['right'], next_waypoint)
        a_prime = self.get_a_prime(s_prime)

        # print(s_prime, a_prime, self.q_table[(s_prime, a_prime)])
        # TODO: Learn policy based on state, action, reward
        alpha = (100-self.num_trial) / 100.0
        # gamma = 0.3

        # alpha = 0.7
        self.q_table[(self.state, action)] = (1-alpha)*self.q_table[(self.state, action)] + alpha*reward

        # self.q_table[(self.state, action)] = (1-alpha)*self.q_table[(self.state, action)] \
        #                                     + alpha*(reward + gamma* self.q_table[(s_prime, a_prime)])

        # print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    # sim.run(n_trials=1)
    sim.run(n_trials=101)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line

    print(' *'*60+'\n\n')
    print('100 trials completed!')

    print(' ^ '*20+'\n\n')
    # for k,v in a.q_table.iteritems():
    #    print(k,v)

    # print('q table length is: '+ str(len(a.q_table)))
    # print(' ^ '*20+'\n\n')

    # for s in a.get_states():
    #    print(s)
    # print('there are: '+ str(len(a.get_states())) + ' states.')
    
    print(' ^ '*20+'\n\n')
    # net_rewards =  map(sum, a.rewards[-30:])
    # print('the last 30 net rewards:')
    # print(net_rewards)

    # num_pos = filter(lambda n: n > 0, net_rewards)
    # print('there are ' + str(len(num_pos)) + ' positive net rewards!')

    # save q table as a text file
    # with open('q_table', 'wb') as outFile:
    #   for k, v in a.q_table.iteritems():
    #       outFile.write(str(k)+','+str(v)+'\n')

    # dump q table as pickle file
    # with open('q_table.pkl', 'wb') as outFile:
    #   pickle.dump(a.q_table, outFile)
        
    # print('q_table saved!')

    
    #c = 0
    #for k, v in a.q_table.iteritems():
    #  if v != 99:
    #      print(k, v)
    #      c += 1
    #print('effective states are: '+str(c))

    #print('the last 10 trials rewards:')
    #for s,l in zip(a.trial_step[-10:], a.rewards[-10:]):
    #    print(s,l)
    print(map(sum, a.rewards[-10:]))

    #for i,v in enumerate(a.rewards):
    #    if v == []:
    #        print i
    a.rewards = a.rewards[1:]
    net_rewards = map(sum, a.rewards)

    mean_rewards = map(np.mean, a.rewards)

    # std_rewards = map(np.std, a.rewards)

    num_neg_r = [filter(lambda n: n<0, l) for l in a.rewards]
    num_neg_r = map(len, num_neg_r)

    num_trial_steps = map(len, a.rewards)

    last_rewards = map(lambda l: l[-1], a.rewards[1:])
    success_count = len(filter(lambda e: e > 8, last_rewards))
    success_count += 1
    print('successful delivery: '+str(success_count))


    # trend of net rewards and trial steps over 100 trials
    plt.figure(figsize=(16,9), dpi=75)
    plt.scatter(range(1, 101), net_rewards)
    w1, w0 = np.polyfit(range(1, 101), net_rewards, 1)
    y = w1*np.arange(1, 101) + w0
    plt.plot(range(1, 101), y, color='blue')
    
    #plt.scatter(range(1, 101),num_trial_steps,c='red')
    #w1, w0 = np.polyfit(range(1, 101), num_trial_steps, 1)
    #y = w1*np.arange(1, 101) + w0
    #plt.plot(range(1, 101), y, color='red')

    plt.title('net rewards over trials')
    plt.legend(['expected net rewards'])
    plt.xlabel('number of trials')
    plt.ylabel('net rewards')
    plt.show()

    # trend on number of negative rewards over 100 trials
    plt.figure(figsize=(16,9), dpi=75)
    plt.scatter(range(1, 101), num_neg_r,c='r')
    w1, w0 = np.polyfit(range(1, 101), num_neg_r, 1)
    y = w1*np.arange(1, 101) + w0
    plt.plot(range(1, 101), y, color='red')
    plt.legend(['number of negative rewards'])
    plt.ylabel('number of negative rewards')
    plt.xlabel('number of trials')
    plt.show()


    # number of time steps over 100 trials
    plt.figure(figsize=(16,9), dpi=75)
    plt.scatter(range(1, 101),num_trial_steps,c='green')
    w1, w0 = np.polyfit(range(1, 101), num_trial_steps, 1)
    y = w1*np.arange(1, 101) + w0
    plt.plot(range(1, 101), y, color='green')
    plt.xlabel('number of trials')
    plt.ylabel('number of time steps')
    plt.legend(['trial steps'])
    plt.show()

    # mean of rewards over 100 trials
    plt.figure(figsize=(16, 9), dpi=75)
    plt.scatter(range(1, 101),mean_rewards,c='cyan')
    w1, w0 = np.polyfit(range(1, 101), mean_rewards, 1)
    y = w1*np.arange(1, 101) + w0
    plt.plot(range(1, 101), y, color='cyan')
    plt.xlabel('number of trials')
    plt.ylabel('mean of trial rewards')
    plt.legend(['mean_rewards'])
    plt.show()
    
    print(' *'*60+'\n\n')

if __name__ == '__main__':
    run()
