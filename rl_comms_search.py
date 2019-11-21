
#this should be the thing, right?
from __future__ import division

import gym
import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.layers as layers
import matplotlib.pyplot as plt
import threading
import time
import scipy.signal as signal
import os
import GroupLock
import multiprocessing
# get_ipython().run_line_magic('matplotlib', 'inline')
import mapf_gym_rl_search as mapf_gym
import pickle
# import imageio
from ACNet_rl_search2 import ACNet
# from ACNet_seperate_1hot_hard_coded_simple import ACNet

from tensorflow.python.client import device_lib
# dev_list = device_lib.list_local_devices()
# print(dev_list)
# assert len(dev_list) > 1


# ### Helper Functions

# In[3]:


def make_gif(images, fname, duration=2, true_image=False,salience=False,salIMGS=None):
    import moviepy.editor as mpy

    def make_frame(t):
        try:
            x = images[int(len(images)/duration*t)]
        except:
            x = images[-1]

        if true_image:
            return x.astype(np.uint8)
        else:
            return ((x+1)/2*255).astype(np.uint8)

    def make_mask(t):
        try:
            x = salIMGS[int(len(salIMGS)/duration*t)]
        except:
            x = salIMGS[-1]
        return x

    clip = mpy.VideoClip(make_frame, duration=duration)
    if salience == True:
        mask = mpy.VideoClip(make_mask, ismask=True,duration= duration)
        clipB = clip.set_mask(mask)
        clipB = clip.set_opacity(0)
        mask = mask.set_opacity(0.1)
        mask.write_gif(fname, fps = len(images) / duration,verbose=False)
    else:
        clip.write_gif(fname, fps = len(images) / duration,verbose=False)



# Copies one set of variables to another.
# Used to set worker network parameters to those of global network.
def update_target_graph(from_scope,to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

def discount(x, gamma):
    return signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

def good_discount(x, gamma):
    return discount(x,gamma)
#     positive = np.clip(x,0,None)
#     negative = np.clip(x,None,0)
#     return signal.lfilter([1], [1, -gamma], positive[::-1], axis=0)[::-1]+negative


# ## Worker Agent

# In[4]:


class Worker:
    def __init__(self, game, metaAgentID, workerID, a_size, groupLock):
        self.workerID = workerID
        self.env = game
        self.metaAgentID = metaAgentID
        self.name = "worker_"+str(workerID)
        self.agentID = ((workerID-1) % num_workers) + 1 
        self.groupLock = groupLock

        self.nextGIF = episode_count # For GIFs output
        #Create the local copy of the network and the tensorflow op to copy global parameters to local network
        self.local_AC = ACNet(self.name,a_size, self.env.num_agents, trainer_A,trainer_C,True,GRID_SIZE,GLOBAL_NET_SCOPE)
#         self.copy_weights = self.local_AC.homogenize_weights
        self.pull_global = update_target_graph(GLOBAL_NET_SCOPE, self.name)

    def train(self, rollout, sess, gamma, bootstrap_values):
        global episode_count

       
       #0: s_actor[0]
       #1: s_actor[1]
       #2: s_critic
       #3: a
       #4: action_buffer[self.metaAgentID][other_agentID - 1]
       #5: msg_binary
       #6: r
       #7: s1_actor
       #8: d
       #9: v
       #10: value_buffer[self.metaAgentID][other_agentID - 1]
       

        rollout = np.array(rollout)
        observations_actor = rollout[:,0]
        msg_mats = rollout[:,1]
        observations_critic = rollout[:,2]
        actions = rollout[:,3]
        actions_other = rollout[:,4]
        msgs_binary = rollout[:,5]
        rewards = rollout[:,6]
        done    = rollout[:,8]
        values = rollout[:,9]
        values_other = rollout[:,10]



        # Here we take the rewards and values from the rollout, and use them to 
        # generate the advantage and discounted returns. (With bootstrapping)
        # The advantage function uses "Generalized Advantage Estimation"
        # if the episode had finished, append an additional zero to values (so that we can still use entire episode despite having to shift msg_advantages)
        if done[-1]:
            self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_values[0]] + [0])
            self.value_plus = np.asarray(values.tolist() + [bootstrap_values[0]] + [0])
            self.value_plus_other = np.asarray(values_other.tolist() + [bootstrap_values[1]] + [0])
            rewards = np.concatenate([rewards,[0]])
        else:
            self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_values[0]])
            self.value_plus = np.asarray(values.tolist() + [bootstrap_values[0]])
            self.value_plus_other = np.asarray(values_other.tolist() + [bootstrap_values[1]])

        discounted_rewards = discount(self.rewards_plus,gamma)[:-1]
        action_advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        action_advantages = good_discount(action_advantages,gamma)[:-1]

        action_advantages_other = rewards + gamma * self.value_plus_other[1:] - self.value_plus_other[:-1]
        action_advantages_other = good_discount(action_advantages_other,gamma)
        msg_advantages = action_advantages_other[1:]
        
        num_samples = min(EPISODE_SAMPLES,len(action_advantages))
        sampleInd = np.sort(np.random.choice(action_advantages.shape[0], size=(num_samples,), replace=False))

        # print('observations_critic[:len(action_advantages).shape: ',observations_critic[:len(action_advantages)].shape)
        rnn_state_actor = self.local_AC.state_init_actor

        feed_dict = {
            global_step:episode_count,
            self.local_AC.target_v:np.stack(discounted_rewards[:len(action_advantages)]),
            self.local_AC.actor_inputs:np.stack(observations_actor[:len(action_advantages)]),
            self.local_AC.critic_inputs:np.stack(observations_critic[:len(action_advantages)]),
            self.local_AC.msg_inputs:np.stack(msg_mats[:len(action_advantages)]),
            self.local_AC.msgs_binary:np.stack(msgs_binary[:len(action_advantages)]),
            self.local_AC.actions:actions[:len(action_advantages)],
            self.local_AC.actions_other:actions_other[:len(action_advantages)],
            self.local_AC.action_advantages:action_advantages,
            self.local_AC.msg_advantages:msg_advantages,
            self.local_AC.state_in_actor:rnn_state_actor
        }
        
        v_l,p_l,msg_l,e_l,e_msg,gn_a,gn_c,vn_a,vn_c,_,_ = sess.run([
            self.local_AC.c_loss,
            self.local_AC.policy_loss,
            self.local_AC.msg_loss,
            self.local_AC.entropy,
            self.local_AC.msg_entropy,
            self.local_AC.a_grad_norms,
            self.local_AC.c_grad_norms,
            self.local_AC.a_var_norms,
            self.local_AC.c_var_norms,
            self.local_AC.apply_a_grads,
            self.local_AC.apply_c_grads],
            feed_dict=feed_dict)


        return v_l/len(rollout), p_l/len(rollout), msg_l/len(rollout), e_l/len(rollout), e_msg/len(rollout), gn_a, gn_c, vn_a, vn_c

    def shouldRun(self, coord, episode_count):
        if TRAINING:
            return (not coord.should_stop())
        else:
            return (episode_count < NUM_EXPS)

    def synchronize(self):
        #handy thing for keeping track of which to release and acquire
        if(not hasattr(self,"lock_bool")):
            self.lock_bool=False
        self.groupLock.release(int(self.lock_bool),self.name)
        self.groupLock.acquire(int(not self.lock_bool),self.name)
        self.lock_bool=not self.lock_bool
   
    # def sample_msg_1hot(self, P_msg):
    #     msg = np.random.choice(len(P_msg.flatten()),p=P_msg.flatten())
    #     msg_1hot = np.zeros(len(P_msg.flatten()))
    #     msg_1hot[msg] = 1

    #     return msg_1hot

    def sample_msg_binary(self, P_msg):
        rands = np.random.uniform(size = P_msg.shape)
        msg_binary = (P_msg > rands)
        return msg_binary.flatten()


    
    def choose_action_msg(self, s_actor, s_critic, rnn_state_actor, validActions, training):
        global action_buffer

        if self.agentID == 1:
            other_agentID = 2
        elif self.agentID == 2:
            other_agentID = 1

        a_dist, P_msg, rnn_state_actor, msg_entropy = sess.run([self.local_AC.policy,
                                                self.local_AC.P_msg,
                                                self.local_AC.state_out_actor,
                                                self.local_AC.msg_entropy], 
                                     feed_dict={self.local_AC.actor_inputs:[s_actor[0]],
                                                self.local_AC.msg_inputs:[s_actor[1]],
                                                self.local_AC.state_in_actor:rnn_state_actor})

        # print('P_msg: ', P_msg)
        msg_binary = self.sample_msg_binary(P_msg)
        # assert np.sum(msg_1hot) == 1
        # print('msg_binary: ', msg_binary)
        # print('msg_entropy: ',msg_entropy)

        

        if(not (np.argmax(a_dist.flatten()) in validActions)):
            invalid = True
        else:
            invalid = False



        train_valid = np.zeros(a_size)
        train_valid[validActions] = 1
        valid_dist = np.array([a_dist[0,validActions]])
        valid_dist /= np.sum(valid_dist)

        if TRAINING:
            # if (pred_blocking.flatten()[0] < 0.5) == blocking:
            #     wrong_blocking += 1
            # a           = validActions[ np.random.choice(range(valid_dist.shape[1]),p=valid_dist.ravel()) ]
            a = np.random.choice(a_size, p = a_dist.flatten())
            # train_val   = 1.
        else:
            a         = np.argmax(a_dist.flatten())
            if a not in validActions or not GREEDY:
                a     = validActions[ np.random.choice(range(valid_dist.shape[1]),p=valid_dist.ravel()) ]
            # train_val = 1.

        action_buffer[self.metaAgentID][self.agentID - 1] = a



        self.synchronize()
        # print('action_buffer: ', action_buffer)
        # print('self.metaAgentID: ',self.metaAgentID)

        # print('action_buffer[self.metaAgentID][other_agentID - 1]: ',action_buffer[self.metaAgentID][other_agentID - 1])


        v = sess.run(self.local_AC.value, 
                    feed_dict={self.local_AC.critic_inputs:[s_critic],
                    self.local_AC.actions_other:[action_buffer[self.metaAgentID][other_agentID - 1]]
                    })[0,0]
        
        return a,msg_binary,v,rnn_state_actor,invalid

    def work(self,max_episode_length,gamma,sess,coord,saver):
        global episode_count, swarm_reward, episode_rewards, episode_lengths, episode_mean_values, episode_invalid_ops,episode_wrong_blocking, value_buffer, action_buffer#, episode_invalid_goals
        total_steps, i_buf = 0, 0
        episode_buffers = [ [] for _ in range(NUM_BUFFERS) ]
        s1Values = [np.zeros(self.env.num_agents) for _ in range(NUM_BUFFERS)]

        if self.agentID == 1:
            other_agentID = 2
        elif self.agentID == 2:
            other_agentID = 1

        with sess.as_default(), sess.graph.as_default():
            while self.shouldRun(coord, episode_count):
                sess.run(self.pull_global)
#                 sess.run(self.copy_weights)

                value_buffer[self.metaAgentID]  = np.zeros(self.env.num_agents)
                action_buffer[self.metaAgentID] = np.zeros(self.env.num_agents)
                episode_buffer, episode_values = [], []
                episode_reward = episode_step_count = episode_inv_count = 0
                d = False

                # Initial state from the environment
#                 validActions, on_goal = self.env._reset(self.agentID)
                validActions, _,blocking = self.env._reset(self.agentID)
                s_actor                     = self.env._observe_actor(self.agentID)
                s_critic               = self.env._observe_critic(self.agentID)
                rnn_state_actor             = self.local_AC.state_init_actor
                RewardNb = 0 
               

                saveGIF = False
                if OUTPUT_GIFS and self.workerID == 1 and ((not TRAINING) or (episode_count >= self.nextGIF)):
                    saveGIF = True
                    self.nextGIF += 64
                    GIF_episode = int(episode_count)
                    episode_frames = [ self.env._render(mode='rgb_array',screen_height=200,screen_width=200) ]

                self.synchronize() # synchronize starting time of the threads

                # reset swarm_reward (for tensorboard)
                swarm_reward[self.metaAgentID] = 0

                while (not self.env.finished): # Give me something!
                    #Take an action using probabilities from policy network output.
                   #a,msg_binary,v,rnn_state_actor,invalid
                    a,msg_binary,v,rnn_state_actor,invalid = self.choose_action_msg(s_actor,s_critic,rnn_state_actor, validActions, TRAINING)

                    if invalid:
                        episode_inv_count += 1

                    train_valid = np.zeros(a_size)
                    train_valid[validActions] = 1

                    #place this state's value in the value buffer
                    value_buffer[self.metaAgentID][self.agentID - 1] = v

                   


                    _, _, _, _, _,blocking,_ = self.env._step((self.agentID, a, msg_binary),episode=episode_count)

                    self.synchronize() # synchronize threads

                    r = self.env.compute_reward()  #get shared reward after all agents have stepped

                    # Get common observation for all agents after all individual actions have been performed
                    s1_actor           = self.env._observe_actor(self.agentID)
                    s1_critic      = self.env._observe_critic(self.agentID)
                    validActions = self.env._listNextValidActions(self.agentID, a,episode=episode_count)
                    d            = self.env.finished

                    if saveGIF:
                        episode_frames.append(self.env._render(mode='rgb_array',screen_width=200,screen_height=200))

                    episode_buffer.append([s_actor[0],s_actor[1],s_critic,a,action_buffer[self.metaAgentID][other_agentID - 1],msg_binary,r,s1_actor,d,v,value_buffer[self.metaAgentID][other_agentID - 1]])
#                     episode_buffer.append([s,a,r,s1,d,v[0,0],train_valid,train_val])
                    episode_values.append(v)
                    episode_reward += r
                    s_actor = s1_actor
                    s_critic = s1_critic
                    total_steps += 1
#                     steps_on_goal += int(on_goal)
#                     on_goal = on_goal1
                    episode_step_count += 1

                    if r>0:
                        RewardNb += 1
                    if d == True:
#                         s1 = s  #Oh yeah!! We are done, we did it!!!
                        print('\n{} Goodbye World. We did it!'.format(episode_step_count), end='\n')

                    # If the episode hasn't ended, but the experience buffer is full, then we
                    # make an update step using that experience rollout.
                    if TRAINING and (len(episode_buffer) % EXPERIENCE_BUFFER_SIZE == 0 or d):
                        # Since we don't know what the true final return is, we "bootstrap" from our current value estimation.
                        


                        if not d:
                            #take 1 additional step, the values will become the s1Values
                            
                            _, _,v,_,_ = self.choose_action_msg(s_actor, s_critic, rnn_state_actor, validActions, TRAINING)
                            #place this state's value in the value buffer
                            value_buffer[self.metaAgentID][self.agentID - 1] = v

                            # if TRAINING:
                            #     if (pred_blocking.flatten()[0] < 0.5) == blocking:
                            #         wrong_blocking += 1

                            self.synchronize() # synchronize starting time of the threads


                            
                            s1Values[i_buf][0] = value_buffer[self.metaAgentID][self.agentID - 1]
                            s1Values[i_buf][1] = value_buffer[self.metaAgentID][other_agentID - 1]
                            
                        #if the episode is done, the bootstrap values are 0
                        else:
                            s1Values[i_buf][:] = 0



                        #put the appropriate stuff in episode_buffer into episode_buffers
                        if len(episode_buffer) >= EXPERIENCE_BUFFER_SIZE:
                            episode_buffers[i_buf] = episode_buffer[-EXPERIENCE_BUFFER_SIZE:]
                        else:
                            episode_buffers[i_buf] = episode_buffer[:]


                        if (episode_count-EPISODE_START) < NUM_BUFFERS:
                            i_rand = np.random.randint(i_buf+1)
                        else:
                            i_rand = np.random.randint(NUM_BUFFERS)
                            tmp = np.array(episode_buffers[i_rand])
                            while tmp.shape[0] == 0:
                                i_rand = np.random.randint(NUM_BUFFERS)
                                tmp = np.array(episode_buffers[i_rand])
                       #v_l, p_l, msg_l, e_l, e_msg, gn_a, gn_c, vn_a, vn_c
                        v_l, p_l, msg_l, e_l, e_msg, gn_a, gn_c, vn_a, vn_c = self.train(episode_buffers[i_rand],sess,gamma,s1Values[i_rand])

                        i_buf = (i_buf + 1) % NUM_BUFFERS
                        episode_buffers[i_buf] = []

                       # sess.run(self.pull_global)
#                         sess.run(self.copy_weights)

                    self.synchronize()
                    sess.run(self.pull_global)
                    if episode_step_count >= max_episode_length or d:
                        break

                episode_lengths[self.metaAgentID].append(episode_step_count)
                episode_mean_values[self.metaAgentID].append(np.nanmean(episode_values))
                episode_invalid_ops[self.metaAgentID].append(episode_inv_count)
                # episode_wrong_blocking[self.metaAgentID].append(wrong_blocking)
                # episode_msg_entropies[self.metaAgentID].append(e_msg)
                # episode_msg_losses[self.metaAgentID].append(msg_l)
#                 episode_steps_on_goal[self.metaAgentID].append(steps_on_goal)

                # Periodically save gifs of episodes, model parameters, and summary statistics.
                if episode_count % EXPERIENCE_BUFFER_SIZE == 0 and printQ:
                    print('                                                                                   ', end='\r')
                    print('{} Episode terminated ({},{})'.format(episode_count, self.agentID, RewardNb), end='\r')

                swarm_reward[self.metaAgentID] += episode_reward

                self.synchronize()

                episode_rewards[self.metaAgentID].append(swarm_reward[self.metaAgentID])

                if not TRAINING:
                    mutex.acquire()
                    if episode_count < NUM_EXPS:
                        plan_durations[episode_count] = episode_step_count
                    if self.workerID == 1:
                        episode_count += 1
                        print('({}) Thread {}: {} steps, {:.2f} reward ({} invalids).'.format(episode_count, self.workerID, episode_step_count, episode_reward, episode_inv_count))
                    GIF_episode = int(episode_count)
#                    saveGIF &= (episode_step_count < max_episode_length)
                    mutex.release()
                else:
                    episode_count += 1. / num_workers

                    if episode_count % SUMMARY_WINDOW == 0:
                        if episode_count % 500 == 0:
                            saver.save(sess, model_path+'/model-'+str(int(episode_count))+'.cptk')
                            print ('Saved Model', end='\r')
                        SL = SUMMARY_WINDOW * num_workers
                        mean_reward = np.mean(episode_rewards[self.metaAgentID][-SL:])
                        mean_length = np.mean(episode_lengths[self.metaAgentID][-SL:])
                        mean_value = np.mean(episode_mean_values[self.metaAgentID][-SL:])
                        mean_invalid = np.mean(episode_invalid_ops[self.metaAgentID][-SL:])
                        # mean_wrong_blocking = np.mean(episode_wrong_blocking[self.metaAgentID][-SL:])
                        # mean_msg_entropy = np.mean(episode_msg_entropies[self.metaAgentID][-SL:])
                        # mean_msg_loss = np.mean(episode_msg_losses[self.metaAgentID][-SL:])
#                         mean_steps_on_goal = np.mean(episode_steps_on_goal[self.metaAgentID][-SL:])
                        current_learning_rate_actor = sess.run(lr_a,feed_dict={global_step:episode_count})
                        current_learning_rate_critic = sess.run(lr_c,feed_dict={global_step:episode_count})


                        summary = tf.Summary()
                        summary.value.add(tag='Perf/Actor Learning Rate',simple_value=current_learning_rate_actor)
                        summary.value.add(tag='Perf/Critic Learning Rate',simple_value=current_learning_rate_critic)
                        summary.value.add(tag='Perf/Reward', simple_value=mean_reward)
                        summary.value.add(tag='Perf/Length', simple_value=mean_length)
                        summary.value.add(tag='Perf/Valid Rate', simple_value=(mean_length-mean_invalid)/mean_length)
                        # summary.value.add(tag='Perf/Blocking Prediction Accuracy', simple_value=(mean_length-mean_wrong_blocking)/mean_length)
#                         summary.value.add(tag='Perf/On_Goal Rate', simple_value=mean_steps_on_goal/mean_length)

                        summary.value.add(tag='Losses/Value Loss', simple_value=v_l)
                        summary.value.add(tag='Losses/Policy Loss', simple_value=p_l)
                        # summary.value.add(tag='Losses/Blocked Loss', simple_value=b_l)
                        # summary.value.add(tag='Losses/Valid Loss', simple_value=valid_l)
                        summary.value.add(tag='Losses/Actor Grad Norm', simple_value=gn_a)
                        summary.value.add(tag='Losses/Critic Grad Norm', simple_value=gn_c)
                        summary.value.add(tag='Losses/Actor Var Norm', simple_value=vn_a)
                        summary.value.add(tag='Losses/Critic Var Norm', simple_value=vn_c)
                        summary.value.add(tag='Losses/Message Entropy', simple_value=e_msg)
                        summary.value.add(tag='Losses/Message Loss', simple_value=msg_l)
                        global_summary.add_summary(summary, int(episode_count))

                        global_summary.flush()

                        if printQ:
                            print('{} Tensorboard updated ({})'.format(episode_count, self.workerID), end='\r')

                if saveGIF:
                    # Dump episode frames for external gif generation (otherwise, makes the jupyter kernel crash)
                    time_per_step = 0.1
                    images = np.array(episode_frames)
                    if TRAINING:
                        gif_creation = lambda: make_gif(images, '{}/episode_{:d}_{:d}_{:.1f}.gif'.format(gifs_path,GIF_episode,episode_step_count,swarm_reward[self.metaAgentID]), duration=len(images)*time_per_step,true_image=True,salience=False)
                        threading.Thread(target=(gif_creation)).start()
                    else:
                        make_gif(images, '{}/episode_{:d}_{:d}.gif'.format(gifs_path,GIF_episode,episode_step_count), duration=len(images)*time_per_step,true_image=True,salience=False)
                if SAVE_EPISODE_BUFFER:
                    with open('gifs3D/episode_{}.dat'.format(GIF_episode), 'wb') as file:
                        pickle.dump(episode_buffer, file)


# ## Training

# In[5]:


# Learning parameters
max_episode_length     = 256
episode_count          = 0
EPISODE_START          = episode_count
gamma                  = .95 # discount rate for advantage estimation and reward discounting
#moved network parameters to ACNet.py
EXPERIENCE_BUFFER_SIZE = 128
GRID_SIZE              = 10
OBSTACLE_DENSITY       = 0 #range of densities
DIAG_MVMT              = False # Diagonal movements allowed?
a_size                 = 5 + int(DIAG_MVMT)*4
SUMMARY_WINDOW         = 10
NUM_META_AGENTS        = 8
NUM_THREADS            = 2 #int(multiprocessing.cpu_count() / (2 * NUM_META_AGENTS))
NUM_BUFFERS            = 1 # NO EXPERIENCE REPLAY int(NUM_THREADS / 2)
EPISODE_SAMPLES        = EXPERIENCE_BUFFER_SIZE # 64
LR_actor               = 2.e-5 #8.e-5 / NUM_THREADS # default: 1e-5
LR_critic              = 2.e-5
ADAPT_LR               = True
ADAPT_COEFF            = 5.e-5 #the coefficient A in LR_Q/sqrt(A*steps+1) for calculating LR
load_model             = False
RESET_TRAINER          = False
model_path             = './model_rl_comms_search2_AAAI'
gifs_path              = './gifs_rl_comms_search2_AAAI'
train_path             = 'train_rl_comms_search2_AAAI'
GLOBAL_NET_SCOPE       = 'global'

# Simulation options
FULL_HELP              = False
RANDOMIZED_PLANS       = False
OUTPUT_GIFS            = False
SAVE_EPISODE_BUFFER    = False

# Testing
TRAINING               = True
GREEDY                 = False
NUM_EXPS               = 100
MODEL_NUMBER           = 40000

# Shared arrays for tensorboard
episode_rewards        = [ [] for _ in range(NUM_META_AGENTS) ]
episode_lengths        = [ [] for _ in range(NUM_META_AGENTS) ]
episode_mean_values    = [ [] for _ in range(NUM_META_AGENTS) ]
episode_invalid_ops    = [ [] for _ in range(NUM_META_AGENTS) ]
# episode_wrong_blocking = [ [] for _ in range(NUM_META_AGENTS) ]
value_buffer           = [ [] for _ in range(NUM_META_AGENTS) ]
action_buffer          = [ [] for _ in range(NUM_META_AGENTS) ]
# episode_steps_on_goal  = [ [] for _ in range(NUM_META_AGENTS) ]
printQ                 = False # (for headless)
swarm_reward           = [0]*NUM_META_AGENTS


# In[6]:


tf.reset_default_graph()
print("Hello World")
print('train path (ensure this is correct!!!): ',train_path)
if not os.path.exists(model_path):
    os.makedirs(model_path)
config = tf.ConfigProto(allow_soft_placement = True)
config.gpu_options.allow_growth=True

if not TRAINING:
    plan_durations = np.array([0 for _ in range(NUM_EXPS)])
    mutex = threading.Lock()
    gifs_path += '_tests'
    if SAVE_EPISODE_BUFFER and not os.path.exists('gifs3D'):
        os.makedirs('gifs3D')

#Create a directory to save episode playback gifs to
if OUTPUT_GIFS and not os.path.exists(gifs_path):
    os.makedirs(gifs_path)

with tf.device("/gpu:0"):
    master_network = ACNet(GLOBAL_NET_SCOPE,a_size, NUM_THREADS, None,     None,    False,   GRID_SIZE,GLOBAL_NET_SCOPE) 

    global_step = tf.placeholder(tf.float32)
    if ADAPT_LR:
        #computes LR_Q/sqrt(ADAPT_COEFF*steps+1)
        #we need the +1 so that lr at step 0 is defined
        lr_a=tf.divide(tf.constant(LR_actor),tf.sqrt(tf.add(1.,tf.multiply(tf.constant(ADAPT_COEFF),global_step))))
        lr_c=tf.divide(tf.constant(LR_critic),tf.sqrt(tf.add(1.,tf.multiply(tf.constant(ADAPT_COEFF),global_step))))
    else:
        lr_a=tf.constant(LR_actor)
        lr_c=tf.constant(LR_critic)
    trainer_A = tf.contrib.opt.NadamOptimizer(learning_rate=lr_a, use_locking=True)
    trainer_C = tf.contrib.opt.NadamOptimizer(learning_rate=lr_c, use_locking=True)

    if TRAINING:
        num_workers = NUM_THREADS # Set workers # = # of available CPU threads
    else:
        num_workers = NUM_THREADS
        NUM_META_AGENTS = 1
    
    gameEnvs, workers, groupLocks = [], [], []
    n=1#counter of total number of agents (for naming)
    for ma in range(NUM_META_AGENTS):
#         num_agents=((ma%4)+1)*2
#         print(num_agents)
#         num_workers=num_agents
        num_agents=NUM_THREADS
        gameEnv = mapf_gym.MAPFEnv(num_agents=num_agents, DIAGONAL_MOVEMENT=DIAG_MVMT, SIZE=GRID_SIZE, 
                                   PROB=OBSTACLE_DENSITY, FULL_HELP=FULL_HELP)
        gameEnvs.append(gameEnv)

        # Create groupLock
        workerNames = ["worker_"+str(i) for i in range(n,n+num_workers)]
        groupLock = GroupLock.GroupLock([workerNames,workerNames])
        groupLocks.append(groupLock)

        # Create worker classes
        workersTmp = []
        for i in range(ma*num_workers+1,(ma+1)*num_workers+1):
            workersTmp.append(Worker(gameEnv,ma,n,a_size,groupLock))
            n+=1
        workers.append(workersTmp)

    global_summary = tf.summary.FileWriter(train_path)
    saver = tf.train.Saver(max_to_keep=2)

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        if load_model == True:
            print ('Loading Model...')
            if not TRAINING:
                with open(model_path+'/checkpoint', 'w') as file:
                    file.write('model_checkpoint_path: "model-{}.cptk"'.format(MODEL_NUMBER))
                    file.close()
            ckpt = tf.train.get_checkpoint_state(model_path)
            p=ckpt.model_checkpoint_path
            p=p[p.find('-')+1:]
            p=p[:p.find('.')]
            episode_count=int(p)
            saver.restore(sess,ckpt.model_checkpoint_path)
            print("episode_count set to ",episode_count)
            if RESET_TRAINER:
                trainer_A = tf.contrib.opt.NadamOptimizer(learning_rate=lr_a, use_locking=True)
                trainer_C = tf.contrib.opt.NadamOptimizer(learning_rate=lr_c, use_locking=True)

        # This is where the asynchronous magic happens.
        # Start the "work" process for each worker in a separate thread.
        worker_threads = []
        for ma in range(NUM_META_AGENTS):
            for worker in workers[ma]:
                groupLocks[ma].acquire(0,worker.name) # synchronize starting time of the threads
                worker_work = lambda: worker.work(max_episode_length,gamma,sess,coord,saver)
                print("Starting worker " + str(worker.workerID))
                t = threading.Thread(target=(worker_work))
                t.start()
                worker_threads.append(t)
        coord.join(worker_threads)

if not TRAINING:
    print([np.mean(plan_durations), np.sqrt(np.var(plan_durations)), np.mean(np.asarray(plan_durations < max_episode_length, dtype=float))])

