
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
import mapf_gym_diff_comms as mapf_gym
import pickle
from ACNet_diff_comms_recon_error import ACNet
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
    def __init__(self, game, metaAgentID, a_size, ):#groupLock):
        self.workerID = metaAgentID
        self.env = game
        self.metaAgentID = metaAgentID
        self.name = "worker_"+str(self.workerID)
        # self.groupLock = groupLock

        self.nextGIF = episode_count # For GIFs output
        #Create the local copy of the network and the tensorflow op to copy global parameters to local network
        # scope, a_size, num_agents, trainer_A,trainer_C,TRAINING,GRID_SIZE,GLOBAL_NET_SCOPE
        self.local_AC = ACNet(self.name,a_size, self.env.num_agents,trainer_A,trainer_C,True,GRID_SIZE,GLOBAL_NET_SCOPE)
#         self.copy_weights = self.local_AC.homogenize_weights
        self.pull_global = update_target_graph(GLOBAL_NET_SCOPE, self.name)
        
        # logging stuff for tensorboard
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.episode_invalid_ops = []

    def train(self, rollout, sess, gamma, bootstrap_values):
        global episode_count

        

        #0: s_actors,
        #1: s_critic,
        #2: actions,
        #3: r,
        #4: d,
        #5: v,
        #6: recon_error
        

        # observations = rollout[:,0]
        rollout = np.array(rollout)

        observations_actors = rollout[:,0]
        observations_critics = rollout[:,1]
        actions = np.stack(rollout[:,2])
        rewards = rollout[:,3]
        dones = rollout[:,4]
        values = np.stack(rollout[:,5])
        recon_error = np.stack(rollout[:,6])

        # @Renbo: get reconstruction error back out of rollout (same as episode buffer in work fucntion)
        
        # Here we take the rewards and values from the rollout, and use them to 
        # generate the advantage and discounted returns. (With bootstrapping)
        # The advantage function uses "Generalized Advantage Estimation"
        # if the episode had finished, append an additional zero to values (so that we can still use entire episode despite having to shift msg_advantages)
        
        advantages_all = np.zeros((len(rollout),self.env.num_agents))
        discounted_rewards_all = np.zeros((len(rollout),self.env.num_agents))
        for i in range(self.env.num_agents):
            self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_values[i]])
            discounted_rewards = discount(self.rewards_plus,gamma)[:-1]
            self.value_plus = np.asarray(values[:,i].tolist() + [bootstrap_values[i]])
            advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
            advantages = good_discount(advantages,gamma)

            discounted_rewards_all[:,i] = discounted_rewards
            advantages_all[:,i] = advantages
        
        msgs_in = self.local_AC.msgs_init
        # @Renbo: add in reconstruction error as input to ACNet diff_comms
        feed_dict = {
            global_step:episode_count,
            self.local_AC.target_v:np.stack(discounted_rewards_all),
            self.local_AC.actor_inputs:np.stack(observations_actors),
            self.local_AC.critic_inputs:np.stack(observations_critics),
            self.local_AC.actions:actions,
            self.local_AC.advantages:advantages_all,
            self.local_AC.msgs_in:msgs_in,
            self.local_AC.reconstruction_error:recon_error
        }
        
        v_l,p_l,e_l,gn_a,gn_c,vn_a,vn_c,_,_ = sess.run([
            self.local_AC.c_loss,
            self.local_AC.policy_loss,
            
            self.local_AC.entropy,
            
            self.local_AC.a_grad_norms,
            self.local_AC.c_grad_norms,
            self.local_AC.a_var_norms,
            self.local_AC.c_var_norms,
            
            self.local_AC.apply_a_grads,
            self.local_AC.apply_c_grads],
            feed_dict=feed_dict)


        return v_l/len(rollout), p_l/len(rollout), e_l/len(rollout), gn_a, gn_c, vn_a, vn_c

    def shouldRun(self, coord, episode_count):
        if TRAINING:
            return (not coord.should_stop())
        else:
            return (episode_count < NUM_EXPS)

    # def synchronize(self):
    #     #handy thing for keeping track of which to release and acquire
    #     if(not hasattr(self,"lock_bool")):
    #         self.lock_bool=False
    #     self.groupLock.release(int(self.lock_bool),self.name)
    #     self.groupLock.acquire(int(not self.lock_bool),self.name)
    #     self.lock_bool=not self.lock_bool

    
    def choose_actions_msgs(self, s_actors, s_critics, msgs_in, validActions, training):
        '''
        Chooses 2 actions (1 for EACH agent) and gets 2 messages (again 1 for each agent)
        Computes value for EACH agent's state

        Takes as input:
            s_actors:  concatenation of both actors' observations
            s_critics: concatenation of both critics' observations
        '''

        assert len(s_actors) == self.env.num_agents
        assert len(s_critics) == self.env.num_agents

        #a_dist should be of shape [# agents, a_size]
        policies, msgs_out = sess.run([self.local_AC.policies,
                                   self.local_AC.msgs_out], 
                                   feed_dict={self.local_AC.actor_inputs:[np.stack(s_actors)],
                                              self.local_AC.msgs_in:msgs_in,
                                              self.local_AC.reconstruction_error:np.zeros((1,msgs_in.shape[0],msgs_in.shape[1]))})  #here we feed in 0 reconstruction error because the encoding/decoding adds in error for us

        #TODO: encode/decode
        #encode
        # noise = np.random.uniform(low= -BIN_WIDTH/2, high= BIN_WIDTH/2, size= msgs_out.shape)
        # new_msgs_out = msgs_out + noise
        # num_bins = 1.0 // BIN_WIDTH 
        # assert num_bins > 0, "Number of bins needs to be positive"
        # bins = np.arange(-BIN_WIDTH/2, 1 + 3/2 * BIN_WIDTH, BIN_WIDTH)
        # discretized_msgs = np.digitize(new_msgs_out, bins)


        # #decode
        # new_msgs_out = (discretized_msgs - 1) * BIN_WIDTH
        # new_msgs_out = new_msgs_out - noise
        # recon_error = new_msgs_out - msgs_out
        #return new_msgs_out or msgs_out? I think it's new_msgs_out? 

        recon_error = np.zeros_like(msgs_out)


        invalid = 0
        
        #select an action
        actions = []
        if TRAINING:
            for i in range(self.env.num_agents):
                a = np.random.choice(a_size, p = policies[i,:])
                actions.append(a)
                if a not in validActions:
                    invalid += 1
            
        else:
            # TODO
            raise NotImplementedError
            # train_valid = np.zeros(a_size)
            # train_valid[validActions] = 1
            # valid_dist = np.array([a_dist[0,validActions]])
            # valid_dist /= np.sum(valid_dist)
            # a         = np.argmax(a_dist.flatten())
            # if a not in validActions or not GREEDY:
            #     a     = validActions[ np.random.choice(range(valid_dist.shape[1]),p=valid_dist.ravel()) ]
            
        
        



        

        #value should be of shape [# agents]
        v = sess.run(self.local_AC.values, 
                    feed_dict={self.local_AC.critic_inputs:[np.stack(s_critics)],
                    self.local_AC.actions:[actions]  #TODO: make this more elegant.  Ideally, compute it inside tensorflow graph.
                    })
        
        return actions,msgs_out,v,invalid,recon_error

    def work(self,max_episode_length,gamma,sess,coord,saver):
        
        global episode_count#, episode_lengths, episode_mean_values, episode_invalid_ops
        total_steps, i_buf = 0, 0
        episode_buffers = [ [] for _ in range(NUM_BUFFERS) ]
        s1Values = [np.zeros(self.env.num_agents) for _ in range(NUM_BUFFERS)]

        

        with sess.as_default(), sess.graph.as_default():
            while self.shouldRun(coord, episode_count):
                sess.run(self.pull_global)
#                 sess.run(self.copy_weights)

                # value_buffer[self.metaAgentID]  = np.zeros(self.env.num_agents)
                # action_buffer[self.metaAgentID] = np.zeros(self.env.num_agents)
                episode_buffer, episode_values = [], []
                episode_reward = episode_step_count = episode_inv_count = 0
                d = False

                self.env._reset()

                s_actors = []
                s_critics = []
                validActions = []

                for a in range(1,self.env.num_agents+1):
                    #observe world from each agent's perspectives
                    s_actors.append(self.env._observe_actor(a))
                    s_critics.append(self.env._observe_critic(a))
                    # TODO: shouldn't this take an agetn id as arg?
                    validActions.append(self.env._listNextValidActions)

                #initialize messages
                msgs = self.local_AC.msgs_init

                saveGIF = False
                if OUTPUT_GIFS and self.workerID == 1 and ((not TRAINING) or (episode_count >= self.nextGIF)):
                    saveGIF = True
                    self.nextGIF += 64
                    GIF_episode = int(episode_count)
                    episode_frames = [ self.env._render(mode='rgb_array',screen_height=200,screen_width=200) ]

                

                # reset swarm_reward (for tensorboard)
                swarm_reward = 0

                while (not self.env.finished): # Give me something!
                    #Take an action using probabilities from policy network output.
                    #actions is a vector containing action for each agents
                    #msgs is a matrix containing the (real-valued) message vectors each agent generates
                    #invalid is how many agents attempted to take an invalid action in this time step

                    # @Renbo: so right now, I'm just passing the msgs output of the net (which indicates msg TO each agent, i.e. msgs[i,:]=message to agent i)
                    # Rather than passing it right back in, we need to encode (convert to 2x100 vector of DISCRETE ELEMENTS) and decode it like we discussed 
                    # We need a different epsilon for EACH element of msgs
                    # Then we need to keep track of the reconstruction error (difference btween original message and decoding of it) for each element of each message
                    actions,msgs,v,invalid,recon_error = self.choose_actions_msgs(s_actors,s_critics, msgs, validActions, TRAINING)

                    #reconstructed_msgs = 
                    #delta = 

                    #keep track of how many times agents attempt to take "invalid" actions
                    episode_inv_count += invalid

                    
                    #TODO: should probably shuffle order in which agents get to step
                    for agent in range(1,self.env.num_agents + 1):
                        
                        _ = self.env._step((agent, actions[agent - 1]),episode=episode_count)

                    

                    r = self.env.compute_reward()  #get shared reward after all agents have stepped

                    # get observatins for each agent


                    s1_actors = []
                    s1_critics = []
                    validActions = []

                    for a in range(1,self.env.num_agents+1):
                        #observe world from each agent's perspectives
                        s1_actors.append(self.env._observe_actor(a))
                        s1_critics.append(self.env._observe_critic(a))
                        validActions.append(self.env._listNextValidActions)
                        
                    d = self.env.finished

                    if saveGIF:
                        episode_frames.append(self.env._render(mode='rgb_array',screen_width=200,screen_height=200))

                    # @Renbo: we'll need to add reconstruction error to this episode buffer.  This is where we store up all the data we need for training.
                    episode_buffer.append([s_actors,s_critics,actions,r,d,v,recon_error])
                    episode_values.append(v)
                    episode_reward += r
                    s_actors = s1_actors
                    s_critics = s1_critics
                    total_steps += 1
#                     steps_on_goal += int(on_goal)
#                     on_goal = on_goal1
                    episode_step_count += 1

                    
                    if d == True:
#                         s1 = s  #Oh yeah!! We are done, we did it!!!
                        print('\n{} Goodbye World. We did it!'.format(episode_step_count), end='\n')

                    # If the episode hasn't ended, but the experience buffer is full, then we
                    # make an update step using that experience rollout.



                    
                    if TRAINING and (len(episode_buffer) % EXPERIENCE_BUFFER_SIZE == 0 or d):
                        # Since we don't know what the true final return is, we "bootstrap" from our current value estimation.
                        


                        if not d:
                            #take 1 additional step, the values will become the s1Values
                            
                            _, _,v,_,_ = self.choose_actions_msgs(s_actors, s_critics, msgs, validActions, TRAINING)
                            
                           

                            # self.synchronize() # synchronize starting time of the threads


                            
                            s1Values[i_buf][:] = v 
                            
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
                        
                        #v_l,p_l, e_l, gn_a, gn_c, vn_a, vn_c
                        v_l, p_l, e_l, gn_a, gn_c, vn_a, vn_c = self.train(episode_buffers[i_rand],sess,gamma,s1Values[i_rand])

                        i_buf = (i_buf + 1) % NUM_BUFFERS
                        episode_buffers[i_buf] = []

                       # sess.run(self.pull_global)
#                         sess.run(self.copy_weights)

                    # self.synchronize()
                    sess.run(self.pull_global)
                    if episode_step_count >= max_episode_length or d:
                        break

                self.episode_lengths.append(episode_step_count)
                # TODO: for some reason episode_values is empty when we're trying to take mean of it
                self.episode_mean_values.append(np.nanmean(episode_values))
                self.episode_invalid_ops.append(episode_inv_count)
                

                # Periodically save gifs of episodes, model parameters, and summary statistics.
                if episode_count % EXPERIENCE_BUFFER_SIZE == 0 and printQ:
                    print('                                                                                   ', end='\r')
                    print('{} Episode terminated ({},{})'.format(episode_count, self.agentID), end='\r')

                
                swarm_reward += episode_reward

                # self.synchronize()

                self.episode_rewards.append(swarm_reward)

                # TODO: figure out what the heck all of this is doing

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
                        mean_reward = np.mean(self.episode_rewards[-SL:])
                        mean_length = np.mean(self.episode_lengths[-SL:])
                        mean_value = np.mean(self.episode_mean_values[-SL:])
                        mean_invalid = np.mean(self.episode_invalid_ops[-SL:])
                        current_learning_rate_actor = sess.run(lr_a,feed_dict={global_step:episode_count})
                        current_learning_rate_critic = sess.run(lr_c,feed_dict={global_step:episode_count})


                        summary = tf.Summary()
                        summary.value.add(tag='Perf/Actor Learning Rate',simple_value=current_learning_rate_actor)
                        summary.value.add(tag='Perf/Critic Learning Rate',simple_value=current_learning_rate_critic)
                        summary.value.add(tag='Perf/Reward', simple_value=mean_reward)
                        summary.value.add(tag='Perf/Length', simple_value=mean_length)
                        summary.value.add(tag='Perf/Valid Rate', simple_value=(mean_length-mean_invalid)/mean_length)
                        summary.value.add(tag='Losses/Value Loss', simple_value=v_l)
                        summary.value.add(tag='Losses/Policy Loss', simple_value=p_l)
                        summary.value.add(tag='Losses/Actor Grad Norm', simple_value=gn_a)
                        summary.value.add(tag='Losses/Critic Grad Norm', simple_value=gn_c)
                        summary.value.add(tag='Losses/Actor Var Norm', simple_value=vn_a)
                        summary.value.add(tag='Losses/Critic Var Norm', simple_value=vn_c)
                       
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
BIN_WIDTH              = 1/15
max_episode_length     = 256
episode_count          = 0
EPISODE_START          = episode_count
gamma                  = .95 # discount rate for advantage estimation and reward discounting
#moved network parameters to ACNet.py
EXPERIENCE_BUFFER_SIZE = 128
GRID_SIZE              = 10
OBSTACLE_DENSITY       = (0,0) #range of densities
DIAG_MVMT              = False # Diagonal movements allowed?
a_size                 = 5 + int(DIAG_MVMT)*4
SUMMARY_WINDOW         = 10
NUM_META_AGENTS        = 8
NUM_AGENTS             = 2 #int(multiprocessing.cpu_count() / (2 * NUM_META_AGENTS))
NUM_BUFFERS            = 1 # NO EXPERIENCE REPLAY int(NUM_THREADS / 2)
EPISODE_SAMPLES        = EXPERIENCE_BUFFER_SIZE # 64
LR_actor               = 1.e-5 #8.e-5 / NUM_THREADS # default: 1e-5
LR_critic              = 2.e-5
ADAPT_LR               = True
ADAPT_COEFF            = 5.e-5 #the coefficient A in LR_Q/sqrt(A*steps+1) for calculating LR
load_model             = False
RESET_TRAINER          = False
model_path             = './model_MAPF_diff_continuous_comms'
gifs_path              = './gifs_MAPF_diff_continuous_comms'
train_path             = 'train_MAPF_diff_continuous_comms'
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
# episode_rewards        = [ [] for _ in range(NUM_META_AGENTS) ]
# episode_lengths        = [ [] for _ in range(NUM_META_AGENTS) ]
# episode_mean_values    = [ [] for _ in range(NUM_META_AGENTS) ]
# episode_invalid_ops    = [ [] for _ in range(NUM_META_AGENTS) ]
# value_buffer           = [ [] for _ in range(NUM_META_AGENTS) ]
# action_buffer          = [ [] for _ in range(NUM_META_AGENTS) ]
# episode_steps_on_goal  = [ [] for _ in range(NUM_META_AGENTS) ]
printQ                 = False # (for headless)
# swarm_reward           = [0]*NUM_META_AGENTS


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
    master_network = ACNet(GLOBAL_NET_SCOPE,a_size, NUM_AGENTS,None,None,False,GRID_SIZE,GLOBAL_NET_SCOPE) # Generate global network

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

    # TODO: make all this num_workers stuff more elegant
    if TRAINING:
        #num_workers = NUM_THREADS # Set workers # = # of available CPU threads
        num_workers = 1
    else:
        # num_workers = NUM_THREADS
        num_workrs =1
        NUM_META_AGENTS = 1
    
    gameEnvs, workers, groupLocks = [], [], []
    n=0#counter of total number of agents (for naming)

    # Redo such that we only have 1 worker per env, and each worker contains all the agents
    for ma in range(NUM_META_AGENTS):
#         
        num_workers=1
        
        gameEnv = mapf_gym.MAPFEnv(num_agents=NUM_AGENTS, DIAGONAL_MOVEMENT=DIAG_MVMT, SIZE=GRID_SIZE, 
                                   PROB=OBSTACLE_DENSITY, FULL_HELP=FULL_HELP)
        gameEnvs.append(gameEnv)

        # Create groupLock
        workerNames = ["worker_"+str(i) for i in range(n,n+num_workers)]
        # groupLock = GroupLock.GroupLock([workerNames,workerNames])
        # groupLocks.append(groupLock)

        # # Create worker classes
        # workersTmp = []
        # for i in range(ma*num_workers+1,(ma+1)*num_workers+1):
        #     # self, game, metaAgentID, a_size, groupLock
        #     workersTmp.append(Worker(gameEnv,ma,a_size,groupLock))
        #     n+=1
        # workers.append(workersTmp)
        workers.append([Worker(gameEnv,ma,a_size)])#,groupLock)])

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
                # groupLocks[ma].acquire(0,worker.name) # synchronize starting time of the threads
                worker_work = lambda: worker.work(max_episode_length,gamma,sess,coord,saver)
                print("Starting worker " + str(worker.workerID))
                t = threading.Thread(target=(worker_work))
                t.start()
                worker_threads.append(t)
        coord.join(worker_threads)

if not TRAINING:
    print([np.mean(plan_durations), np.sqrt(np.var(plan_durations)), np.mean(np.asarray(plan_durations < max_episode_length, dtype=float))])

