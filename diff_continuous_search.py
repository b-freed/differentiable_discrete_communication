#TODO:

    #can get rid of recipeint set
    
#change code around observations because we no longer have 2 part observations, it's just one big matrix

#this should be the thing, right?
from __future__ import division

import gym
import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.layers as layers
import matplotlib.pyplot as plt
# from od_mstar3 import cpp_mstar
# from od_mstar3.col_set_addition import OutOfTimeError,NoSolutionError
import threading
import time
import scipy.signal as signal
import os
import GroupLock
import multiprocessing
#%matplotlib inline
import mapf_gym_search_diff_comms as mapf_gym
import pickle
# import imageio
from ACNet_search_diff_comms_recon_error_team import ACNet

from tensorflow.python.client import device_lib
dev_list = device_lib.list_local_devices()
print(dev_list)
# assert len(dev_list) > 1 #is this necessary?




def make_gif(images, fname, duration=2, true_image=False,salience=False,salIMGS=None):
    imageio.mimwrite(fname,images,subrectangles=True)
    print("wrote gif")

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




class Worker:
    def __init__(self, game, metaAgentID, a_size):
        self.workerID = metaAgentID
        self.env = game
        self.metaAgentID = metaAgentID
        self.name = "worker_"+str(self.workerID)

        self.nextGIF = episode_count # For GIFs output
        #Create the local copy of the network and the tensorflow op to copy global parameters to local network
        #                        scope, a_size,      num_agents,     trainer_A,trainer_C,TRAINING,GRID_SIZE,GLOBAL_NET_SCOPE
        self.local_AC = ACNet(self.name,a_size, self.env.num_agents, trainer_A,trainer_C,True,GRID_SIZE,GLOBAL_NET_SCOPE)
#         self.copy_weights = self.local_AC.homogenize_weights
        self.pull_global = update_target_graph(GLOBAL_NET_SCOPE, self.name)

        # logging stuff for tensorboard
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.episode_invalid_ops = []
        
    def train(self, rollout, sess, gamma, bootstrap_values,imitation=False):
        #TODO: consider making action_advantages_all a global variable that each agent fills in with their own action advantages.  This would allow us to only calculate each one once
        global episode_count, advantages_buffer



        
        
        #0: s_actors,
        #1: s_critics,
        #2: actions,
        #3: r,
        #4: d,
        #5: v,
        #6: recon_error

        rollout = np.array(rollout)

        observations_actors = rollout[:,0]
        observations_critics = rollout[:,1]
        actions = np.stack(rollout[:,2])
        rewards = rollout[:,3]
        dones = rollout[:,4]
        values = np.stack(rollout[:,5])
        recon_error = np.stack(rollout[:,6])

 

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
        rnn_state_in = self.local_AC.rnn_state_init

        feed_dict = {
            global_step:episode_count,
            self.local_AC.target_v:np.stack(discounted_rewards_all),
            self.local_AC.actor_inputs:np.stack(observations_actors),
            self.local_AC.critic_inputs:np.stack(observations_critics),
            self.local_AC.actions:actions,
            self.local_AC.advantages:advantages_all,
            self.local_AC.msgs_in:msgs_in,
            self.local_AC.rnn_state_in:rnn_state_in,
            self.local_AC.reconstruction_error:recon_error
        }
        
        v_l,p_l,e_l,gn_a,gn_c,vn_a,vn_c,_,_,policies,msgs_out,rnn_state_out = sess.run([
            self.local_AC.c_loss,
            self.local_AC.policy_loss,
            self.local_AC.entropy,
            self.local_AC.a_grad_norms,
            self.local_AC.c_grad_norms,
            self.local_AC.a_var_norms,
            self.local_AC.c_var_norms,
            self.local_AC.apply_a_grads,
            self.local_AC.apply_c_grads,
            self.local_AC.policies,
            self.local_AC.msgs_out,
            self.local_AC.rnn_state_out],
            feed_dict=feed_dict)


        
        
        # feed_dict = {
        #     global_step:episode_count,
        #     self.local_AC.target_v:np.stack(discounted_rewards[:T]),
        #     self.local_AC.actor_inputs:np.stack(observations_actor[:T]),
        #     self.local_AC.critic_inputs:np.stack(observations_critic[:T]),
        #     self.local_AC.msgs_binary:np.stack(msgs_binary[:T]),
        #     self.local_AC.actions:actions[:T],
        #     self.local_AC.train_valid:np.stack(valids[:T]),
        #     self.local_AC.action_advantages:action_advantages,
        #     self.local_AC.msg_advantages:msg_advantages,
        #     # self.local_AC.train_value:train_value,
        #     # self.local_AC.target_blockings:blockings,
        #     self.local_AC.target_on_goals:on_goals[:T],
        #     self.local_AC.state_in_actor[0]:rnn_state_actor[0],
        #     self.local_AC.state_in_actor[1]:rnn_state_actor[1],
        #     self.local_AC.state_in_critic[0]:rnn_state_critic[0],
        #     self.local_AC.state_in_critic[1]:rnn_state_critic[1]

        # }
        
        # v_l,p_l,msg_l,valid_l,og_l,e_l,e_msg,gn_a,gn_c,vn_a,vn_c,_,_ = sess.run([
        #     self.local_AC.c_loss,
        #     self.local_AC.policy_loss,
        #     self.local_AC.msg_loss,
        #     self.local_AC.valid_loss,
        #     self.local_AC.on_goal_loss,
        #     self.local_AC.entropy,
        #     self.local_AC.msg_entropy,
        #     self.local_AC.a_grad_norms,
        #     self.local_AC.c_grad_norms,
        #     self.local_AC.a_var_norms,
        #     self.local_AC.c_var_norms,
        #     # self.local_AC.blocking_loss,
        #     self.local_AC.apply_a_grads,
        #     self.local_AC.apply_c_grads],
        #     feed_dict=feed_dict)

       
        return v_l/len(rollout), p_l/len(rollout), e_l/len(rollout), gn_a, gn_c, vn_a, vn_c

    def shouldRun(self, coord, episode_count):
        if TRAINING:
            return (not coord.should_stop())
        else:
            return (episode_count < NUM_EXPS)

    # def parse_path(self,path):
    #     '''needed function to take the path generated from M* and create the 
    #     observations and actions for the agent
    #     path: the exact path ouput by M*, assuming the correct number of agents
    #     returns: the list of rollouts for the "episode": 
    #             list of length num_agents with each sublist a list of tuples 
    #             (observation[0],observation[1],optimal_action,reward)'''
    #     result=[[] for i in range(num_workers)]
       
    #     for t in range(len(path[:-1])):
    #         observations=[]
    #         #this is a list of agents we have yet to move
    #         move_queue=list(range(num_workers))
    #         for agent in range(1,num_workers+1):
    #             #observations will contain list of [[poss_map,goal_map,goals_map,obs_map],[dx,dy,mag], msg_mat]
    #             observations.append(self.env._observe(agent))
               
    #         steps=0
    #         while len(move_queue)>0:
    #             steps+=1
    #             i=move_queue.pop(0)
    #             o=observations[i]
    #             pos=path[t][i]
    #             newPos=path[t+1][i]#guaranteed to be in bounds by loop guard
    #             direction=(newPos[0]-pos[0],newPos[1]-pos[1])
    #             a=self.env.world.getAction(direction)

    #             #get message output for this time step
    #             msg_output = sess.run([self.local_AC.msg_output], 
    #                         feed_dict={self.local_AC.inputs:[o[0]], 
    #                                    self.local_AC.goal_pos:[o[1]],
    #                                    self.local_AC.msg_input:[o[2]]})
                
    #             #calculate valid actions, and also update environment with agent's message
    #             _, _, _, _, _, _, valid_action=self.env._step((i+1,a, msg_output[0]))

    #             if steps>num_workers**2:
    #                 #if we have a very confusing situation where lots of agents move
    #                 #in a circle (difficult to parse and also (mostly) impossible to learn)
    #                 return None
    #             if not valid_action:
    #                 #the tie must be broken here
    #                 move_queue.append(i)
    #                 continue
    #             #o[2] here should be the message mat... we need to append it to result (rollout) so we can use it in calculating i_l later
    #             result[i].append([o[0],o[1],a, o[2]])
    #     return result

    def choose_action_msg(self, s_actors, s_critics, msgs_in, rnn_state_in, validActions, training):
        '''
        Chooses an action for each agent and get a message for each agent
        Computes value for EACH agent's state

        Takes as input:
            s_actors:  concatenation of both actors' observations
            s_critics: concatenation of both critics' observations
        '''
        
        #@Ben: need to put in agents positions here? 
        policies, msgs_out, rnn_state_out  = sess.run([self.local_AC.policies,
                                                       self.local_AC.msgs_out,
                                                       self.local_AC.rnn_state_out],
                                            feed_dict={self.local_AC.actor_inputs:[np.stack(s_actors)],
                                                       self.local_AC.rnn_state_in:rnn_state_in,
                                                       self.local_AC.msgs_in:msgs_in,
                                                       self.local_AC.reconstruction_error:np.zeros((1,msgs_in.shape[0],msgs_in.shape[1]))})

        # @Renbo: encode/decode messages
                #TODO: encode/decode
        #encode
        # noise = np.random.uniform(low= -BIN_WIDTH/2, high= BIN_WIDTH/2, size= msgs_in.shape)
        # perturbed_msgs_out = msgs_out + noise
        # num_bins = 1.0 // BIN_WIDTH 
        # assert num_bins > 0, "Number of bins needs to be positive"
        # bins = np.arange(-BIN_WIDTH/2, 1 + 3/2 * BIN_WIDTH, BIN_WIDTH)
        # discretized_msgs = np.digitize(perturbed_msgs_out, bins)


        # #decode
        # new_msgs_out = (discretized_msgs - 1) * BIN_WIDTH
        # new_msgs_out = new_msgs_out - noise
        # recon_error = new_msgs_out - msgs_out

        recon_error = np.zeros_like(msgs_out)
       
         #select an action
        invalid = 0
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

        # TODO: figure out what to do about conditioning the ciritc on actions? Maybe not necessary
        # action_map = self.env._observe_action_map(self.agentID,action_buffer[self.metaAgentID])

        v = sess.run(self.local_AC.values, 
                    feed_dict={self.local_AC.critic_inputs:[np.stack(s_critics)]})
        
        return actions,msgs_out,v,rnn_state_out,invalid,recon_error

    def work(self,max_episode_length,gamma,sess,coord,saver):
        global episode_count
        total_steps, i_buf = 0, 0
        episode_buffers = [ [] for _ in range(NUM_BUFFERS) ]
        s1Values = [np.zeros(self.env.num_agents) for _ in range(NUM_BUFFERS)]

        print('==================================')

        with sess.as_default(), sess.graph.as_default():
            while self.shouldRun(coord, episode_count):
                sess.run(self.pull_global)
#                 sess.run(self.copy_weights)

                episode_buffer, episode_values = [], []
                episode_reward = episode_step_count = episode_inv_count = 0
                d = False

                # Initial state from the environment
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

                rnn_state            = self.local_AC.rnn_state_init
                msgs                 = self.local_AC.msgs_init

                saveGIF = False
                if OUTPUT_GIFS and self.workerID == 1 and ((not TRAINING) or (episode_count >= self.nextGIF)):
                    saveGIF = True
                    self.nextGIF += 64
                    GIF_episode = int(episode_count)
                    episode_frames = [ self.env._render(mode='rgb_array',screen_height=200,screen_width=200) ]

                # reset swarm_reward (for tensorboard)
                swarm_reward = 0
                    
                #runs for single episode
                while (not self.env.finished): # Give me something!
                    
                    #actions,new_msgs_out,v,rnn_state_out,invalid,recon_error
                    actions,msgs,v,rnn_state,invalid, recon_error = self.choose_action_msg(s_actors,
                                                                        s_critics,
                                                                        msgs,
                                                                        rnn_state, 
                                                                        validActions,
                                                                        TRAINING)
                   


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
                        episode_frames.append(self.env._render(mode='rgb_array',screen_width=900,screen_height=900))

                   # @Renbo: we'll need to add reconstruction error to this episode buffer.  This is where we store up all the data we need for training.
                    episode_buffer.append([s_actors,s_critics,actions,r,d,v,recon_error])
                    episode_values.append(v)
                    episode_reward += r
                    s_actors = s1_actors
                    s_critics = s1_critics
                    total_steps += 1

                    episode_step_count += 1

                    if d == True:
#                         s1 = s  #Oh yeah!! We are done, we did it!!!
                        print('\n{} Goodbye World. We did it!'.format(episode_step_count), end='\n')

                    # If the episode hasn't ended, but the experience buffer is full, then we
                    # make an update step using that experience rollout.
                    if TRAINING and (len(episode_buffer) % EXPERIENCE_BUFFER_SIZE == 0 or d):
                  
                        if not d:
                                #take 1 additional step, the values will become the s1Values
                                _, _,v,_,_,_ = self.choose_action_msg(s_actors,
                                                                    s_critics,
                                                                    msgs,
                                                                    rnn_state, 
                                                                    validActions,
                                                                    TRAINING)
                               
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
                       
                        v_l, p_l, e_l, gn_a, gn_c, vn_a, vn_c = self.train(episode_buffers[i_rand],sess,gamma,s1Values[i_rand])

                        i_buf = (i_buf + 1) % NUM_BUFFERS
                        episode_buffers[i_buf] = []

                    sess.run(self.pull_global)
                    if episode_step_count >= max_episode_length or d:
                        break

                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(np.nanmean(episode_values))
                self.episode_invalid_ops.append(episode_inv_count)

                # Periodically save gifs of episodes, model parameters, and summary statistics.
                if episode_count % EXPERIENCE_BUFFER_SIZE == 0 and printQ:
                    print('                                                                                   ', end='\r')
                    print('{} Episode terminated ({},{})'.format(episode_count, self.agentID, RewardNb), end='\r')

                # TODO: figure out if swarm_reward is really necessary
                swarm_reward += episode_reward

                self.episode_rewards.append(swarm_reward)

                # TODO: sort this all out
                if not TRAINING:
                    raise NotImplementedError
#                     mutex.acquire()
#                     if episode_count < NUM_EXPS:
#                         plan_durations[episode_count] = episode_step_count
#                     if self.workerID == 1:
#                         episode_count += 1
#                         print('({}) Thread {}: {} steps, {:.2f} reward ({} invalids).'.format(episode_count, self.workerID, episode_step_count, episode_reward, episode_inv_count))
#                     GIF_episode = int(episode_count)
# #                    saveGIF &= (episode_step_count < max_episode_length)
#                     mutex.release()
                else:
                    episode_count+=1./num_workers

                    if episode_count % SUMMARY_WINDOW == 0:
                        if episode_count % 100 == 0:
                            print ('Saving Model', end='\n')
                            saver.save(sess, model_path+'/model-'+str(int(episode_count))+'.cptk')
                            print ('Saved Model', end='\n')
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
#                         gif_creation = lambda: 
                        make_gif(images, '{}/episode_{:d}_{:d}_{:.1f}.gif'.format(gifs_path,GIF_episode,episode_step_count,swarm_reward[self.metaAgentID]))
#                         threading.Thread(target=(gif_creation)).start()
                    else:
                        make_gif(images, '{}/episode_{:d}_{:d}.gif'.format(gifs_path,GIF_episode,episode_step_count), duration=len(images)*time_per_step,true_image=True,salience=False)
                if SAVE_EPISODE_BUFFER:
                    with open('gifs3D/episode_{}.dat'.format(GIF_episode), 'wb') as file:
                        pickle.dump(episode_buffer, file)




# Learning parameters
BIN_WIDTH              = 1/15
max_episode_length     = 256
episode_count          = 0
EPISODE_START          = episode_count
gamma                  = .95 # discount rate for advantage estimation and reward discounting
#moved network parameters to ACNet.py
EXPERIENCE_BUFFER_SIZE = 128
GRID_SIZE              = 10 #the size of the FOV grid to apply to each agent
ENVIRONMENT_SIZE       = 10#the total size of the environment (length of one side)
OBSTACLE_DENSITY       = 0#(0,.5) #range of densities
BLANK_WORLD            = True
if OBSTACLE_DENSITY >0:
    assert not BLANK_WORLD
DIAG_MVMT              = False # Diagonal movements allowed?
a_size                 = 5 + int(DIAG_MVMT)*4
SUMMARY_WINDOW         = 10
NUM_META_AGENTS        = 1
NUM_AGENTS             = 2 #int(multiprocessing.cpu_count() / (2 * NUM_META_AGENTS))
NUM_BUFFERS            = 1 # NO EXPERIENCE REPLAY int(NUM_AGENTS / 2)
EPISODE_SAMPLES        = EXPERIENCE_BUFFER_SIZE # 64
LR_actor               = 1.e-5 #8.e-5 / NUM_AGENTS # default: 1e-5
LR_critic              = 2.e-5
ADAPT_LR               = True
ADAPT_COEFF            = 5.e-5 #the coefficient A in LR_Q/sqrt(A*steps+1) for calculating LR
load_model             = False
RESET_TRAINER          = False
model_path             = './model_diff_continuous_search_AAAI'
gifs_path              = './gifs_diff_continuous_search_AAAI'
train_path             = 'train_diff_continuous_search_AAAI'
GLOBAL_NET_SCOPE       = 'global'

#Imitation options
PRIMING_LENGTH         = 0      #number of episodes at the beginning to train only on demonstrations
DEMONSTRATION_PROB     = 0.0    #probability of training on a demonstration per episode

# Simulation options
FULL_HELP              = False
OUTPUT_GIFS            = False
SAVE_EPISODE_BUFFER    = False

# Testing
TRAINING               = True
GREEDY                 = False
NUM_EXPS               = 100
MODEL_NUMBER           = 40000

printQ                 = False # (for headless)


tf.reset_default_graph()
print("Hello World")
if not os.path.exists(model_path):
    os.makedirs(model_path)
config      = tf.ConfigProto( allow_soft_placement = True)
config.gpu_options.allow_growth                    = True
config.gpu_options.per_process_gpu_memory_fraction = .8

if not TRAINING:
    plan_durations = np.array([0 for _ in range(NUM_EXPS)])
    mutex = threading.Lock()
    gifs_path += '_tests'
    if SAVE_EPISODE_BUFFER and not os.path.exists('gifs3D'):
        os.makedirs('gifs3D')

#Create a directory to save episode playback gifs to
if not os.path.exists(gifs_path) and OUTPUT_GIFS:
    os.makedirs(gifs_path)

#with tf.device("/device:XLA_GPU:0"):
with tf.device("/device:GPU:0"):
    #                                scope, a_size, num_agents, trainer_A,trainer_C,TRAINING,GRID_SIZE,GLOBAL_NET_SCOPE
    master_network = ACNet(GLOBAL_NET_SCOPE,a_size, NUM_AGENTS, None,     None,    False,   GRID_SIZE,GLOBAL_NET_SCOPE) # Generate global network

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
        num_workers = NUM_AGENTS # Set workers # = # of available CPU threads
    else:
        num_workers = NUM_AGENTS
        NUM_META_AGENTS = 1
    
    gameEnvs, workers, groupLocks = [], [], []
    n=0#counter of total number of agents (for naming)
    for ma in range(NUM_META_AGENTS):
#         num_agents=((ma%4)+1)*2
#         print(num_agents)
#         num_workers=num_agents
        num_workers = 1
        #num_agents=1,world0=None, goal_locs=None, goal_types=None, num_goal_types=2, DIAGONAL_MOVEMENT=False, SIZE=10, PROB=(0,.5), FULL_HELP=False,blank_world=False

        gameEnv = mapf_gym.MAPFEnv(num_agents=NUM_AGENTS, DIAGONAL_MOVEMENT=DIAG_MVMT, SIZE=ENVIRONMENT_SIZE, 
                                   PROB=OBSTACLE_DENSITY, FULL_HELP=FULL_HELP, blank_world = BLANK_WORLD)
        gameEnvs.append(gameEnv)

        # Create groupLock
        workerNames = ["worker_"+str(i) for i in range(n,n+num_workers)]
        # groupLock = GroupLock.GroupLock([workerNames,workerNames])
        # groupLocks.append(groupLock)

        # Create worker classes
        # workersTmp = []
        # for i in range(ma*num_workers+1,(ma+1)*num_workers+1):
        #     workersTmp.append(Worker(gameEnv,ma,n,a_size,groupLock))
        #     n+=1
        # workers.append(workersTmp)

        workers.append([Worker(gameEnv,ma,a_size)])

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
