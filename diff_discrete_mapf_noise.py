
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
import mapf_gym_diff_comms as mapf_gym
import pickle
from ACNet_diff_comms_recon_error_circle import ACNet

from tensorflow.python.client import device_lib


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



class Worker:
    def __init__(self, game, metaAgentID, a_size, ):#groupLock):
        self.workerID = metaAgentID
        self.env = game
        self.metaAgentID = metaAgentID
        self.name = "worker_"+str(self.workerID)
        
        self.nextGIF = episode_count # For GIFs output
        #Create the local copy of the network and the tensorflow op to copy global parameters to local network
        self.local_AC = ACNet(self.name,a_size, self.env.num_agents,trainer_A,trainer_C,True,GRID_SIZE,GLOBAL_NET_SCOPE)
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
        
        v_l,p_l,e_l,gn_a,gn_c,vn_a,vn_c,_,_,policies = sess.run([
            self.local_AC.c_loss,
            self.local_AC.policy_loss,
            
            self.local_AC.entropy,
            
            self.local_AC.a_grad_norms,
            self.local_AC.c_grad_norms,
            self.local_AC.a_var_norms,
            self.local_AC.c_var_norms,
            
            self.local_AC.apply_a_grads,
            self.local_AC.apply_c_grads,
            self.local_AC.policies],
            feed_dict=feed_dict)

       


        return v_l/len(rollout), p_l/len(rollout), e_l/len(rollout), gn_a, gn_c, vn_a, vn_c

    def shouldRun(self, coord, episode_count):
        if TRAINING:
            return (not coord.should_stop())
        else:
            return (episode_count < NUM_EXPS)


    def rotate_by_angle(self, msg, theta):
        '''
        msg: array of shape [#agents, msg_length, 2] representing msg to be perturbed (rotated)
        theta: array of shape [#agents, msg_length] representing angle to perturb (rotate) msg by

        this is a slightly unfortunate way to implement this, but I didn't want to have to futz around with array dimensions
        '''

        x = msg[:,:,0]
        y = msg[:,:,1]

        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        x_new = cos_theta*x - sin_theta*y
        y_new = sin_theta*x + cos_theta*y

        msg_rotated = np.concatenate([np.expand_dims(x_new,-1), np.expand_dims(y_new, -1)], axis = -1)

        assert msg_rotated.shape == msg.shape

        return msg_rotated

    def compute_error_angle(self, v1, v2):
        '''
        computes the error angle (error) that, when applied to vector_from yields vector_to (i.e. v2 = R(error)@v1)
        '''
        
        angle1 = self.get_vector_angles(v1)
        angle2 = self.get_vector_angles(v2)

        error = angle2 - angle1

        # we want error to between +/-pi
        while np.any(error > np.pi):
            error[error > np.pi] -= 2*np.pi

        while np.any(error < -np.pi):
            error[error < -np.pi] += 2*np.pi

        

        return error

    def get_vector_angles(self, msg):

        '''
        returns the angle of each vector in msg

        each element in angle is between -pi and +pi
        '''
       
        x = msg[:,:,0]
        y = msg[:,:,1]

        angle = np.arctan2(y,x)

        return angle

    def get_vector_at_center_of_bin(self, bin_inds):

        '''
        bin_inds is a vector of integers, 0 to NUM_BINS - 1, that is of shape [#agents x msg_length]
        '''

        bins = np.linspace(-np.pi,np.pi,NUM_BINS, endpoint = False)
        bin_centers = bins + BIN_WIDTH/2

        angles = bin_centers[bin_inds]

        assert np.all(angles <= np.pi)
        assert np.all(angles >= -np.pi)

        x = np.cos(angles)
        y = np.sin(angles)

        msgs = np.concatenate([np.expand_dims(x,-1),np.expand_dims(y,-1)], axis = 2)

        assert np.all(angles == self.get_vector_angles(msgs))
        
        return msgs 




    
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

        
        #encode
        noise = np.random.uniform(low= -BIN_WIDTH/2, high= BIN_WIDTH/2, size= msgs_out.shape[:2])
        new_msgs_out = self.rotate_by_angle(msgs_out, noise)
        assert new_msgs_out.shape == msgs_out.shape
        msg_out_angles = self.get_vector_angles(new_msgs_out)


        bins = np.linspace(-np.pi,np.pi,NUM_BINS, endpoint = False) #endpoint should be false, otherwise we have only 15 bins
        discretized_msgs = np.digitize(msg_out_angles, bins) - 1 #subtract 1 because np.digitize returns the index of the first bin we are less than

        
        def process_ind_msg(msg, noise_type = 3):
            msg, rand_arr = encode_for_channel_noise(msg, method = "perm")
            msg = convert_msg_to_bitstring(msg)
            msg = add_noise_to_msg(msg, noise_type = noise_type)
            msg = convert_bitstring_to_msg(msg)
            msg = decode_for_channel_noise(msg, rand_arr, method = "perm")
            return msg
        
        
        # here we process the encoded msg such that we can add noise to it
        np.random.seed(int(time.time()))
        new_msgs = []
        for msg in discretized_msgs:
            new_msgs.append(process_ind_msg(msg))
        discretized_msgs = np.asarray(new_msgs)
        
        #decode
        # get vectors correponding to centers of bins indicated by discretized_msgs

        new_msgs_out = self.get_vector_at_center_of_bin(discretized_msgs)
        # rotate msgs BACK by noise
        new_msgs_out = self.rotate_by_angle(new_msgs_out, -noise)
        recon_error = self.compute_error_angle(msgs_out, new_msgs_out)

        

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
            raise NotImplementedError
           

        #value should be of shape [# agents]
        v = sess.run(self.local_AC.values, 
                    feed_dict={self.local_AC.critic_inputs:[np.stack(s_critics)],
                    self.local_AC.actions:[actions]  #TODO: make this more elegant.  Ideally, compute it inside tensorflow graph.
                    })
        
        
        return actions,new_msgs_out,v,invalid,recon_error

    def work(self,max_episode_length,gamma,sess,coord,saver):
        
        global episode_count#, episode_lengths, episode_mean_values, episode_invalid_ops
        total_steps, i_buf = 0, 0
        episode_buffers = [ [] for _ in range(NUM_BUFFERS) ]
        s1Values = [np.zeros(self.env.num_agents) for _ in range(NUM_BUFFERS)]

        

        with sess.as_default(), sess.graph.as_default():
            while self.shouldRun(coord, episode_count):
                sess.run(self.pull_global)

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

                    

                    # old_msgs = msgs.copy()

                    actions,msgs,v,invalid,recon_error = self.choose_actions_msgs(s_actors,s_critics, msgs, validActions, TRAINING)




                    #keep track of how many times agents attempt to take "invalid" actions
                    episode_inv_count += invalid

                    
                    for agent in range(1,self.env.num_agents + 1):
                        
                        _ = self.env._step((agent, actions[agent - 1]),episode=episode_count)

                    

                    r = self.env.compute_reward()  #get shared reward after all agents have stepped



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

                    
                    if TRAINING and (len(episode_buffer) % EXPERIENCE_BUFFER_SIZE == 0 or d):

                        if not d:
                            #take 1 additional step, the values will become the s1Values
                            
                            _, _,v,_,_ = self.choose_actions_msgs(s_actors, s_critics, msgs, validActions, TRAINING)
                            
                           
                            
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


                    sess.run(self.pull_global)
                    if episode_step_count >= max_episode_length or d:
                        break

                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(np.nanmean(episode_values))
                self.episode_invalid_ops.append(episode_inv_count)
                

                # Periodically save gifs of episodes, model parameters, and summary statistics.
                if episode_count % EXPERIENCE_BUFFER_SIZE == 0 and printQ:
                    print('                                                                                   ', end='\r')
                    print('{} Episode terminated ({},{})'.format(episode_count, self.agentID), end='\r')

                
                swarm_reward += episode_reward


                self.episode_rewards.append(swarm_reward)


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



# Learning parameters
NUM_BINS               = 16
BIN_WIDTH              = 2*np.pi/NUM_BINS
max_episode_length     = 256
episode_count          = 0
EPISODE_START          = episode_count
gamma                  = .95 # discount rate for advantage estimation and reward discounting
#moved network parameters to ACNet.py
EXPERIENCE_BUFFER_SIZE = 256
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
model_path             = './model_diff_discrete_toy_noise_AAAI'
gifs_path              = './gifs_diff_discrete_toy_noise_AAAI'
train_path             = 'train_diff_discrete_toy_noise_AAAI'
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

printQ                 = False # (for headless)

###################################################
####### section for noisy channel functions  ######
###################################################



msg_length             = 10                     #the length of the original msg
digit_width            = int(np.ceil(np.log2(NUM_BINS))) #how many bits per number
bit_length             = digit_width*msg_length #the length of the binary msg
hamm_start             = 5                      #inclusive
hamm_end               = 15                     #exclusive

p_flip_zero = 0.1
p_flip_one = 0.05


#this function adds noise to a msg, approximating the real-life scenario
def add_noise_to_msg(msg, noise_type, noise_mag = 0.2):
    #adding hamming noise between hamm_start and hamm_end
    if noise_type is 0:
        flip = np.random.choice([0, 1], size= hamm_end - hamm_start, p=[1 - noise_mag, noise_mag])
        print(f"flip is   {flip}")
        msg[hamm_start:hamm_end] = np.bitwise_xor(msg[hamm_start:hamm_end], flip)

    #adding random bits
    elif noise_type is 1:
        #this is just to keep consistant with the magnitude
        flip = np.random.choice([0, 1], size= bit_length, p=[1 - noise_mag/10, noise_mag/10])
        print(f"flip is \n{flip}")
        for i in range(bit_length):
            j = bit_length - i - 1
            if(flip[j]):
                msg = np.insert(msg,j,np.random.randint(2))
        msg = msg[:bit_length]
    #deleting random bits
    elif noise_type is 2:
        flip = np.random.choice([0, 1], size= bit_length, p=[1 - noise_mag/10, noise_mag/10])
        print(f"flip is \n{flip}")
        for i in range(bit_length):
            j = bit_length - i - 1
            if(flip[j]):
                msg = np.delete(msg,j)
        missing_bits = bit_length - len(msg)
        add = np.random.randint(2, size=int(missing_bits))
        msg = np.concatenate([msg,add])
    # noise that depends on the actual context
    elif noise_type is 3:
        for i in range(bit_length):
            if(msg[i]==0):
                if(np.random.uniform()<p_flip_zero):
                    msg[i] = 1
                    #print(f"flip 0 at position {i}")
            else:
                if(np.random.uniform()<p_flip_one):
                    msg[i] = 0
                    #print(f"flip 1 at position {i}")

    return msg

def convert_msg_to_bitstring(msg):
    new_msg = []
    for num in msg:
        binary = np.binary_repr(num, width = digit_width)
        for char in binary:
            new_msg.append(int(char))
    return np.asarray(new_msg)

def convert_bitstring_to_msg(msg):
    new_msg = []
    for i in range(int(len(msg)/digit_width)):
        binary = ""
        for j in range(digit_width):
            binary += str(msg[i*digit_width + j])
        #print(f"binary is {binary}")
        num = int(binary, 2) #this line also need to be changed
        new_msg.append(num)
    return np.asarray(new_msg)


def encode_for_channel_noise(msg, method = "perm"):
    if method is "xor":
        random_arr = np.random.randint(2, size=int(bit_length))
        msg = np.bitwise_xor(msg, random_arr)
    elif method is "perm":
        random_arr = []
        new_msg = np.empty(msg_length, dtype=np.int32)
        for i in range(msg_length):
            rand_perm = np.random.permutation(NUM_BINS)
            random_arr.append(rand_perm)
            new_msg[i] = rand_perm[msg[i]]
        msg = new_msg
    return msg, random_arr

def decode_for_channel_noise(msg, random_arr, method = "perm"):
    if method is "xor":
        #random_arr = np.random.randint(2, size=int(bit_length))
        msg = np.bitwise_xor(msg, random_arr)
    elif method is "perm":
        #permutation = np.random.permutation(bit_length)
        new_msg = np.empty(msg_length, dtype=np.int32)
        for i in range(msg_length):
            ind = np.where(random_arr[i] == msg[i])[0]
            #print(f"ind is {ind}")
            if(len(ind) == 0):
                #print("mapping to invalid msg")
                new_msg[i] = np.random.randint(NUM_BINS)
            elif(len(ind) == 1):
                new_msg[i] = ind[0]
            else:
                print("something is wrong with the msg permutation: having multiple indices")
                exit()
        msg = new_msg
    return msg


#########################################
##### end of noisy channel functions  ###
#########################################



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

