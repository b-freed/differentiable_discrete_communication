###### TODO ###############
# add as an input recipient_matrix, which is a (sequence of) matricies specifying which agent sends a message to which other agent at next time step.
    # i.e. if we want agent i to recieve a message from agent j: recipient_matrix[t,i,j] = 1
    # for simple 2 agent environment in which agents send to eachother (but not themselves), recipient_matrix[t,:,:]=[[0,1],[1,0]]
    # then, (assuming the msg output of the net denotes the message sent TO each agent at next time step), the final step in message computation
    # should be to multiply the message outputs of each net by the recipient matrix 

# for discrete comms version:  
    # add a boolean variable indicating whether we're doing feedforward or backpropping.  
        # In feedforward mode, we don't need to worry about adding noise, because the environment will do that for us
        # In backpropping mode, we'll need to add noise, but make sure it's the same noise as what was added as a result of discretization info loss
    # add a variable epsilon, which will specify 


# @Renbo: This is going to be the tricky part.  You'll need to modify the RNN cell I made so that it takes reconstruction error as part of the input.  Probably input will need to become a tuple of (normal input, reconstruction error)
# Reconstruction error should get added to msg just before it is output.
# During a rollout (when we're in the work() function), we don't want to add any reconstruction error because the encoding/decoding procedure will do taht for us
# We only want to add reconstruction error during a training update (when we're in the train() function), considtent with what ever error actually occured during the corresponding time step in the rollout 
# (which is why we need to keep track of reconstruction error during the rollout.) 

import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.python.framework import tensor_shape
# import tensorflow.math_ops as math_ops
import numpy as np
#parameters for training
GRAD_CLIP              = 500.0
RNN_SIZE               = 512
ACTION_REPR_SIZE       = 50
MSG_REPR_SIZE          = 100
num_agents             = 2
msg_length             = 10

#Used to initialize weights for policy and value output layers (Do we need to use that? Maybe not now)
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

class ACNet:
    def __init__(self, scope, a_size, num_agents, trainer_A,trainer_C,TRAINING,GRID_SIZE,GLOBAL_NET_SCOPE):
        
        with tf.variable_scope(scope):
            
            self.num_agents = num_agents
            self.actor_inputs = tf.placeholder(shape=[None,self.num_agents,2,GRID_SIZE,GRID_SIZE], dtype=tf.float32)
            self.my_actor_input = tf.transpose(self.actor_inputs, perm=[0,1,3,4,2])

            self.critic_inputs = tf.placeholder(shape=[None,self.num_agents,4,GRID_SIZE,GRID_SIZE], dtype=tf.float32)
            self.my_critic_input = tf.transpose(self.critic_inputs, perm=[0,1,3,4,2])

            self.reconstruction_error = tf.placeholder(shape=[None,self.num_agents,msg_length], dtype=tf.float32)
            
           
            self.actions             = tf.placeholder(shape=[None, self.num_agents], dtype=tf.int32)
            self.actions_onehot      = tf.one_hot(self.actions, a_size, dtype=tf.float32)

            actions_other             = tf.reverse(self.actions,[1])
            actions_other_onehot      = tf.one_hot(actions_other, a_size, dtype=tf.float32)

            self.msgs_init = np.zeros((self.num_agents,msg_length))
            
            self.policies, self.msgs_out, self.values = self._build_net(self.my_actor_input,
                                                                    self.my_critic_input,
                                                                    actions_other_onehot,
                                                                    self.reconstruction_error,
                                                                    GRID_SIZE,
                                                                    RNN_SIZE,
                                                                    TRAINING,
                                                                    a_size)
        
            if TRAINING:
                # TODO: this all needs to get modified to deal with multiple agents
                # self.msg_binary          = tf.placeholder(shape = [None, msg_length], dtype = tf.float32)
                
                self.target_v            = tf.placeholder(tf.float32, [None,self.num_agents], 'Vtarget')
                self.advantages          = tf.placeholder(shape=[None,self.num_agents], dtype=tf.float32)
                # self.msg_advantages      = tf.placeholder(shape=[None], dtype=tf.float32)
               
                self.responsible_outputs = tf.reduce_sum(self.policies * self.actions_onehot, [2])
                # self.responsible_msgs    = tf.reduce_sum(self.P_msg  * self.msg_binary,     [1])
                # self.log_prob_msgs       = tf.reduce_sum(self.msg_binary*tf.log(tf.clip_by_value(self.P_msg,1e-15,1.0)) + (1-self.msg_binary)*tf.log(tf.clip_by_value(1 - self.P_msg,1e-15,1.0)), axis=1) 

                # Loss Functions
                with tf.name_scope('c_loss'):
                    self.c_loss    = tf.reduce_sum(tf.square(self.target_v - self.values))

                # something to encourage exploration
                self.entropy       = - tf.reduce_sum(self.policies * tf.log(tf.clip_by_value(self.policies,1e-10,1.0)))/self.num_agents
                # self.msg_entropy   = - tf.reduce_sum(self.P_msg  * tf.log(tf.clip_by_value(self.P_msg, 1e-10,1.0)))
                # self.msg_entropy   = - tf.reduce_sum(self.P_msg * tf.log(tf.clip_by_value(self.P_msg, 1e-15,1.0)) + (1-self.P_msg) * (tf.log(tf.clip_by_value(1 - self.P_msg, 1e-15,1.0))))
                self.policy_loss   = - tf.reduce_sum(tf.log(tf.clip_by_value(self.responsible_outputs,1e-15,1.0)) * self.advantages)
                # self.msg_loss      = - tf.reduce_sum(tf.log(tf.clip_by_value(self.responsible_msgs,   1e-15,1.0)) * self.msg_advantages)
                # self.msg_loss      = - tf.reduce_sum(self.log_prob_msgs * self.msg_advantages)
                
                # self.blocking_loss = - tf.reduce_sum(self.target_blockings*tf.log(tf.clip_by_value(self.blocking,1e-10,1.0))\
                #                           +(1-self.target_blockings)*tf.log(tf.clip_by_value(1-self.blocking,1e-10,1.0)))
                # self.on_goal_loss   = - tf.reduce_sum(self.target_on_goals*tf.log(tf.clip_by_value(self.on_goal,1e-10,1.0))\
                                      # +(1-self.target_on_goals)*tf.log(tf.clip_by_value(1-self.on_goal,1e-10,1.0)))
                # self.msg_loss      = - tf.reduce_sum(self.log_prob_msgs  * tf.expand_dims(self.msg_advantages, axis = 1))
                with tf.name_scope('a_loss'):
                    self.a_loss          = self.policy_loss - self.entropy * 0.01 #+ 0.25*self.blocking_loss #+ self.msg_loss

                # Get gradients from local network using local losses and
                # normalize the gradients using clipping
                local_a_vars         = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope+'/actor')
                local_c_vars         = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope+'/critic')
                self.a_grads         = tf.gradients(self.a_loss, local_a_vars)
                self.c_grads         = tf.gradients(self.c_loss, local_c_vars)
                self.a_var_norms     = tf.global_norm(local_a_vars)
                self.c_var_norms     = tf.global_norm(local_c_vars)
                a_grads, self.a_grad_norms = tf.clip_by_global_norm(self.a_grads, GRAD_CLIP)
                c_grads, self.c_grad_norms = tf.clip_by_global_norm(self.c_grads, GRAD_CLIP)

                # Apply local gradients to global network
                global_actor_vars        = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, GLOBAL_NET_SCOPE+'/actor')
                global_critic_vars       = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, GLOBAL_NET_SCOPE+'/critic')
                self.apply_a_grads       = trainer_A.apply_gradients(zip(a_grads, global_actor_vars))
                self.apply_c_grads       = trainer_C.apply_gradients(zip(c_grads, global_critic_vars))

#             self.homogenize_weights = update_target_graph(str(scope)+'/qvaluesB', str(scope)+'/qvalues')
        print("Hello World... From  "+str(scope))     # :)

    def _build_net(self,actor_inputs,critic_inputs,actions_other,recon_error,GRID_SIZE,RNN_SIZE,TRAINING,a_size):
        w_init   = layers.variance_scaling_initializer()

        with tf.variable_scope('actor'):

            # TODO: check that this reshaping actually works properly
            # First step is to squash the axes corresponding to time and agent into 1, because apparently conv2d can't deal with 2 extra axes
            # Reshape so that instead of [T,#agents,...] it's [T*#agents,...]
            actor_inputs_reshaped = tf.reshape(actor_inputs, [-1,GRID_SIZE,GRID_SIZE,2])       
            conv1_actor    =  layers.conv2d(inputs=actor_inputs_reshaped,   padding="SAME", num_outputs=RNN_SIZE//4,  kernel_size=[3, 3],   stride=1, data_format="NHWC", weights_initializer=w_init,activation_fn=tf.nn.relu)
            conv1a_actor   =  layers.conv2d(inputs=conv1_actor,   padding="SAME", num_outputs=RNN_SIZE//4,  kernel_size=[3, 3],   stride=1, data_format="NHWC", weights_initializer=w_init,activation_fn=tf.nn.relu)
            conv1a_actor_reshaped = tf.reshape(conv1a_actor, [-1,self.num_agents,conv1a_actor.shape[1]*conv1a_actor.shape[2]*conv1a_actor.shape[3]]) 
            # conv1a_actor_reshaped is now of shape [T, #agents, blah]
            # recon_error is of shape [T, #agents, blahblah]
            # We need to concatenate them along axis=2
            rnn_in_actor   = tf.concat([conv1a_actor_reshaped, recon_error], axis=2)
            # rnn_in_actor = tf.expand_dims(h1_actor,[1])
            # rnn_in_actor = tf.reshape(h1_actor, [actor_inputs.shape[0],self.num_agents,-1])#tf.reshape(h1_actor, [-1,self.num_agents,h1_actor.shape[1]])
            
            # print('rnn_in_actor.shape: ',rnn_in_actor.shape)


            routing_cell = RoutingRNN(RNN_SIZE, MSG_REPR_SIZE, self.num_agents, a_size, msg_length)
            # routing_cell = MyBasicRNNCell(100)
            seq_length = tf.shape(actor_inputs)[:1]
            # print('seq_length.shape: ',seq_length.shape)

            # msgs_in = tf.placeholder(shape=[1, self.num_agents, msg_length], dtype = tf.float32) #shape of message

            self.msgs_in = tf.placeholder(shape = [self.num_agents,msg_length], dtype = tf.float32)

            rnn_output, msgs_out = tf.nn.dynamic_rnn(routing_cell, rnn_in_actor, initial_state=self.msgs_in, dtype = tf.float32,time_major=True)
            policies = tf.squeeze(rnn_output)


        with tf.variable_scope('critic'):

            critic_inputs_reshaped = tf.reshape(critic_inputs, [-1,4,GRID_SIZE,GRID_SIZE])
            conv1_critic    =  layers.conv2d(inputs=critic_inputs_reshaped,   padding="SAME", num_outputs=RNN_SIZE//4,  kernel_size=[3, 3],   stride=1, data_format="NHWC", weights_initializer=w_init,activation_fn=tf.nn.relu)
            conv1a_critic   =  layers.conv2d(inputs=conv1_critic,   padding="SAME", num_outputs=RNN_SIZE//4,  kernel_size=[3, 3],   stride=1, data_format="NHWC", weights_initializer=w_init,activation_fn=tf.nn.relu)
            

            h1_critic = tf.reshape(conv1a_critic, [-1,self.num_agents,conv1a_critic.shape[1]*conv1a_critic.shape[2]*conv1a_critic.shape[3]])

            # flat_critic = layers.flatten(conv1a_critic)
            # h1_critic = tf.reshape(tf.nn.relu(flat_critic),[critic_inputs.shape[0],self.num_agents,-1])

            # msg_action = tf.concat([msg_input,actions_other],1)
            action_layer = layers.fully_connected(inputs=actions_other, num_outputs=ACTION_REPR_SIZE)
            hidden_input_critic = tf.concat([h1_critic, action_layer],2)

            h1_critic = layers.fully_connected(inputs=hidden_input_critic,  num_outputs=RNN_SIZE)

            
            values        = tf.squeeze(layers.fully_connected(inputs=h1_critic, num_outputs=1, weights_initializer=normalized_columns_initializer(1.0), biases_initializer=None, activation_fn=None))
       



        return policies, msgs_out, values

#TODO: modify this such that the recipient matrix is part of input (so it can change between time steps)

class RoutingRNN(tf.nn.rnn_cell.RNNCell):

    """
    RNN cell to be used for diff comms learning.

    One time-step evaluation of this RNN takes the as "input" the processed observations for each agent, and takes as "state" the messages TO each agent from last time step
    Returns as "output": policy
    Returns as "state": messages TO each agent from current time step (to be recieved at next time step, when state is fed back into this RNN cell)
    """

    def __init__(self, num_units, msg_repr_size, num_agents, a_size, msg_length, reuse=None):
        super(RoutingRNN, self).__init__(_reuse=reuse)
        self._num_units  = num_units
        self._msg_length = msg_length
        self._msg_repr_size = msg_repr_size
        self._a_size     = a_size
        # self._activation = activation or math_ops.tanh
        # self._linear = None

    @property
    def state_size(self):
        # return (1,self._num_agents, self._msg_length) #the leading 1 is because tf wants RNN inputs to be in shape (batch_size, time, etc...)
        return self._msg_length

    @property
    def output_size(self):
        
        return self._a_size

    def call(self, inputs_and_recon_error, state):
        """
        inputs_and_recon_error = observations for each agent after being processed by conv layers, concatenated with message reconstruction error
        state = messages TO each agent, generated at previous time step

        """
        # seperate back out original inputs and reconstruction error
        inputs = inputs_and_recon_error[:,:-self.state_size]
        recon_error = inputs_and_recon_error[:,-self.state_size:]

        # add reconstruction error to msg
        msg = state
        #begin processing messages into message representation
        msg_repr = layers.fully_connected(inputs=msg, num_outputs=self._msg_repr_size)

        # concatenate inputs and message representation
        hidden_input=tf.concat([inputs,msg_repr],1, name="hidden_input")
        
        # compute policy and pre_msg
        h1 = layers.fully_connected(inputs=hidden_input,  num_outputs=self._num_units, activation_fn=tf.nn.relu)
        policy_layer = layers.fully_connected(inputs=h1,  num_outputs=self._a_size, weights_initializer=normalized_columns_initializer(1./float(self._a_size)), biases_initializer=None, activation_fn=None)
        policy       = tf.nn.softmax(policy_layer,name="policy")
        msg_layer    = layers.fully_connected(inputs=h1, num_outputs=self._msg_length, weights_initializer=normalized_columns_initializer(1./float(self._a_size)), biases_initializer=None, activation_fn=None)
        pre_msg          = tf.nn.sigmoid(msg_layer) # these are messages FROM each agent (i.e. pre_msg[:,i]= message from agent i)
        
        # compute messages TO each agent (i.e. msg[:,i]=message that will become input to agent i at next time step)
        # Here we simply assume agent 1 sends to 2 and vise versa, so we simply reverse the messages along the agents dimension
        msg_out = tf.reverse(pre_msg, [0]) + recon_error #tf.matmul(recipient_mat,pre_msg)
        
            
        return policy, msg_out

# class MyBasicRNNCell(tf.nn.rnn_cell.RNNCell):
#     '''
#     Args:
#       inputs: `2-D` tensor with shape `[batch_size, input_size]`.
#       state: if `self.state_size` is an integer, this should be a `2-D Tensor`
#         with shape `[batch_size, self.state_size]`.  Otherwise, if
#         `self.state_size` is a tuple of integers, this should be a tuple
#         with shapes `[batch_size, s] for s in self.state_size`.
#     '''

#     def __init__(self, num_units, activation=None, reuse=None):
#         super(MyBasicRNNCell, self).__init__(_reuse=reuse)
#         self._num_units = num_units
        

#     @property
#     def state_size(self):
#         return self._num_units

#     @property
#     def output_size(self):
#         return self._num_units

#     def call(self, inputs, state):
#         """Most basic RNN: output = new_state = act(W * input + U * state + B)."""
        

#         output = layers.fully_connected(inputs, num_outputs = self._num_units)
#         msgs = tf.reverse(output,[0])
#         return output,msgs