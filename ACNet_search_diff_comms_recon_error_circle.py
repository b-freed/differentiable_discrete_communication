import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np
#parameters for training
GRAD_CLIP              = 100.0
RNN_SIZE               = 512
STATE_REPR_SIZE        = 512 #added 2 to incorporate position
HIDDEN_STATE_SIZE      = 100 # size of hidden state that gets conveyed back to the same agent at the next time step
msg_length             = 10 #number of elements in message
N_CHANNELS_ACTOR       = 5
N_CHANNELS_CRITIC      = 5
ENV_DIMENSION          = 2

# num_recieved_msgs      = 3  #number of neighbor agents we can get messages from

# def normalized_columns_initializer(std=1.0):
#     def _initializer(shape, dtype=None, partition_info=None):
#         out = np.random.randn(*shape).astype(np.float32)
#         out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
#         return tf.constant(out)
#     return _initializer

class ACNet:
    def __init__(self, scope, a_size, num_agents, trainer_A,trainer_C,TRAINING,GRID_SIZE,GLOBAL_NET_SCOPE):
        with tf.variable_scope(scope):
            #The input size may require more work to fit the interface.
            self.num_agents = num_agents
            self.actor_inputs = tf.placeholder(shape=[None,self.num_agents,N_CHANNELS_ACTOR,GRID_SIZE,GRID_SIZE], dtype=tf.float32)
            self.my_actor_inputs = tf.transpose(self.actor_inputs, perm=[0,1,3,4,2])
            self.critic_inputs = tf.placeholder(shape=[None,self.num_agents,N_CHANNELS_CRITIC,GRID_SIZE,GRID_SIZE], dtype=tf.float32)
            self.my_critic_inputs = tf.transpose(self.critic_inputs, perm=[0,1,3,4,2])
            self.reconstruction_error = tf.placeholder(shape=[None,self.num_agents,msg_length], dtype=tf.float32)
            #self.agents_pos = tf.placeholder(shape=[self.num_agents, ENV_DIMENSION], dtype=tf.float32)

            self.actions             = tf.placeholder(shape=[None, self.num_agents], dtype=tf.int32)
            self.actions_onehot      = tf.one_hot(self.actions, a_size, dtype=tf.float32)

            self.msgs_init      = np.zeros((self.num_agents,msg_length,2))
            self.rnn_state_init = np.zeros((self.num_agents,HIDDEN_STATE_SIZE))

            self.policies, self.rnn_state_out, self.msgs_out, self.values = self._build_net(self.my_actor_inputs, self.my_critic_inputs, self.reconstruction_error, a_size,GRID_SIZE)
                                                                                    
            if TRAINING:

                self.target_v            = tf.placeholder(tf.float32, [None,self.num_agents], 'Vtarget')
                self.advantages          = tf.placeholder(shape=[None,self.num_agents], dtype=tf.float32)
                self.responsible_outputs = tf.reduce_sum(self.policies * self.actions_onehot, [2])
                
                # Loss Functions
                with tf.name_scope('c_loss'):
                    self.c_loss    = tf.reduce_sum(tf.square(self.target_v - self.values))

                # something to encourage exploration
                self.entropy       = - tf.reduce_sum(self.policies * tf.log(tf.clip_by_value(self.policies,1e-10,1.0)))/self.num_agents
                self.policy_loss   = - tf.reduce_sum(tf.log(tf.clip_by_value(self.responsible_outputs,1e-15,1.0)) * self.advantages)
               
                with tf.name_scope('a_loss'):
                    self.a_loss          = self.policy_loss - self.entropy * 0.01  

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
                # TODO: fix this
                global_actor_vars        = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, GLOBAL_NET_SCOPE+'/actor')
                global_critic_vars       = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, GLOBAL_NET_SCOPE+'/critic')
                self.apply_a_grads       = trainer_A.apply_gradients(zip(a_grads, global_actor_vars))
                self.apply_c_grads       = trainer_C.apply_gradients(zip(c_grads, global_critic_vars))
        print("Hello World... From  "+str(scope))     # :)

    def _build_net(self,actor_inputs,critic_inputs,recon_error, a_size,GRID_SIZE):
        w_init   = layers.variance_scaling_initializer()

        with tf.variable_scope('actor'):

            actor_inputs_reshaped = tf.reshape(actor_inputs, [-1,GRID_SIZE,GRID_SIZE,N_CHANNELS_ACTOR])       
            conv1_actor    =  layers.conv2d(inputs=actor_inputs_reshaped,   padding="SAME", num_outputs=RNN_SIZE//4,  kernel_size=[3, 3],   stride=1, data_format="NHWC", weights_initializer=w_init,activation_fn=tf.nn.relu)
            conv1a_actor   =  layers.conv2d(inputs=conv1_actor,   padding="SAME", num_outputs=RNN_SIZE//4,  kernel_size=[3, 3],   stride=1, data_format="NHWC", weights_initializer=w_init,activation_fn=tf.nn.relu)
            conv1a_actor_reshaped   = tf.reshape(conv1a_actor, [-1,self.num_agents,conv1a_actor.shape[1]*conv1a_actor.shape[2]*conv1a_actor.shape[3]])
            #                        num_units, msg_repr_size,      num_agents, a_size, msg_length, hidden_state_size
            rnn_in_actor = tf.concat([conv1a_actor_reshaped, recon_error], axis=2)
            routing_cell = RoutingRNN(RNN_SIZE, self.num_agents, STATE_REPR_SIZE, a_size, msg_length, HIDDEN_STATE_SIZE)
            
            seq_length = tf.shape(actor_inputs)[:1]
            
            self.rnn_state_in = tf.placeholder(shape = [self.num_agents, HIDDEN_STATE_SIZE], dtype = tf.float32)
            self.msgs_in      = tf.placeholder(shape = [self.num_agents, msg_length,2], dtype = tf.float32)
            self.msgs_in_flat = tf.reshape(self.msgs_in, [self.num_agents, 2*msg_length])

            state_in = tf.concat([self.rnn_state_in, self.msgs_in_flat], axis = 1) # "state_in corresponds to the rnn state for each agent combined with the messages to each agent from all other agents"


            rnn_output, state_out = tf.nn.dynamic_rnn(routing_cell, rnn_in_actor, initial_state=state_in, dtype = tf.float32,time_major=True)
            policies = tf.squeeze(rnn_output)
            rnn_state_out = state_out[:,:HIDDEN_STATE_SIZE] # first HIDDEN_STATE_SIZE elements of state_out correspond to rnn hidden state
            msgs_out_flat      = state_out[:,HIDDEN_STATE_SIZE:] # The rest of the elements are messages TO each agent 
            msgs_out     = tf.reshape(msgs_out_flat, [self.num_agents, msg_length, 2])
            #agents_pos[[1,2]] = agents_pos[[2,1]]
            #agents_pos = tf.gather(agents_pos, [0,2,1,3], axis = 0)
            #msgs_out      = tf.concat([msgs_out, agents_pos], axis = -1)

        with tf.variable_scope('critic'):

            critic_inputs_reshaped = tf.reshape(critic_inputs, [-1,N_CHANNELS_CRITIC,GRID_SIZE,GRID_SIZE])
            conv1_critic    =  layers.conv2d(inputs=critic_inputs_reshaped,   padding="SAME", num_outputs=RNN_SIZE//4,  kernel_size=[3, 3],   stride=1, data_format="NHWC", weights_initializer=w_init,activation_fn=tf.nn.relu)
            conv1a_critic   =  layers.conv2d(inputs=conv1_critic,   padding="SAME", num_outputs=RNN_SIZE//4,  kernel_size=[3, 3],   stride=1, data_format="NHWC", weights_initializer=w_init,activation_fn=tf.nn.relu)
            h1_critic = tf.reshape(conv1a_critic, [-1,self.num_agents,conv1a_critic.shape[1]*conv1a_critic.shape[2]*conv1a_critic.shape[3]])
            h2_critic = layers.fully_connected(inputs=h1_critic,  num_outputs=RNN_SIZE)
            values        = tf.squeeze(layers.fully_connected(inputs=h2_critic, num_outputs=1,activation_fn=None))

           
       
       
        return policies, rnn_state_out, msgs_out, values

class RoutingRNN(tf.nn.rnn_cell.RNNCell):

    """
    RNN cell to be used for diff comms learning.

    One time-step evaluation of this RNN takes the as "input" the processed observations for each agent, and takes as "state" the messages TO each agent from last time step.
    messages (which are of shape [# agents, msg_length, 2]) are flattened to shape [# agents, msg_length, 2] so they can be concatenated with hidden state
    We allow each agent to send a message to itself, accomplishing the function of a typical RNN state.  

    Returns as "output": policy
    Returns as "state": messages TO each agent from current time step (to be recieved at next time step, when state is fed back into this RNN cell)
    """

    def __init__(self, num_units, num_agents, state_repr_size, a_size, msg_length, hidden_state_size, reuse=None):
        super(RoutingRNN, self).__init__(_reuse=reuse)
        self._num_units  = num_units
        self._msg_length = msg_length # this is message length for 1 particular agent
        self._state_repr_size = state_repr_size
        self._a_size     = a_size
        self.num_agents = num_agents
        self._hidden_state_size = hidden_state_size
        #compute length of message input to all agents
        # self._msg_input_length = (self.num_agents - 1)*2*self._msg_length

        

    @property
    def state_size(self):
        # the state is composed of the hidden state, and a message to be sent TO each agent
        return self._hidden_state_size + 2*self._msg_length

    @property
    def output_size(self):
        
        return self._a_size

    def rotate_by_angle(self, msg, theta):
        '''
        msg: array of shape [#agents, msg_length, 2] representing msg to be perturbed (rotated)
        theta: array of shape [#agents, msg_length] representing angle to perturb (rotate) msg by

        this is a slightly unfortunate way to implement this, but I didn't want to have to futz around with array dimensions
        '''

        x = msg[:,:,0]
        y = msg[:,:,1]

        cos_error = tf.math.cos(theta)
        sin_error = tf.math.sin(theta)

        x_new = cos_error*x - sin_error*y
        y_new = sin_error*x + cos_error*y

        msg_rotated = tf.concat([tf.expand_dims(x_new,-1), tf.expand_dims(y_new,-1)], axis = 2)
        
        assert msg_rotated.shape == msg.shape

        return msg_rotated

    def call(self, inputs_and_recon_error, state_in):
        """

        INPUTS:

        inputs_and_recon_error = concatenation of: 
            [observations for each agent after being processed by conv layers, 
            reconstruction error for messages TO each agent, generated at previous time step]
        state_in = concatenation of: 
            [hidden_state, which each agent passes directly to itself at the next time step,
            msg: messages TO each agent, generated at previous time step, having already been encoded and decoded

        OUTPUTS:

        policies: # timesteps x #agents x #actions array of action prob dists
        state_out: concatenation of:
            [hidden_state_out]

        """
       
        # elements of inputs_and_recon_error up to self._msg_length from end is the input (processed observation of each agent)
        inputs = inputs_and_recon_error[:,:-self._msg_length]
        # final self._msg_input_length of inputs_and_recon_error is recon error
        recon_error = inputs_and_recon_error[:,-self._msg_length:]

        assert recon_error.shape[0] == self.num_agents 
        assert recon_error.shape[1] == self._msg_length

        state_repr       = layers.fully_connected(inputs=state_in, num_outputs=self._state_repr_size)
        hidden_input     = tf.concat([inputs,state_repr],1, name="hidden_input")
        h1               = layers.fully_connected(inputs=hidden_input,  num_outputs=self._num_units, activation_fn=tf.nn.relu)
        policy           = layers.fully_connected(inputs=h1,  num_outputs=self._a_size, activation_fn=tf.nn.softmax)
        hidden_state_out = layers.fully_connected(inputs=h1, num_outputs = self._hidden_state_size, activation_fn = tf.nn.relu)
        # Compute the messages FROM each agent
        msg_layer    = layers.fully_connected(inputs=h1, num_outputs=2*self._msg_length, activation_fn=None)
        msg_layer_reshaped = tf.reshape(msg_layer, [self.num_agents,self._msg_length, 2]) # Because I love you Jeff <3
        pre_msg          = tf.math.l2_normalize(msg_layer_reshaped, axis = -1) # pre_msg should have norm 1 along final axis
        msg_out = tf.reverse(pre_msg, [0]) 
        msg_out_with_error = self.rotate_by_angle(msg_out, recon_error)
        msg_out_with_error_flat = tf.reshape(msg_out_with_error, [self.num_agents, 2*self._msg_length])

        
        # combine hidden_state_out and msgs_out to create state_out
        state_out        = tf.concat([hidden_state_out,msg_out_with_error_flat],axis = 1)
            
        return policy, state_out
