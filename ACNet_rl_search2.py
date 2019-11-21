import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np
#parameters for training
GRAD_CLIP              = 100.0
RNN_SIZE               = 512
STATE_REPR_SIZE        = 512
HIDDEN_STATE_SIZE      = 100
ACTION_REPR_SIZE       = 50
msg_length             = 40 #number of elements in message

# num_recieved_msgs      = 3  #number of neighbor agents we can get messages from

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
            #The input size may require more work to fit the interface.
            self.num_agents = num_agents
            self.actor_inputs = tf.placeholder(shape=[None,5,GRID_SIZE,GRID_SIZE], dtype=tf.float32)
            self.my_actor_inputs = tf.transpose(self.actor_inputs, perm=[0,2,3,1])
            self.critic_inputs = tf.placeholder(shape=[None,5,GRID_SIZE,GRID_SIZE], dtype=tf.float32)
            self.my_critic_inputs = tf.transpose(self.critic_inputs, perm=[0,2,3,1])
            self.msg_inputs = tf.placeholder(shape = [None, self.num_agents - 1, msg_length], dtype=tf.float32)
            self.msg_inputs_flat = layers.flatten(self.msg_inputs)
            self.actions_other             = tf.placeholder(shape=[None], dtype=tf.int32)
            self.actions_other_onehot      = tf.one_hot(self.actions_other, a_size, dtype=tf.float32)
            self.actions_other_flat = layers.flatten(self.actions_other_onehot)
            # self.msg_input = tf.placeholder(shape=[None, num_recieved_msgs, msg_length], dtype = tf.float32) #shape of message
            # self.myinput = tf.transpose(self.inputs, perm=[0,2,3,1])
             # args: actor_inputs, msg_inputs, critic_inputs,RNN_SIZE,TRAINING,a_size
            
            #    policy,      P_msg,      value,      state_out_actor ,     state_in_actor,      state_init_actor
            self.policy, self.P_msg, self.value, self.state_out_actor, self.state_init_actor = self._build_net(self.my_actor_inputs,
                                                                                                               self.msg_inputs_flat,
                                                                                                               self.my_critic_inputs,
                                                                                                               self.actions_other_flat, 
                                                                                                               RNN_SIZE,
                                                                                                               TRAINING,
                                                                                                               a_size)

            # self.policy, self.value, self.state_out, self.state_in, self.state_init, self.blocking, self.on_goal,
            # self.valids = self._build_net(self.myinput,self.goal_pos, self.msg_input,RNN_SIZE,TRAINING,a_size)
            if TRAINING:
                self.actions                = tf.placeholder(shape=[None], dtype=tf.int32)
                self.actions_onehot         = tf.one_hot(self.actions, a_size, dtype=tf.float32)
                self.msgs_binary            = tf.placeholder(shape = [None, msg_length], dtype = tf.float32)
                self.train_valid            = tf.placeholder(shape=[None,a_size], dtype=tf.float32)
                self.target_v               = tf.placeholder(tf.float32, [None], 'Vtarget')
                self.action_advantages      = tf.placeholder(shape=[None], dtype=tf.float32)
                self.msg_advantages         = tf.placeholder(shape=[None], dtype=tf.float32)
                self.target_blockings       = tf.placeholder(tf.float32, [None])
                self.target_on_goals        = tf.placeholder(tf.float32, [None])
                self.responsible_outputs    = tf.reduce_sum(self.policy * self.actions_onehot, [1])
                self.train_value            = tf.placeholder(tf.float32, [None])
                self.optimal_actions        = tf.placeholder(tf.int32,[None])
                self.optimal_actions_onehot = tf.one_hot(self.optimal_actions, a_size, dtype=tf.float32)

                
                # prob_msg_bits = self.msgs_binary*tf.log(tf.clip_by_value(self.P_msg,1e-15,1.0)) + (1-self.msgs_binary)*tf.log(tf.clip_by_value(1 - self.P_msg,1e-15,1.0)) 
                self.log_prob_msgs       = tf.reduce_sum(self.msgs_binary*tf.log(tf.clip_by_value(self.P_msg,1e-15,1.0)) + (1-self.msgs_binary)*tf.log(tf.clip_by_value(1 - self.P_msg,1e-15,1.0)))

                #Loss Functions
                with tf.name_scope('c_loss'):
                    self.c_loss    = tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value, shape=[-1])))
                
                self.entropy        = - tf.reduce_sum(self.policy * tf.log(tf.clip_by_value(self.policy,1e-10,1.0)))
                self.msg_entropy    = - tf.reduce_sum(self.P_msg * tf.log(tf.clip_by_value(self.P_msg, 1e-15,1.0)) + (1-self.P_msg) * (tf.log(tf.clip_by_value(1 - self.P_msg, 1e-15,1.0))))
                self.policy_loss    = - tf.reduce_sum(tf.log(tf.clip_by_value(self.responsible_outputs,1e-15,1.0)) * self.action_advantages)
                self.msg_loss      = - tf.reduce_sum(self.log_prob_msgs * self.msg_advantages)
                
                with tf.name_scope('a_loss'):
                    self.a_loss          = self.policy_loss - self.entropy * 0.01  + self.msg_loss 
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
        print("Hello World... From  "+str(scope))     # :)

    def _build_net(self,actor_inputs, msg_inputs, critic_inputs, actions_other,RNN_SIZE,TRAINING,a_size):
        w_init   = layers.variance_scaling_initializer()
        
        with tf.variable_scope('actor'):
            conv1_actor    =  layers.conv2d(inputs=actor_inputs,   padding="SAME", num_outputs=RNN_SIZE//4,  kernel_size=[3, 3],   stride=1, data_format="NHWC", weights_initializer=w_init,activation_fn=tf.nn.relu)
            conv1a_actor   =  layers.conv2d(inputs=conv1_actor,   padding="SAME", num_outputs=RNN_SIZE//4,  kernel_size=[3, 3],   stride=1, data_format="NHWC", weights_initializer=w_init,activation_fn=tf.nn.relu)
            
            flat    = layers.flatten(conv1a_actor)

            state_init_actor = np.zeros((1,HIDDEN_STATE_SIZE))
            # rnn_input: a concatenation of the processed observations and message input
            rnn_input = tf.expand_dims(tf.concat([flat, msg_inputs], axis = 1), [0])
            seq_length = tf.shape(rnn_input)[:1]
            self.state_in_actor = tf.placeholder(shape = (1,HIDDEN_STATE_SIZE), dtype = tf.float32)
            rnn_cell = RNN_Cell(RNN_SIZE, self.num_agents, STATE_REPR_SIZE, a_size, msg_length, HIDDEN_STATE_SIZE)
            # rnn_cell = tf.nn.rnn_cell.BasicRNNCell(num_units = RNN_SIZE, activation = 'relu')
            rnn_output, state_out_actor = tf.nn.dynamic_rnn(rnn_cell, rnn_input, initial_state=self.state_in_actor, sequence_length = seq_length, dtype = tf.float32,time_major=False)

            

            rnn_out = tf.reshape(rnn_output, [-1, RNN_SIZE])
            
            policy           = layers.fully_connected(inputs=rnn_out,  num_outputs=a_size, activation_fn=tf.nn.softmax)
            P_msg            = layers.fully_connected(inputs=rnn_out,  num_outputs=msg_length, activation_fn=tf.nn.sigmoid)
        
        with tf.variable_scope('critic'):
            conv1_critic    =  layers.conv2d(inputs=critic_inputs,   padding="SAME", num_outputs=RNN_SIZE//4,  kernel_size=[3, 3],   stride=1, data_format="NHWC", weights_initializer=w_init,activation_fn=tf.nn.relu)
            conv1a_critic   =  layers.conv2d(inputs=conv1_critic,   padding="SAME", num_outputs=RNN_SIZE//4,  kernel_size=[3, 3],   stride=1, data_format="NHWC", weights_initializer=w_init,activation_fn=tf.nn.relu)
            
            flat_critic     = tf.nn.relu(layers.flatten(conv1a_critic))

            action_layer = layers.fully_connected(inputs=actions_other, num_outputs=ACTION_REPR_SIZE)
            hidden_input_critic = tf.concat([flat_critic, action_layer],1)

            h1_critic = layers.fully_connected(inputs=flat_critic,  num_outputs=RNN_SIZE)
            h2_critic = layers.fully_connected(inputs=h1_critic,  num_outputs=RNN_SIZE)
            
            value = layers.fully_connected(inputs=h2_critic, num_outputs=1, weights_initializer=normalized_columns_initializer(1.0), biases_initializer=None, activation_fn=None)


        return policy, P_msg, value, state_out_actor, state_init_actor


# this is just to make NN strucutre identical to diff comms version


class RNN_Cell(tf.nn.rnn_cell.RNNCell):

    """
    RNN cell to be used for diff comms learning.

    One time-step evaluation of this RNN takes the as "input" the processed observations for each agent, and takes as "state" the messages TO each agent from last time step.
    We allow each agent to send a message to itself, accomplishing the function of a typical RNN state.  

    Returns as "output": policy
    Returns as "state": messages TO each agent from current time step (to be recieved at next time step, when state is fed back into this RNN cell)
    """

    def __init__(self, num_units, num_agents, state_repr_size, a_size, msg_length, hidden_state_size, reuse=None):
        super(RNN_Cell, self).__init__(_reuse=reuse)
        self._num_units  = num_units
        self._msg_length = msg_length # this is message length for 1 particular agent
        self._state_repr_size = state_repr_size
        self._a_size     = a_size
        self.num_agents = num_agents
        self._hidden_state_size = hidden_state_size
        #compute length of message input to all agents
        self._msg_input_length = (self.num_agents - 1)*self._msg_length
        

    @property
    def state_size(self):
        # the state is composed of the hidden state, and a message to be sent TO each agent
        return self._hidden_state_size

    @property
    def output_size(self):
        
        return self._num_units

    def call(self, inputs_and_msg, state_in):
        """

        # in this version, the RNN cell will take:
            # - inputs_and_msg: a concatenation of the processed observations and message input
            # - state_in: rnn hidden state input

        # and return:
            # - output:  the activations that will get turned into policy and P_msg
            # - state_out: rnn hidden state output

        """
       
        # elements of inputs_and_recon_error up to self._msg_input_length from end is the input (processed observation of each agent)
        inputs = inputs_and_msg[:,:-self._msg_input_length]
        msg = inputs_and_msg[:,-self._msg_input_length:]

        state_repr = layers.fully_connected(tf.concat([state_in,msg], axis = 1),num_outputs=self._state_repr_size)

        hidden_input     = tf.concat([inputs,state_repr],1, name="hidden_input")
        outputs               = layers.fully_connected(inputs=hidden_input,  num_outputs=self._num_units, activation_fn=tf.nn.relu)
        state_out = layers.fully_connected(inputs=outputs, num_outputs = self._hidden_state_size, activation_fn = tf.nn.relu)
        
            
        return outputs, state_out