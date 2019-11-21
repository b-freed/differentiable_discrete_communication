import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np
#parameters for training
GRAD_CLIP              = 1000.0
KEEP_PROB1             = .9 # was 0.5
KEEP_PROB2             = .9 # was 0.7
RNN_SIZE               = 512
ACTION_REPR_SIZE       = 50
MSG_REPR_SIZE          = 100
num_agents             = 2
msg_length             = 40

#Used to initialize weights for policy and value output layers (Do we need to use that? Maybe not now)
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

class ACNet:
    def __init__(self, scope, a_size, trainer_A,trainer_C,TRAINING,GRID_SIZE,GLOBAL_NET_SCOPE):
        # print('TRAINING??????: ',TRAINING)
        with tf.variable_scope(scope):
            #The input size may require more work to fit the interface.
            self.actor_inputs = tf.placeholder(shape=[None,2,GRID_SIZE,GRID_SIZE], dtype=tf.float32)
            self.my_actor_input = tf.transpose(self.actor_inputs, perm=[0,2,3,1])

            self.critic_inputs = tf.placeholder(shape=[None,4,GRID_SIZE,GRID_SIZE], dtype=tf.float32)
            self.my_critic_input = tf.transpose(self.critic_inputs, perm=[0,2,3,1])

            self.msg_input = tf.placeholder(shape=[None, num_agents - 1, msg_length], dtype = tf.float32) #shape of message
            self.msg_input_flat = layers.flatten(self.msg_input)

            self.actions_other             = tf.placeholder(shape=[None], dtype=tf.int32)
            self.actions_other_onehot      = tf.one_hot(self.actions_other, a_size, dtype=tf.float32)
            self.actions_other_flat = layers.flatten(self.actions_other_onehot)
            #    policy,      value,      state_out_actor,      state_in_actor,      state_init_actor,      state_out_critic ,     state_in_critic,      state_init_critic,      blocking,      on_goal, policy_sig
            self.policy, self.P_msg, self.value, self.blocking, self.on_goal, self.valids = self._build_net(self.my_actor_input,
                                                                                                self.my_critic_input,
                                                                                                self.msg_input_flat,
                                                                                                self.actions_other_flat,
                                                                                                RNN_SIZE,
                                                                                                TRAINING,
                                                                                                a_size)
        
            if TRAINING:
                self.actions             = tf.placeholder(shape=[None], dtype=tf.int32)
                self.actions_onehot      = tf.one_hot(self.actions, a_size, dtype=tf.float32)
                self.msg_binary          = tf.placeholder(shape = [None, msg_length], dtype = tf.float32)
                self.train_valid         = tf.placeholder(shape=[None,a_size], dtype=tf.float32)
                self.target_v            = tf.placeholder(tf.float32, [None], 'Vtarget')
                self.action_advantages   = tf.placeholder(shape=[None], dtype=tf.float32)
                self.msg_advantages      = tf.placeholder(shape=[None], dtype=tf.float32)
                self.target_blockings    = tf.placeholder(tf.float32, [None])
                self.target_on_goals     = tf.placeholder(tf.float32, [None])
                self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])
                # self.responsible_msgs    = tf.reduce_sum(self.P_msg  * self.msg_binary,     [1])
                self.log_prob_msgs       = tf.reduce_sum(self.msg_binary*tf.log(tf.clip_by_value(self.P_msg,1e-15,1.0)) + (1-self.msg_binary)*tf.log(tf.clip_by_value(1 - self.P_msg,1e-15,1.0)), axis=1) 

                # Loss Functions
                with tf.name_scope('c_loss'):
                    self.c_loss    = tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value, shape=[-1])))

                # something to encourage exploration
                self.entropy       = - tf.reduce_sum(self.policy * tf.log(tf.clip_by_value(self.policy,1e-10,1.0)))
                # self.msg_entropy   = - tf.reduce_sum(self.P_msg  * tf.log(tf.clip_by_value(self.P_msg, 1e-10,1.0)))
                self.msg_entropy   = - tf.reduce_sum(self.P_msg * tf.log(tf.clip_by_value(self.P_msg, 1e-15,1.0)) + (1-self.P_msg) * (tf.log(tf.clip_by_value(1 - self.P_msg, 1e-15,1.0))))
                self.policy_loss   = - tf.reduce_sum(tf.log(tf.clip_by_value(self.responsible_outputs,1e-15,1.0)) * self.action_advantages)
                # self.msg_loss      = - tf.reduce_sum(tf.log(tf.clip_by_value(self.responsible_msgs,   1e-15,1.0)) * self.msg_advantages)
                self.msg_loss      = - tf.reduce_sum(self.log_prob_msgs * self.msg_advantages)
                self.valid_loss    = - tf.reduce_sum(tf.log(tf.clip_by_value(self.valids,1e-10,1.0)) *\
                                    self.train_valid+tf.log(tf.clip_by_value(1-self.valids,1e-10,1.0)) * (1-self.train_valid))

                self.blocking_loss = - tf.reduce_sum(self.target_blockings*tf.log(tf.clip_by_value(self.blocking,1e-10,1.0))\
                                          +(1-self.target_blockings)*tf.log(tf.clip_by_value(1-self.blocking,1e-10,1.0)))
                self.on_goal_loss   = - tf.reduce_sum(self.target_on_goals*tf.log(tf.clip_by_value(self.on_goal,1e-10,1.0))\
                                      +(1-self.target_on_goals)*tf.log(tf.clip_by_value(1-self.on_goal,1e-10,1.0)))
                # self.msg_loss      = - tf.reduce_sum(self.log_prob_msgs  * tf.expand_dims(self.msg_advantages, axis = 1))
                with tf.name_scope('a_loss'):
                    self.a_loss          = self.policy_loss + 0.5*self.valid_loss \
                                    - self.entropy * 0.01 + 0.25*self.blocking_loss + self.msg_loss

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

#             self.homogenize_weights = update_target_graph(str(scope)+'/qvaluesB', str(scope)+'/qvalues')
        print("Hello World... From  "+str(scope))     # :)

    def _build_net(self,actor_inputs,critic_inputs,msg_input,actions_other,RNN_SIZE,TRAINING,a_size):
        w_init   = layers.variance_scaling_initializer()

        with tf.variable_scope('actor'):        
            conv1_actor    =  layers.conv2d(inputs=actor_inputs,   padding="SAME", num_outputs=RNN_SIZE//4,  kernel_size=[3, 3],   stride=1, data_format="NHWC", weights_initializer=w_init,activation_fn=tf.nn.relu)
            conv1a_actor   =  layers.conv2d(inputs=conv1_actor,   padding="SAME", num_outputs=RNN_SIZE//4,  kernel_size=[3, 3],   stride=1, data_format="NHWC", weights_initializer=w_init,activation_fn=tf.nn.relu)
            
            flat_actor     = tf.nn.relu(layers.flatten(conv1a_actor))
            msg_layer_actor = layers.fully_connected(inputs=msg_input, num_outputs=MSG_REPR_SIZE)
            hidden_input_actor=tf.concat([flat_actor,msg_layer_actor],1)
            # hidden_input_actor = flat_actor # goal is specified by input to conv layers so we don't need to concatenate anything 
            h1_actor = layers.fully_connected(inputs=hidden_input_actor,  num_outputs=RNN_SIZE, activation_fn=tf.nn.relu)
            # h1_actor = layers.fully_connected(inputs=flat_actor,  num_outputs=RNN_SIZE)
            policy_layer = layers.fully_connected(inputs=h1_actor, num_outputs=a_size,weights_initializer=normalized_columns_initializer(1./float(a_size)), biases_initializer=None, activation_fn=None)
            msg_layer    = layers.fully_connected(inputs=h1_actor, num_outputs=msg_length,weights_initializer=normalized_columns_initializer(1./float(a_size)), biases_initializer=None, activation_fn=None)
            policy       = tf.nn.softmax(policy_layer)
            P_msg          = tf.nn.sigmoid(msg_layer)
            policy_sig   = tf.sigmoid(policy_layer)
            # value        = layers.fully_connected(inputs=rnn_out, num_outputs=1, weights_initializer=normalized_columns_initializer(1.0), biases_initializer=None, activation_fn=None)
            blocking      = layers.fully_connected(inputs=h1_actor, num_outputs=1, weights_initializer=normalized_columns_initializer(1.0), biases_initializer=None, activation_fn=tf.sigmoid)
            on_goal      = layers.fully_connected(inputs=h1_actor, num_outputs=1, weights_initializer=normalized_columns_initializer(1.0), biases_initializer=None, activation_fn=tf.sigmoid)
        with tf.variable_scope('critic'):
            conv1_critic    =  layers.conv2d(inputs=critic_inputs,   padding="SAME", num_outputs=RNN_SIZE//4,  kernel_size=[3, 3],   stride=1, data_format="NHWC", weights_initializer=w_init,activation_fn=tf.nn.relu)
            conv1a_critic   =  layers.conv2d(inputs=conv1_critic,   padding="SAME", num_outputs=RNN_SIZE//4,  kernel_size=[3, 3],   stride=1, data_format="NHWC", weights_initializer=w_init,activation_fn=tf.nn.relu)
            flat_critic     = tf.nn.relu(layers.flatten(conv1a_critic))\

            # msg_action = tf.concat([msg_input,actions_other],1)
            action_layer = layers.fully_connected(inputs=actions_other, num_outputs=ACTION_REPR_SIZE)
            hidden_input_critic = tf.concat([flat_critic, action_layer],1)

            h1_critic = layers.fully_connected(inputs=hidden_input_critic,  num_outputs=RNN_SIZE)

            
            value        = layers.fully_connected(inputs=h1_critic, num_outputs=1, weights_initializer=normalized_columns_initializer(1.0), biases_initializer=None, activation_fn=None)
       



        return policy, P_msg, value, blocking, on_goal, policy_sig
