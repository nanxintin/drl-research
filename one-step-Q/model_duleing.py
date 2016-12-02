import tensorflow as tf
from keras import backend as K
from keras.layers import Convolution2D, Flatten, Dense, Input, Lambda, merge
from keras.models import Model

def build_network(num_actions, agent_history_length, resized_width, resized_height):
  with tf.device("/cpu:0"):
    state = tf.placeholder("float", [None, agent_history_length, resized_width, resized_height])
    inputs = Input(shape=(agent_history_length, resized_width, resized_height,))
    model = Convolution2D(nb_filter=32, nb_row=8, nb_col=8, subsample=(4,4), activation='relu', border_mode='same')(inputs)
    model = Convolution2D(nb_filter=64, nb_row=4, nb_col=4, subsample=(2,2), activation='relu', border_mode='same')(model)
    model = Convolution2D(nb_filter=64, nb_row=3, nb_col=3, subsample=(1,1), activation='relu', border_mode='same')(model)
    model = Flatten()(model)
    
    # state value tower - V
    state_value = Dense(256, activation='relu')(model)
    state_value = Dense(1)(state_value)
    state_value = Lambda(lambda s: K.expand_dims(s[:, 0], dim=-1), output_shape=(num_actions,))(state_value)
    # action advantage tower - A
    action_advantage = Dense(256, activation='relu')(model)
    action_advantage = Dense(output_dim=num_actions)(action_advantage)
    action_advantage = Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True), output_shape=(num_actions,))(action_advantage)
    # merge to state-action value function Q
    state_action_value = merge([state_value, action_advantage], mode='sum')
    m = Model(input=inputs, output=state_action_value)
    #model.compile(rmsprop(lr=self.learning_rate), "mse")
    m.summary()
  return state, m