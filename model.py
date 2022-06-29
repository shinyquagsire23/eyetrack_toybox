import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.constraints import Constraint

from globals import *

class EchoRNNCell(keras.layers.Layer):
    '''
    Used to keep a recurrent state for previous values of any Tensor
    '''
    def __init__(self, dims=16, batch_size=Globals.TRAIN_BATCH_SIZE, **kwargs):
        self.state_size = dims
        self.stateful=True
        self.batch_size = batch_size
        super(EchoRNNCell, self).__init__(**kwargs)

    def build(self, input_shape):
        #self.state_size = input_shape[0]
        self.built = True

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return np.ones((batch_size, self.state_size)) * 0.0

    def call(self, inputs, states):
        prev_output = states[0]
        
        # We return the current input, and store the current input
        return inputs, [inputs]

class EchoRNN(keras.layers.RNN):

    def __init__(self, dims=16, **kwargs):
        super(EchoRNN, self).__init__(EchoRNNCell(dims, **kwargs), stateful=True, unroll=True)

class MinMaxConstraint(Constraint):
    """constrain model weights between [x_min, x_max]."""
    def __init__(self, x_min=0.0, x_max=1.0):
        super().__init__()
        self.x_min = x_min
        self.x_max = x_max
        # TODO: add sanity check if x_max == x_min or w_max == w_min
    
    def __call__(self, w):
        w_min = tf.minimum(tf.math.reduce_min(w), self.x_min)
        w_max = tf.maximum(tf.math.reduce_max(w), self.x_max)
        scale = (self.x_max - self.x_min) / (w_max - w_min)
        m = self.x_min - w_min * scale
        w = w * scale
        return w + m

class EyeNet(tf.keras.Model):

    def __init__(self,input_shape=(288,384,1),batch_size=Globals.TRAIN_BATCH_SIZE):
        super().__init__()

        self.patch_size = Globals.NETWORK_PATCH_SIZE

        def relu_clipped(x):
            return K.relu(x, alpha=0.05, max_value=1.0)
        def relu_clipped_1(x):
            return K.relu(x, alpha=0.05, max_value=1.1)

        #self._train_epoch_num_tensor = tf.Variable(0, dtype=tf.int64, trainable=False, name="_train_epoch_num_tensor")

        #self.boxes = None
        self.batch_size = batch_size
        self.box_indices = None
        self.input_layer = layers.Layer(name="input_layer", input_shape=(None, input_shape[0], input_shape[1], input_shape[2]))
        self.fe_initial_downsample = layers.AveragePooling2D(pool_size=(4, 4))
        
        # Initial feature extractor
        self.fe_conv_1 = layers.Conv2D(filters=32, kernel_size=(7,7), activation='tanh',input_shape=input_shape[1:], name="fe_conv_1")
        self.fe_conv_1_batchnorm = layers.BatchNormalization()
        self.fe_pool_1 = layers.MaxPool2D()
        self.fe_conv_2 = layers.Conv2D(filters=64, kernel_size=(5,5), activation='tanh', input_shape=input_shape[1:], name="fe_conv_2")
        self.fe_conv_2_batchnorm = layers.BatchNormalization()
        self.fe_pool_2 = layers.MaxPool2D()
        self.fe_conv_3 = layers.Conv2D(filters=128, kernel_size=(3,3), activation='tanh', input_shape=input_shape[1:], name="fe_conv_3")
        self.fe_conv_3_batchnorm = layers.BatchNormalization()
        self.fe_pool_3 = layers.MaxPool2D()
        self.fe_conv_4 = layers.Conv2D(filters=128, kernel_size=(3,3), activation='tanh', input_shape=input_shape[1:], name="fe_conv_4")
        self.fe_conv_4_batchnorm = layers.BatchNormalization()
        self.fe_conv_5 = layers.Conv2D(filters=4, kernel_size=(1,1), activation='tanh', input_shape=input_shape[1:], name="fe_conv_5")
        self.fe_conv_5_batchnorm = layers.BatchNormalization()
        #self.fe_global_pool = layers.GlobalMaxPooling2D(name="fe_global_pool")
        self.fe_predense_flatten = layers.Flatten()
        self.fe_dense = layers.Dense(1024, activation='tanh', name="fe_dense")
        self.fe_dense__2 = layers.Dense(units=192, activation='tanh', name="fe_dense__2")
        self.fe_reshape_predense = layers.Reshape((Globals.NETWORK_RECURRENT_SIZE, -1))
        self.fe_dense_2 = layers.Dense(units=1, activation='tanh', name="fe_dense_2")
        self.fe_dense_3 = layers.Dense(Globals.NETWORK_RECURRENT_SIZE, activation='tanh', name="fe_dense_3")
        self.fe_reshape_postdense = layers.Reshape((Globals.NETWORK_RECURRENT_SIZE,))
        
        #self.fe_rnn_cell_reshape_1 = layers.Reshape((1, 8))
        #self.fe_rnn_cell_1 = tf.keras.layers.RNN(tf.keras.layers.LSTMCell(units=8, activation = 'tanh'), return_sequences = True)
        self.fe_flatten = layers.Flatten()
        self.fe_dense_4 = layers.Dense(units=2048, activation='tanh', name="fe_dense4")
        self.fe_dense_5 = layers.Dense(units=512, activation="tanh", name="fe_dense5")
        self.fe_dense_6 = layers.Dense(units=128, activation="tanh", name="fe_dense6")
        self.fe_dense_7 = layers.Dense(units=64, activation="tanh", name="fe_dense7")
        #self.fe_dense_4_2 = layers.Dense(64, activation='tanh')
        #self.fe_out = layers.Dense(5, activation='linear', kernel_constraint=MinMaxConstraint(0.1, 1.0), bias_constraint=MinMaxConstraint(-1.0, 1.0))
        self.fe_out = layers.Dense(Globals.NETWORK_RECURRENT_SIZE, activation='tanh')
        self.fe_out_2 = layers.Dense(Globals.NETWORK_RECURRENT_SIZE, activation='tanh')

        #self.fe_indirect_access = tf.Variable(tf.ones((self.batch_size, Globals.NETWORK_RECURRENT_SIZE)) * 0.5, trainable=False, name="fe_indirect_access")
        #self.outputs_indirect_access = tf.Variable(tf.ones((self.batch_size, Globals.NETWORK_OUTPUT_SIZE)) * 0.5, trainable=False, name="outputs_indirect_access")
        #self.box_indirect_access = tf.Variable(tf.ones((self.batch_size, Globals.NETWORK_OUTPUT_SIZE)) * 0.5, trainable=False, name="box_indirect_access")

        self.fe_final_concat = layers.Concatenate(axis=1)
        self.fe_x_concat = layers.Concatenate(axis=1)

        self.image_crop_reshape = layers.Reshape((self.patch_size, self.patch_size, 1))
        self.image_crop = layers.Layer(input_shape=(self.patch_size,self.patch_size), name="image_crop")

        # Final feature extractor
        self.ffe_conv_1 = layers.Conv2D(filters=64, kernel_size=(5,5), activation='tanh', input_shape=input_shape[1:], name="ffe_conv_1")
        self.ffe_conv_1_batchnorm = layers.BatchNormalization()
        self.ffe_pool_1 = layers.MaxPool2D()

        self.ffe_conv_2 = layers.Conv2D(filters=128, kernel_size=(3,3), activation='tanh', input_shape=input_shape[1:], name="ffe_conv_2")
        self.ffe_conv_2_batchnorm = layers.BatchNormalization()
        self.ffe_pool_2 = layers.MaxPool2D()

        self.ffe_conv_3 = layers.Conv2D(filters=128, kernel_size=(3,3), activation='tanh', input_shape=input_shape[1:], name="ffe_conv_3")
        self.ffe_conv_3_batchnorm = layers.BatchNormalization()

        self.ffe_conv_4 = layers.Conv2D(filters=4, kernel_size=(1,1), activation='tanh', input_shape=input_shape[1:], name="ffe_conv_4")
        self.ffe_conv_4_batchnorm = layers.BatchNormalization()

        self.infoshare_concat = layers.Concatenate(axis=1)

        self.ffe_flatten = layers.Flatten()
        self.dense_1 = layers.Dense(units=1024, activation="tanh", name="dense1")
        self.dense_2 = layers.Dense(units=512, activation="tanh", name="dense2")
        self.dense_3 = layers.Dense(units=256, activation="tanh", name="dense3")
        self.reshape_1 = layers.Reshape((Globals.NETWORK_OUTPUT_SIZE, -1))
        self.dense_4 = layers.Dense(units=256, activation="tanh", name="dense4")
        #self.dense_4_2 = layers.Dense(units=64, activation="tanh", name="dense4_2")
        #self.ffe_dropout_dense = layers.Dropout(0.5)
        #self.dense_out = layers.Dense(units=1, activation="linear", name="p_out", kernel_constraint=MinMaxConstraint(-1.0, 1.0), bias_constraint=MinMaxConstraint(-1.0, 1.0))
        self.dense_out = layers.Dense(units=1, name="p_out", activation='tanh')
        #self.dense_out_2 = layers.Dense(units=1, activation=relu_clipped_1, name="p_out_2")
        self.reshape_2 = layers.Reshape((Globals.NETWORK_OUTPUT_SIZE,))
        #self.metric_1 = tf.keras.metrics.Mean(name='metric_1')

        self.last_output_rnn = EchoRNN(dims=Globals.NETWORK_OUTPUT_SIZE, batch_size=self.batch_size, name="last_output_rnn")
        self.last_output_raw_rnn = EchoRNN(dims=Globals.NETWORK_OUTPUT_SIZE, batch_size=self.batch_size, name="last_output_raw_rnn")
        self.last_fe_output_rnn = EchoRNN(dims=Globals.NETWORK_RECURRENT_SIZE, batch_size=self.batch_size, name="last_fe_output_rnn")
        self.last_rect_output_rnn = EchoRNN(dims=4, batch_size=self.batch_size, name="last_rect_output_rnn")

        self.last_output_rnn.build(input_shape=(self.batch_size,1,Globals.NETWORK_OUTPUT_SIZE))
        self.last_output_raw_rnn.build(input_shape=(self.batch_size,1,Globals.NETWORK_OUTPUT_SIZE))
        self.last_fe_output_rnn.build(input_shape=(self.batch_size,1,Globals.NETWORK_RECURRENT_SIZE))
        self.last_rect_output_rnn.build(input_shape=(self.batch_size,1,4))

        self.last_output_rnn.reset_states()
        self.last_output_raw_rnn.reset_states()
        self.last_fe_output_rnn.reset_states()
        self.last_rect_output_rnn.reset_states()

    def _make_train_function(self):
        super().make_train_function()

    def reset_states(self):
        super().reset_states()

        # We set all of the states to uniform randoms so that it generalizes the
        # velocity driving a bit better. Kinda just flings the box around
        rands = tf.random.uniform(shape=(self.batch_size, Globals.NETWORK_OUTPUT_SIZE), minval=-1.0, maxval=1.0, dtype=tf.float32)
        self.last_output_rnn.states[0].assign(rands)
        rands = tf.random.uniform(shape=(self.batch_size, Globals.NETWORK_OUTPUT_SIZE), minval=-1.0, maxval=1.0, dtype=tf.float32)
        self.last_output_raw_rnn.states[0].assign(rands)
        rands = tf.random.uniform(shape=(self.batch_size, 4), minval=-1.0, maxval=1.0, dtype=tf.float32)
        self.last_rect_output_rnn.states[0].assign(rands)
        #self.last_fe_output_rnn.states[0].assign(np.ones((Globals.TRAIN_BATCH_SIZE, Globals.NETWORK_RECURRENT_SIZE), dtype=np.float32) * (1.0 / float(random.randint(2, 5))))
        rands = tf.random.uniform(shape=(self.batch_size, Globals.NETWORK_RECURRENT_SIZE), minval=-1.0, maxval=1.0, dtype=tf.float32)
        self.last_fe_output_rnn.states[0].assign(rands)



    def call(self, inputs, training=False):

        if self.box_indices is None:
            if training:
                self.box_indices = range(0,self.batch_size)#tf.Variable(range(0,self.batch_size))
            else:
                self.box_indices = np.array([0.0,])

        inputs_shape = tf.shape(inputs)
        inputs_shape_fp32 = tf.cast(inputs_shape, tf.float32)
        x = self.input_layer(inputs, input_shape=inputs_shape)
        image_input = tf.cast(x, dtype=tf.float32)

        def salt_and_pepper(image, prob_salt=0.1, prob_pepper=0.1):
            random_values = tf.random.uniform(shape=Globals.TRAIN_INPUT_SHAPE)
            image = tf.where(random_values < prob_salt, 1., image)
            image = tf.where(1 - random_values < prob_pepper, 0., image)
            return image

        if training:
            image_input = salt_and_pepper(image_input)
        '''
        if training:
            noise = 0.3
            sigma = tf.abs(tf.random.normal([], 0, noise))
            noise = tf.random.normal(inputs_shape, 0, sigma)
            image_input += noise
            image_input = tf.reshape(image_input, (inputs_shape[0], 288,384,1))
        '''

        image_height = inputs_shape_fp32[1]
        image_width = inputs_shape_fp32[2]

        shift_left = [[(-0.5/image_height), (-0.5/image_width)]]
        shift_right = [[(0.5/image_height), (0.5/image_width)]]
        shift_left = tf.tile(shift_left, (inputs_shape[0],1))
        shift_right = tf.tile(shift_right, (inputs_shape[0],1))

        last_output = self.last_output_rnn.states[0]
        last_output_raw = self.last_output_rnn.states[0]
        last_fe_output = self.last_fe_output_rnn.states[0]
        last_rect = self.last_rect_output_rnn.states[0]

        inputs_shape_tiled = tf.tile(tf.expand_dims(inputs_shape_fp32, 0), (inputs_shape[0],1))
        
        # The feature extractor attempts to get the following information:
        # [patch_center_x, patch_center_y, patch_scale, patch_vel_x, patch_vel_y]
        fe_x = tf.cast(x, dtype=tf.float32)
        fe_x = self.fe_initial_downsample(fe_x)
        if training:
            gamma = tf.random.uniform([], 1.8, 1.9)
            fe_x = tf.keras.layers.Lambda(lambda fe_x: tf.map_fn(lambda img: tf.image.adjust_gamma(
                img, gamma=gamma, gain=1
            ), fe_x))(fe_x)
        else:
            gamma = 1.8
            fe_x = tf.keras.layers.Lambda(lambda fe_x: tf.map_fn(lambda img: tf.image.adjust_gamma(
                img, gamma=gamma, gain=1
            ), fe_x))(fe_x)

        fe_x = tf.keras.layers.Lambda(lambda fe_x: tf.keras.backend.map_fn(lambda img: tf.image.per_image_standardization(img), fe_x))(fe_x)
        fe_x = self.fe_conv_1(fe_x)
        fe_x = self.fe_conv_1_batchnorm(fe_x)
        fe_x = self.fe_pool_1(fe_x)
        
        fe_x = self.fe_conv_2(fe_x)
        fe_x = self.fe_conv_2_batchnorm(fe_x)
        fe_x = self.fe_pool_2(fe_x)

        fe_x = self.fe_conv_3(fe_x)
        fe_x = self.fe_conv_3_batchnorm(fe_x)
        fe_x = self.fe_pool_3(fe_x)

        fe_x = self.fe_conv_4(fe_x)
        fe_x = self.fe_conv_4_batchnorm(fe_x)

        fe_x = self.fe_conv_5(fe_x)
        fe_x = self.fe_conv_5_batchnorm(fe_x)

        #fe_x = self.fe_global_pool(fe_x)
        fe_x = self.fe_dense(fe_x)
        fe_x = self.fe_predense_flatten(fe_x)
        fe_x = self.fe_dense__2(fe_x)
        fe_x = self.fe_reshape_predense(fe_x)
        fe_x = self.fe_dense_2(fe_x)
        fe_x = self.fe_reshape_postdense(fe_x)

        # We want to give this area a bit more information to work with,
        # and some recurrent state
        #print (last_output, last_fe_output)
        #fe_x = tf.concat([fe_x, inputs_shape_tiled, shift_right, last_output, last_fe_output], -1)
        #fe_x = tf.concat([fe_x, inputs_shape_tiled, shift_right], -1)
        fe_x = tf.concat([fe_x, last_output, last_output_raw, last_fe_output, last_rect], -1)
        #fe_x = tf.concat([fe_x, last_fe_output], -1)
        fe_x = self.fe_flatten(fe_x)
        
        fe_x = self.fe_dense_3(fe_x)
        fe_x = self.fe_dense_4(fe_x) #- self.fe_dense_4_2(fe_x) this causes values to only be positive??
        fe_x = self.fe_dense_5(fe_x)
        fe_x = self.fe_dense_6(fe_x)
        fe_x = self.fe_dense_7(fe_x)
        #fe_x = tf.clip_by_value(self.fe_out(fe_x) - 2.0, -2.0, 2.0)
        fe_x = self.fe_out(fe_x)# * self.fe_out_2(fe_x)#tf.clip_by_value(self.fe_out(fe_x), 0.001, 1.0) - tf.clip_by_value(self.fe_out_2(fe_x), 0.001, 1.0)
        #fe_x = tf.clip_by_value(fe_x, 0.1, 1.0)
        #if type(fe_x) is tf.Tensor:
        #    self.fe_indirect_access.assign(tf.reshape(fe_x, (self.batch_size, Globals.NETWORK_RECURRENT_SIZE)))


        fe_x = tf.concat([fe_x[:, 0:3], fe_x[:, 3:5], fe_x[:, 5:]], -1)
        fe_x_cur = fe_x

        fe_x_yx = last_fe_output[:, 0:2] + fe_x[:, 3:5]
        fe_x_yx = tf.math.floormod(fe_x_yx, 1.0)
        #fe_x_yx = last_fe_output[:, 0:2] + last_fe_output[:, 3:5]
        fe_x_scale = tf.clip_by_value(last_fe_output[:, 2:3] + fe_x[:, 5:6], 0.5, 1.5)


        #fe_x = tf.reshape(fe_x, (-1, 5))
        #print (fe_x.shape)
        #print (fe_x[:, 2:3].shape)
        

        shift_left = tf.math.multiply(shift_left, image_width*0.3) #float(self.patch_size)
        shift_right = tf.math.multiply(shift_right, image_width*0.3)
        shift_left = tf.math.multiply(shift_left, fe_x_scale)
        shift_right = tf.math.multiply(shift_right, fe_x_scale)
        #shift_left = tf.math.multiply(shift_left, fe_x[:, 2:3])
        #shift_right = tf.math.multiply(shift_right, fe_x[:, 2:3])


        #shift_left = [1,2]
        #shift_right = [1,2]

        

        #fe_x1y1 = tfp.math.clip_by_value_preserve_gradient(fe_x[:, 0:2], 0.0, 1.0) + shift_left #+ fe_x[:, 3:5]
        #fe_x2y2 = tfp.math.clip_by_value_preserve_gradient(fe_x[:, 0:2], 0.0, 1.0) + shift_right #+ fe_x[:, 3:5]
        fe_y1x1 = fe_x_yx[:, 0:2] + shift_left #+ fe_x[:, 3:5]
        fe_y2x2 = fe_x_yx[:, 0:2] + shift_right #+ fe_x[:, 3:5]

        # Prevent the box from running off the sides of the image
        
        '''
        fe_y2x2 -= tf.minimum(fe_y1x1, 0.0)
        fe_y1x1 -= tf.minimum(fe_y1x1, 0.0)
        
        fe_y1x1 -= (tf.maximum(fe_y2x2, 1.0) - 1.0)
        fe_y2x2 -= (tf.maximum(fe_y2x2, 1.0) - 1.0)
        '''
        
        fe_x_final = self.fe_final_concat([fe_y1x1, fe_y2x2])

        #fe_x_yx = (fe_y2x2 + fe_y1x1) * 0.5
        fe_x = self.fe_x_concat([tf.clip_by_value(fe_x_yx, 0.0, 1.0), fe_x_scale, fe_x[:, 3:4], fe_x[:, 4:5], fe_x[:, 5:6], fe_x[:, 6:Globals.NETWORK_RECURRENT_SIZE]])
        fe_x_rnnin = tf.reshape(fe_x, (inputs_shape[0], 1, Globals.NETWORK_RECURRENT_SIZE))
        fe_x_ = self.last_fe_output_rnn(fe_x_rnnin)
        #print (self.boxes.shape)

        crop_size = (self.patch_size, self.patch_size)

        #self.add_metric(x, name='crop')

        #boxes =

        fe_x_final_rnnin = tf.reshape(fe_x_final, (inputs_shape[0], 1, 4))
        fe_x_final_ = self.last_rect_output_rnn(fe_x_final_rnnin)

        x = tf.image.crop_and_resize(
            image_input,
            fe_x_final,
            self.box_indices,
            crop_size,
            method='bilinear',
            name="image_crop_tfop"
        )

        x = tf.cast(x, dtype=tf.float32)

        if training:
            gamma = tf.random.uniform([], 1.8, 1.9)
            x = tf.keras.layers.Lambda(lambda x: tf.keras.backend.map_fn(lambda img: tf.image.adjust_gamma(
                img, gamma=gamma, gain=1
            ), x))(x)
        else:
            gamma = 1.8
            x = tf.keras.layers.Lambda(lambda x: tf.keras.backend.map_fn(lambda img: tf.image.adjust_gamma(
                img, gamma=gamma, gain=1
            ), x))(x)

        x = tf.keras.layers.Lambda(lambda x: tf.keras.backend.map_fn(lambda img: tf.image.per_image_standardization(img), x))(x)
        

        x = self.image_crop_reshape(x)
        x = self.image_crop(x)
        #self.metric_1(x)
        #tf.print(inputs)
        #tf.print(x)

        #self.add_metric(tf.reduce_sum(x), name='metric_2')
        #self.add_metric(self.metric_1(x))

        x = self.ffe_conv_1(x)
        x = self.ffe_conv_1_batchnorm(x)
        #x = self.ffe_conv_1_relu(x)
        x = self.ffe_pool_1(x)
        
        x = self.ffe_conv_2(x)
        x = self.ffe_conv_2_batchnorm(x)
        #x = self.ffe_conv_2_relu(x)
        x = self.ffe_pool_2(x)

        x = self.ffe_conv_3(x)
        x = self.ffe_conv_3_batchnorm(x)
        #x = self.ffe_conv_3_relu(x)
        #x = self.ffe_pool_3(x)

        x = self.ffe_conv_4(x)
        x = self.ffe_conv_4_batchnorm(x)
        #x = self.ffe_conv_4_relu(x)
        #x = self.ffe_pool_4(x)

        x = self.ffe_flatten(x)
        x = self.dense_1(x)
        x = self.dense_2(x)
        x = self.dense_3(x)
        x = self.infoshare_concat([x, inputs_shape_tiled, fe_x, last_output, last_output_raw, last_fe_output, last_rect, tf.zeros((inputs_shape[0], 16), dtype=tf.float32)])
        x = self.reshape_1(x)
        x = self.dense_4(x)# + (self.dense_4_2(x) * 0.0)
        x = self.dense_out(x)# - self.dense_out_2(x)
        outputs = self.reshape_2(x)

        fe_x1y1 = tf.concat([fe_y1x1[:, 1:2], fe_y1x1[:, 0:1]], -1)
        fe_x2y2 = tf.concat([fe_y2x2[:, 1:2], fe_y2x2[:, 0:1]], -1)

        box_center = ((fe_x2y2 + fe_x1y1) * 0.5)

        outputs = outputs * 0.5#(1.1 * outputs) - (1.1 * 0.5)

        outputs_raw_rnnin = tf.reshape(outputs, (inputs_shape[0], 1, Globals.NETWORK_OUTPUT_SIZE))
        outputs_raw_ = self.last_output_raw_rnn(outputs_raw_rnnin)
        #tf.print(outputs)
        
        #outputs = tf.reshape(outputs, (-1, Globals.NETWORK_OUTPUT_SIZE))
        #tf.print(outputs)
        #tf.print(state)

        outputs_x = outputs[:, 0:1]
        outputs_y = outputs[:, 1:2]

        outputs_x *= (fe_y2x2[:, 1:2] - fe_y1x1[:, 1:2])
        outputs_y *= (fe_y2x2[:, 0:1] - fe_y1x1[:, 0:1])

        outputs_x += box_center[:, 0:1]#fe_y1x1[:, 1:2]
        outputs_y += box_center[:, 1:2]#fe_y1x1[:, 0:1]

        last_outputs_x = last_output[:, 0:1]
        last_outputs_y = last_output[:, 1:2]
        
        # keep points within box
        #outputs_x = tf.clip_by_value(outputs_x, fe_y1x1[:, 1:2], fe_y2x2[:, 1:2])
        #outputs_y = tf.clip_by_value(outputs_y, fe_y1x1[:, 0:1], fe_y2x2[:, 0:1])
        

        #outputs_x += fe_x[:, 4:5]
        #outputs_y += fe_x[:, 3:4]

        last_outputs_x_vel = last_output[:, 2:3]
        last_outputs_y_vel = last_output[:, 3:4]
        outputs_x_vel = (outputs_x - fe_y1x1[:, 1:2]) - (last_outputs_x - last_rect[:, 1:2]) + fe_x[:, 4:5]
        outputs_y_vel = (outputs_y - fe_y1x1[:, 0:1]) - (last_outputs_y - last_rect[:, 0:1]) + fe_x[:, 3:4]

        '''
        outputs_x_vel_fixed = tf.where(tf.greater(last_outputs_x_vel, 0.0), 
                                 tf.where(tf.less(outputs_x_vel, 0.0), tf.math.floormod(outputs_x_vel, 1.0), outputs_x_vel),
                                 tf.where(tf.greater(outputs_x_vel, 0.0), tf.math.floormod(outputs_x_vel, -1.0), outputs_x_vel))
        outputs_y_vel_fixed = tf.where(tf.greater(last_outputs_y_vel, 0.0), 
                                 tf.where(tf.less(outputs_y_vel, 0.0), tf.math.floormod(outputs_y_vel, 1.0), outputs_y_vel),
                                 tf.where(tf.greater(outputs_y_vel, 0.0), tf.math.floormod(outputs_y_vel, -1.0), outputs_y_vel))

        outputs_x_vel = tf.where(tf.greater(outputs_x_vel, 0.9 - last_outputs_x_vel),
                                 outputs_x_vel_fixed,
                                 tf.where(tf.less(outputs_x_vel, -0.9 + last_outputs_x_vel),
                                          outputs_x_vel_fixed,
                                          outputs_x_vel)
                                 )
        outputs_y_vel = tf.where(tf.greater(outputs_y_vel, 0.9 - last_outputs_y_vel),
                                 outputs_y_vel_fixed,
                                 tf.where(tf.less(outputs_y_vel, -0.9 + last_outputs_y_vel),
                                          outputs_y_vel_fixed,
                                          outputs_y_vel)
                                 )
        '''

        outputs = tf.concat([outputs_x, outputs_y, outputs_x_vel, outputs_y_vel], -1)
        outputs_flipped = tf.concat([outputs_y, outputs_x], -1)
        outputs_rnnin = tf.reshape(outputs, (inputs_shape[0], 1, Globals.NETWORK_OUTPUT_SIZE))
        outputs_ = self.last_output_rnn(outputs_rnnin)
       

        if type(outputs) is tf.Tensor and type(box_center) is tf.Tensor:
            self.fe_x_cur_tensor = tf.concat([fe_x_cur[:, 4:5], fe_x_cur[:, 3:4]], -1)
            self.box_center_tensor = box_center
            self.outputs_tensor = outputs
            self.scale_tensor = fe_x_scale

            #self.outputs_indirect_access.assign(tf.reshape(outputs_flipped, (self.batch_size, Globals.NETWORK_OUTPUT_SIZE)))
            #self.box_indirect_access.assign(tf.reshape(box_center, (self.batch_size, 2)))
            
        #return (outputs + ((fe_x1y1 + fe_x2y2) * 0.5)) * 0.5
        #return ((fe_x2y2 + fe_x1y1) * 0.5) + (outputs * 0.0)
        return outputs


def build_model(batch_size):

    # Model / data parameters
    input_shape = (batch_size, Globals.TRAIN_INPUT_SHAPE[1], Globals.TRAIN_INPUT_SHAPE[2], Globals.TRAIN_INPUT_SHAPE[3])
    
    #input_shape = (batch_size, 256,256,1)
    output_shape = (Globals.NETWORK_OUTPUT_SIZE,)

    #print ("Input shape:", input_shape)
    #print ("Output shape:", output_shape)

    metric_tensors = []
    inputs = keras.Input(shape=input_shape)

    model_out = EyeNet(input_shape=input_shape, batch_size=batch_size)

    test_inputs = keras.Input(batch_input_shape=input_shape) #tf.zeros(input_shape)#
    model_out.call(test_inputs, training=True)
    model_out.build(input_shape=input_shape)
    
    #model_out.summary()

    return model_out