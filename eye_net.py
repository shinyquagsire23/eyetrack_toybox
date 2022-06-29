#decord (from source Jun 24 2022) https://github.com/dmlc/decord
# brew install cmake ffmpeg@4

#opencv-python-4.6.0.66
#tensorflow_probability
#tensorflow-macos
#tensorflow-metal (has to run in venv?)
from globals import *
import cv2
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.constraints import Constraint
import tensorflow_probability as tfp
from tensorflow.python.framework.ops import disable_eager_execution

from evaluate import run_network
from model import build_model
from data_serve import data_serve_init, read_landmarks, read_iris_eli, CustomDataGen

import sys

import math
import pickle
import datetime

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0],True)

#from hanging_threads import start_monitoring
#start_monitoring(seconds_frozen=10, test_interval=1000)

class TakingTooLongException(Exception):

    def __init__(self):
        super().__init__("Taking too long")

    def __str__(self):
        return "Taking too long"

def train_model(epochs, resume=False):
    
    model = build_model(Globals.TRAIN_BATCH_SIZE)
    

    model.train_completed = False
    model._epochs_trained = 0

    model._timestamp_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    if resume:
        with open('eyetrack_net_epochs.pkl', 'rb') as f:
            model._epochs_trained = pickle.load(f)
        with open('eyetrack_net_timestampstr.txt', 'r') as f:
            model._timestamp_str = f.read()
        #tf.summary.experimental.set_step(model._train_epoch_num_tensor)
        if model._epochs_trained > 90:
            Globals.TRAIN_MAX_SEQS = 300 + (100 * ((model._epochs_trained - 90) // 30))
        
        if Globals.TRAIN_MAX_SEQS > 500:
            Globals.TRAIN_MAX_SEQS = 500

    if model._epochs_trained == 0:
        model.summary()

    #model._train_epoch_num_tensor.assign()
    #tf.summary.experimental.set_step(model._epochs_trained)

    def train_save_step(batch, logs):
        nonlocal model

        model._epochs_trained += 1
        model.save_weights(filepath="eyetrack_net.h5")
        if model._epochs_trained % 30 == 0:
            model.save_weights(filepath="eyetrack_net_" + str(model._epochs_trained) + ".h5")
        #model.save("eyetrack_train.h5")

        symbolic_weights = getattr(model.optimizer, 'weights')
        weight_values = K.batch_get_value(symbolic_weights)
        with open('eyetrack_net_optimizer.pkl', 'wb') as f:
            pickle.dump(weight_values, f)
        with open('eyetrack_net_epochs.pkl', 'wb') as f:
            pickle.dump(model._epochs_trained, f)
        with open('eyetrack_net_timestampstr.txt', 'w') as f:
            f.write(model._timestamp_str)

        model.reset_states()
        print ("State saved.")

        #print (model._tensorboard_callback._train_step)
        
        model._tensorboard_callback._train_writer.flush()
        
        #model._train_epoch_num_tensor += 1

    def custom_dist_loss(y_true, y_pred):
        nonlocal model

        y_true_pos = y_true[:, 0:2]
        y_true_vel = y_true[:, 2:4]
        y_pred_pos = y_pred[:, 0:2]
        y_pred_vel = y_pred[:, 2:4]

        def velocity_loss(box_pos, pos, vel):
            
            '''
            direction = tf.multiply(vel, pos-box_pos )
            direction_bad = tf.minimum(direction, 0.0)
            direction_bad_mag = tf.math.square(tf.norm(direction_bad, axis=1) * 10.0)
            
            loss = tf.reduce_mean(direction_bad_mag) * 1000.0  # (batch_size, 2)

            #dont_stand_still = tf.math.reduce_mean(tf.where(tf.less(tf.abs(vel), 0.00001), [0.0], [1.0])) * 10000.0

            dont_stand_still = tf.math.reduce_mean(vel + tf.where(tf.equal(vel, 0.0), [1000.0], [0.0]))#tf.math.reduce_mean(tf.where(tf.less(vel*vel, 0.001), [1.0], [0.0])) * 1000.0

            
            #print (loss)

            tf.print("asf", loss, dont_stand_still, vel, pos, box_pos)
            tf.print(model.last_output_rnn.states[0])
            tf.print(model.last_fe_output_rnn.states[0][:, 3:5])

            loss += dont_stand_still
            
            loss = loss / 2000.0
            '''

            
            #loss = tf.math.reduce_sum(tf.math.reduce_euclidean_norm([(vel*10.0), (pos*10.0)-(box_pos*10.0)], [1, 2]))
            #loss = loss*loss

            '''
            # slightly counter-intuitive:
            # with (box_pos*10.0)+(vel*10.0)-(pos*10.0),
            # if the box gets locked in a corner, it sets vel to the box pos
            difference = (box_pos*10.0)+(vel*10.0)-(pos*10.0)
            loss = tf.math.reduce_sum(difference*difference)
            loss = loss*loss
            '''

            direction = tf.multiply(vel, pos-box_pos )
            direction_bad = tf.minimum(direction, 0.0)
            direction_bad_mag = tf.math.square(tf.norm(direction_bad, axis=1) * 10.0)
            
            loss = tf.reduce_sum(direction_bad_mag) * 100.0

            #loss = loss / 2000.0
            

            #tf.print(" vel_loss", loss)
            #tf.print(vel, pos, box_pos)
            #tf.print(model.last_output_rnn.states[0])
            #tf.print(model.last_fe_output_rnn.states[0][:, 3:5])
            #tf.print(model.last_fe_output_rnn.states[0][0, 0:2], model.last_fe_output_rnn.states[0][0, 3:5])


            return loss

        # reduce the box velocity as much as possible, fast movements are kept to the ffe
        vel_loss = velocity_loss(model.box_center_tensor, y_true_pos, model.fe_x_cur_tensor)
        
        #vel_loss_2 = velocity_loss(y_pred_pos, y_true_pos, y_true_vel) # model.fe_x_cur_tensor
        #vel_loss_3 = tf.math.reduce_sum(tf.math.square(tf.norm(y_true_vel-model.fe_x_cur_tensor, axis=1)*5.0))
        
        # ffe should predict the eye as much as possible, though the ffe vel is tied to the box vel directly
        vel_loss_4 = tf.math.reduce_sum(tf.math.square(tf.norm(y_true_vel-y_pred_vel, axis=1)*100.0))

        difference = (model.box_center_tensor*10.0)+((model.fe_x_cur_tensor*10.0) - (y_true_vel * 10.0))-(y_true_pos*10.0) # 
        slow_down_loss = tf.math.reduce_sum(difference*difference)
        slow_down_loss = slow_down_loss*slow_down_loss


        # try and keep the ffe dot in the center of the box as much as possible
        # the weights on this one are a bit lower because we want it to refine later in the train
        dot_dist = tf.norm(model.box_center_tensor-y_pred_pos, axis=1)
        pt_is_too_far_from_box_center = tf.math.reduce_sum(tf.where(tf.greater(dot_dist, 0.15 * model.scale_tensor), 
                                      dot_dist * 1000.0,
                                      dot_dist * 0.001))
        #tf.print("too far", model.box_center_tensor, y_pred_pos, tf.norm(model.box_center_tensor-y_pred_pos, axis=1), pt_is_too_far_from_box_center)

        y_dist = tf.expand_dims(tf.norm(y_true_pos - y_pred_pos, axis=1), axis=1) * 100.0

        # calculating squared difference between target and predicted values 
        loss = y_dist#*y_dist  # (batch_size, 2)
        #print (y_dist, loss)

        #loss += is_far_away
          
        # summing both loss values along batch dimension 
        loss = tf.math.reduce_sum(loss*loss)        # (batch_size,)

        #tf.print(loss)

        #print (loss)

        #tf.print("idk",loss, vel_loss, vel_loss_2)

        model.loss_loss = loss
        model.loss_vel_loss = vel_loss
        model.loss_vel_loss_4 = vel_loss_4
        model.loss_pt_is_too_far_from_box_center = pt_is_too_far_from_box_center
        model.loss_slow_down_loss = slow_down_loss

        tf.summary.scalar('loss_position', data=model.loss_loss)
        tf.summary.scalar('loss_vel_loss', data=model.loss_vel_loss)
        tf.summary.scalar('loss_ffe_eyevel', data=model.loss_vel_loss_4)
        tf.summary.scalar('loss_pt_is_too_far_from_box_center', data=model.loss_pt_is_too_far_from_box_center)
        tf.summary.scalar('loss_slow_down_loss', data=model.loss_slow_down_loss)

        '''
        tf.print("\nlosses:",loss, vel_loss, vel_loss_4, pt_is_too_far_from_box_center)
        tf.print("vel", model.last_fe_output_rnn.states[0][0, 3:5])
        
        tf.print(y_pred)
        tf.print(y_true)
        tf.print(model.box_center_tensor)
        tf.print(model.fe_x_cur_tensor)
        tf.print(model.last_fe_output_rnn.states[0][0, 0:2], model.last_fe_output_rnn.states[0][0, 3:5])
        '''

        # epoch 60:
        # loss 4817.5752 
        # vel_loss 7499.12256
        # pt_is_too_far_from_box_center 138.283066 
        # 694.791138
        out = (loss * 0.5) + vel_loss_4

        # once the model chills out a bit, we can remove the velocity dampening and let it go nuts
        #if model._epochs_trained < 60:
        out += vel_loss

        #if model._epochs_trained < 120:
        out += pt_is_too_far_from_box_center

        out += slow_down_loss

        return out #vel_loss_2, vel_loss_3 
    
    #plot_stuff(model)

    epochs_safe = 20
    '''
    if model._epochs_trained == 0:
        opt = tf.keras.optimizers.Adam(learning_rate=1e-5)
        epochs_safe = 3
    else:
    '''

    #opt = tf.keras.optimizers.Adam(learning_rate=1e-5)
    opt = tf.keras.optimizers.Adagrad(learning_rate=0.0001, initial_accumulator_value=0.1, epsilon=1e-07)
    #opt = tf.keras.optimizers.SGD(learning_rate=1e-3, decay=1e-6, momentum=0.9) #1e-2 
    #loss = tf.keras.losses.MeanSquaredError(reduction='sum_over_batch_size')
    tf.summary.experimental.set_step(opt.iterations)
    model.compile(loss=custom_dist_loss, optimizer=opt, metrics=[tf.keras.metrics.MeanSquaredLogarithmicError()])

    introspect = LambdaCallback(on_epoch_end=train_save_step) #lambda batch, logs: print(model.layers[3].get_weights())
    
    #plotter = LambdaCallback(on_epoch_end=lambda batch, logs: plot_stuff(model))

    model._tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs/fit/" + model._timestamp_str)
    model._tensorboard_callback.set_model(model)
    

    callbacks = []
    callbacks += [introspect]
    callbacks += [model._tensorboard_callback] #histogram_freq=1
    #callbacks += [plotter]

    traingen = CustomDataGen(batch_size=Globals.TRAIN_BATCH_SIZE)
    #valgen = CustomDataGen(batch_size=1, validation=True) #sel_idx_shift=1, 

    if resume:
        # hack: get the optimizer to populate weights
        grad_vars = model.trainable_weights
        zero_grads = [tf.zeros_like(w) for w in grad_vars]
        model.optimizer.apply_gradients(zip(zero_grads, grad_vars))

        with open('eyetrack_net_optimizer.pkl', 'rb') as f:
            weight_values = pickle.load(f)
        
        try:
            model.optimizer.set_weights(weight_values)
        except Exception as e:
            print("Exception during setting optimizer weights?")
            print (e)

        model.load_weights('eyetrack_net.h5', by_name=True, skip_mismatch=True)

    try:
        model.reset_states()
        with model._tensorboard_callback._train_writer.as_default():
            # validation_data=valgen, validation_batch_size=1, 
            model.fit(traingen, batch_size=Globals.TRAIN_BATCH_SIZE, initial_epoch=model._epochs_trained, epochs=model._epochs_trained+(epochs_safe - model._epochs_trained % epochs_safe), callbacks=callbacks, shuffle=False) #, callbacks=[model_checkpoint_callback]
        
        train_save_step(None, None)
        #print (model._epochs_trained)
        if model._epochs_trained >= epochs:
            model.train_completed = True
        else:
            print("Next batch of epochs")
            sys.exit(69)
    except TakingTooLongException:
        print ("Weird stall?")
        sys.exit(69)
        #return model.train_completed, [0,0]
    except Exception as e:
        print("Exception during training, halting")
        print (e)
        sys.exit(69)
        return True, [0,0]
        #sys.exit(-1)
    except KeyboardInterrupt:
        print ("Cancelled early")
        exit(-1)

    print ("Done training 1")

    valgen = CustomDataGen(batch_size=Globals.TRAIN_BATCH_SIZE, validation=True)

    score = [0,0]
    try:
        score = model.evaluate(valgen, batch_size=Globals.TRAIN_BATCH_SIZE, verbose=0)
        print("Test loss:", score[0])
        print("Test accuracy:", score[1])
    except Exception as e:
        print("Exception during eval")
        print (e)

    f = open("score.txt", "w")
    f.write(str(score))
    f.close()

    return model.train_completed, score

#
# Main
#

data_serve_init()

if sys.argv[1].lower() == "train":
    resume = False
    while True:
        train_completed, score = train_model(Globals.TRAIN_NUM_EPOCHS, resume)
        if not train_completed:
            print("Train aborted? Resuming...")
            resume = True
        else:
            break

    print ("Done!!")
    #run_network()
elif sys.argv[1].lower() == "resume":
    resume = True
    while True:
        train_completed, score = train_model(Globals.TRAIN_NUM_EPOCHS, resume)
        if not train_completed:
            print("Train aborted? Resuming...")
            resume = True
        else:
            break
    print ("Done!!")
    #run_network()
elif sys.argv[1].lower() == "run":
    with tf.device('/CPU:0'):
        run_network()
