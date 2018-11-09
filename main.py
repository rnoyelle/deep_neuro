 
 
import sys
import os
import os.path

import lib.matnpy.matnpyio as io
import lib.cnn.cnn as cnn 
import lib.matnpy.matnpy as matnpy
import lib.cnn.helpers as hlp

import tensorflow as tf
import numpy as np
from math import ceil

import random
#from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

import pandas as pd
import datetime


################################################
#                    PARAMS                    #
################################################

#################
### base path ###
#################

base_path =  # .../my_project/
raw_path = base_path + 'data/raw/' # .../my_project/data/raw/

###################
### run params ###
###################

param_index = int(sys.argv[1])
file = base_path + 'scripts/_params/training.txt'
## Get current params from file
with open(file, 'rb') as f:
    params = np.loadtxt(f, dtype='object', delimiter='\n')
params = params.tolist()
curr_params = eval(params[param_index-1])
sess_no = curr_params[0]
decode_for = curr_params[1]
areas = curr_params[2]
align_on, from_time, to_time = curr_params[3], curr_params[4], curr_params[5]
lowcut, highcut, order = curr_params[6], curr_params[7], curr_params[8]
cortex = curr_params[9]
str_to_print = curr_params[10]

#sess_no = '150228'
#decode_for = 'stim'
#only_correct_trials = True

#align_on, from_time, to_time = 'sample', 0, 500 
#lowcut, highcut, order = 80, 300, 3

##cortex = 'Visual' # 
#areas = ['V1']

if decode_for =='resp':
    only_correct_trials = False
else:
    only_correct_trials = True
renorm = True # if True,  Standardize features by removing the mean and scaling to unit variance
elec_type = 'grid' # One of 'single' (use all electrodes within area as single trials), 
                   #'grid' (use whole electrode grid), 
                   # 'average' (mean over all electrodes in area)


# path
raw_path = raw_path + sess_no + '/session01/' ## .../data/raw/sess_no/session01/ 
rinfo_path = raw_path + 'recording_info.mat' ## .../data/raw/sess_no/session01/file.mat 

##################################################
#                 CNN PARAMS                     #
##################################################


if decode_for == 'stim':

    # hyper params
    n_iterations = 100
    size_of_batches = 50
    dist = 'random_normal'
    batch_norm = 'renorm'  # 'after'
    nonlin = 'elu'
    normalized_weights = True
    learning_rate = 1e-5
    l2_regularization_penalty = 5
    keep_prob_train = .5
    
    # layer dimensions
    n_layers = 7
    patch_dim = [1, 5]  # 1xN patch
    pool_dim = [1, 2]
    in1, out1 = 1, 3
    in2, out2 = 3, 6
    in3, out3 = 6, 12
    in4, out4 = 12, 36
    in5, out5 = 36, 72
    in6, out6 = 72, 256
    in7, out7 = 256, 500
    in8, out8 = 500, 1000
    fc_units = 200

else:
    
    # hyper params
    n_iterations = 100
    size_of_batches = 50
    dist = 'random_normal'
    batch_norm = 'after'
    nonlin = 'elu'
    normalized_weights = True
    learning_rate = 1e-5
    l2_regularization_penalty = 20
    keep_prob_train = .1
    
    # layer dimensions
    n_layers = 6
    patch_dim = [1, 5]  # 1xN patch
    pool_dim = [1, 2]
    in1, out1 = 1, 3
    in2, out2 = 3, 6
    in3, out3 = 6, 12
    in4, out4 = 12, 36
    in5, out5 = 36, 72
    in6, out6 = 72, 256
    in7, out7 = 256, 500
    in8, out8 = 500, 1000
    fc_units = 200
    
    
channels_in = [in1, in2, in3, in4, in5, in6, in7, in8][:n_layers]
channels_out = [out1, out2, out3, out4, out5, out6, out7, out8][:n_layers]


# TRAIN/TEST params

#train_size = .8
#test_size = .2
n_splits = 5 # K-fold
seed = np.random.randint(1,10000)

# Auto-define number of classes
classes = 2 if decode_for == 'resp' else 5

##################################################
#                    GET DATA                    #
##################################################

data = matnpy.get_subset_by_areas(sess_no, raw_path,
                         align_on, from_time, to_time, 
                         lowcut, highcut, 
                         areas,
                         epsillon = 100, order= order,
                         only_correct_trials = only_correct_trials, renorm = renorm, elec_type = elec_type)

n_chans = data.shape[1]
samples_per_trial = data.shape[2]

targets = io.get_targets(decode_for, raw_path, n_chans, elec_type=elec_type,
                         only_correct_trials=only_correct_trials,
                         onehot=True)


##################################################
#                   CREATE CNN                   #
##################################################


##########
# LAYERS #
##########

# placeholders
x_ = tf.placeholder(tf.float32, shape=[
        None, n_chans, samples_per_trial
        ])
y_ = tf.placeholder(tf.float32, shape=[None, classes])
training = tf.placeholder_with_default(True, shape=())
keep_prob = tf.placeholder(tf.float32)

# Network
out, weights = cnn.create_network(
        n_layers=n_layers, 
        x_in=x_,
        n_in=channels_in, 
        n_out=channels_out, 
        patch_dim=patch_dim,
        pool_dim=pool_dim,
        training=training, 
        n_chans=n_chans,
        n_samples=samples_per_trial,
        weights_dist=dist, 
        normalized_weights=normalized_weights,
        nonlin=nonlin,
        bn=True)


###################
# FULLY-CONNECTED #
###################

# Fully-connected layer (BN)
fc1, weights[n_layers] = cnn.fully_connected(out, 
            bn=True, 
            units=fc_units,
            training=training,
            nonlin=nonlin,
            weights_dist=dist,
            normalized_weights=normalized_weights)


###################
# DROPOUT/READOUT #
###################

# Dropout (BN)
fc1_drop = tf.nn.dropout(fc1, keep_prob)

# Readout
weights[n_layers+1] = cnn.init_weights([fc_units, classes])
y_conv = tf.matmul(fc1_drop, weights[n_layers+1])
weights_shape = [tf.shape(el) for el in weights.values()]


#############
# OPTIMIZER #
#############

# Loss
loss = cnn.l2_loss(weights, 
                   l2_regularization_penalty, 
                   y_, 
                   y_conv, 
                   'loss')

# Optimizer

train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# ACCURACY

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


#recall_macro = tf.metrics.mean_per_class_accuracy(
        #tf.argmax(y_, 1),
        #tf.argmax(y_conv, 1),
        #classes)
     


    
##################################################
#             TRAIN AND TEST                     #
##################################################
print(str_to_print)

kf = StratifiedKFold(n_splits=n_splits,shuffle=True, random_state=seed)

recall_macro_train_per_fold = []
recall_macro_test_per_fold = []
error_bar_per_fold = []
y_true_per_fold = []
y_pred_per_fold = []


for idx_train, idx_test in kf.split(data, np.argmax(targets[:,:], axis=1)):

    # SPLIT TRAIN AND TEST
    train = data[idx_train]
    test = data[idx_test]
    
    train_labels = targets[idx_train]
    test_labels = targets[idx_test]
    

    ############
    # TRAINING #
    ############
    

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Number of batches to train on
        for i in range(n_iterations):
            ind_train = hlp.subset_train(train_labels, classes, size_of_batches)

            # Every n iterations, print training accuracy
            if i % 10 == 0:
                train_accuracy = accuracy.eval(feed_dict={
                        x_: train[ind_train,:,:],
                        y_: train_labels[ind_train,:],
                        keep_prob: 1.0
                        })
                print('step %d, training accuracy: %g' % (
                        i, train_accuracy))

            # Training
            curr_x = train[ind_train,:,:]
            curr_y = train_labels[ind_train,:]
            train_step.run(feed_dict={
                    x_: curr_x,
                    y_: curr_y,
                    keep_prob: keep_prob_train
                    })
            
        ############
        # RESULTS  #
        ############
            
        # result on train test
        train_predict = y_conv.eval(feed_dict={
                        x_: train,
                        y_: train_labels,
                        keep_prob: 1.0
                })        
        
        y_true = np.argmax(train_labels, axis = 1)
        y_pred = np.argmax(train_predict, axis = 1)
        
        recall_macro_train, _ = cnn.recall_macro(y_true, y_pred)
        
        print('trainning accuracy: %g' %(recall_macro_train))
        
        recall_macro_train_per_fold.append(recall_macro_train)


        # result on the base test
        curr_x_test = test[:,:,:]
        curr_y_test = test_labels[:,:]
        

        curr_y_predict = y_conv.eval(feed_dict={
                        x_: curr_x_test,
                        y_: curr_y_test,
                        keep_prob: 1.0
                })
        
        
        y_true = np.argmax(curr_y_test, axis = 1)
        y_pred = np.argmax(curr_y_predict, axis = 1)
        
        y_true_per_fold.append( np.argmax(curr_y_test, axis = 1).tolist() )
        y_pred_per_fold.append( np.argmax( curr_y_predict, axis = 1).tolist() )
        
        recall_macro_test, error_bar = cnn.recall_macro(y_true, y_pred)
        
        print('test accuracy : %g' %(recall_macro_test))
        recall_macro_test_per_fold.append(recall_macro_test)
        error_bar_per_fold.append(error_bar)
        
        
        
        
recall_macro = np.mean(recall_macro_test_per_fold)
error_bar_th = np.sqrt(np.sum(np.array(error_bar_per_fold)**2))/n_splits
error_bar_emp = np.std(recall_macro_test_per_fold)/np.sqrt(n_splits)
        
        
        
    
#################################################
#           SAVE RESULT                         #
#################################################

    
    

time = str(datetime.datetime.now())
result = [str(sess_no), decode_for, only_correct_trials, 
          str(areas), cortex, elec_type,
          'low'+str(lowcut)+'high'+str(highcut)+'order'+str(order), 
          'align_on'+align_on+'from_time'+str(from_time)+'to_time'+str(to_time),
          recall_macro, error_bar_th, error_bar_emp, np.sum(targets, axis=0),
          str(recall_macro_train_per_fold), str(recall_macro_test_per_fold), 
          seed, n_splits, 
          data.shape[0], n_chans, samples_per_trial,
          str(y_true_per_fold),
          str(y_pred_per_fold),
          renorm,
          n_layers,
          str(patch_dim), str(pool_dim),
          str(channels_in),str(channels_out),
          nonlin,
          fc_units,
          n_iterations, size_of_batches, learning_rate,
          dist, normalized_weights,
          batch_norm,
          keep_prob_train, 
          l2_regularization_penalty,
          time]

df = pd.DataFrame([result],
                  columns=['session', 'decode_for', 'only_correct_trials',
                           'areas', 'cortex', 'elec_type',
                           'frequency_band',
                           'interval',
                           'recall_macro', 'error_bar_th', 'error_bar_emp', 'n_test_per_class',
                           'recall_macro_train_per_fold', 'recall_macro_test_per_fold',
                           'seed', 'n_splits',
                           'data_size', 'n_chans', 'window_size',
                           'y_true_per_fold', 'y_pred_per_fold',
                           'renorm',
                           'n_layers','patch_dim', 'pool_dim',
                           'channels_in', 'channels_out',
                           'nonlin','fc_units',
                           'n_iterations', 'size_of_batches', 'learning_rate',
                           'weights_dist', 'normalized_weights',
                           'batch_norm',
                           'keep_prob_train',
                           'l2_regularization_penalty',
                           'time'],
                  index=[0])
         
          
# Save to file
file_name = (base_path + 'results/training/'
          + sess_no + '_training_'+decode_for+'.csv')
file_exists = os.path.isfile(file_name)
if file_exists :
    with open(file_name, 'a') as f:
        df.to_csv(f, mode ='a', index=False, header=False)
else:
    with open(file_name, 'w') as f:
        df.to_csv(f, mode ='w', index=False, header=True)










