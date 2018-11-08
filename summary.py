import sys
import os
import os.path

import pandas as pd
import numpy as np
import re


# PATH

base_path = # .../my_project/
#raw_path = '/media/rudy/disk2/lucy/' # .../my_project/data/raw/
    

## DECODERS
decoders = ['stim', 'resp'] # ['stim']

for decode_for in decoders :
    
    ## SESSION 
    # this way :
    #session = os.listdir(raw_path)
    #session.remove('unique_recordings.mat')
    # or this way
    available_file = os.listdir(base_path + 'results/training/')
    session = []
    for i in range(len(available_file)):
        txt = re.split('_|\.', available_file[i]) # = [sess_no, 'training', decode_for', 'csv']
        if txt[2] == decode_for :
            session.append(txt[0])
            
    
    ## merge files
    df_session = len(session) * [0]
    
    for count, sess_no in enumerate(session):
        df_session[count] = pd.read_csv(base_path + 'results/training/' + sess_no + '_training_'+decode_for+'.csv')
        
    if len(session) > 1 :
        result = pd.concat(df_session, ignore_index=True)
    elif len(session) == 1 :
        result = df_session[0]
    else:
        print('No file for decode_for = ', decode_for)
        break
    
    
    # 
    #result = result
                               #'renorm',
                           #'n_layers','patch_dim', 'pool_dim',
                           #'channels_in', 'channels_out',
                           #'nonlin','fc_units',
                           #'n_iterations', 'size_of_batches', 'learning_rate',
                           #'weights_dist', 'normalized_weights',
                           #'batch_norm',
                           #'keep_prob_train',
                           #'l2_regularization_penalty',
                           #'time']
     
    # columns to keep
    result = result[ ['session', 'decode_for', 'only_correct_trials',
                           'areas', 'cortex', 'elec_type',
                           'frequency_band',
                           'interval',
                           'recall_macro', 'error_bar_th', 'error_bar_emp', 'n_test_per_class',
                           'recall_macro_train_per_fold', 'recall_macro_test_per_fold',
                           'seed', 'n_splits',
                           'data_size', 'n_chans', 'window_size',
                           'y_true_per_fold', 'y_pred_per_fold'] ]
    
    # Save to file
    file_name = (base_path + 'results/summary/'
            + 'summary_all_sess_'+decode_for+'.csv')
    file_exists = os.path.isfile(file_name)
    if file_exists :
        print(file_name," already exists. Summary file hasn't been saved.")
        #with open(file_name, 'a') as f:
            #result.to_csv(f, mode ='a', index=False, header=False)
    else:
        with open(file_name, 'w') as f:
            result.to_csv(f, mode ='w', index=False, header=True)
        print(file_name, " Succesfully saved")
        
        
