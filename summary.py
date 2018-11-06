import sys
import os
import os.path

import pandas as pd
import numpy as np



base_path =  # .../my_project/
#raw_path = base_path + 'data/raw/' # .../my_project/data/raw/

          
session = 
#session = os.listdir(raw_path)
#session.remove('unique_recordings.mat')

decoders = ['stim', 'resp'] # ['stim']

for decode_for in decoders :
    
    
    ## merge files
    df_session = len(session) * [0]
    
    for count, sess_no in enumerate(session):
        df_session[count] = pd.read_csv(base_path + 'results/training/' + sess_no + '_training_'+decode_for+'.csv')
        
    result = pd.concat(df_session, ignore_index=True)
    
    
    # select columns
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
     
    # keep only columns
    result = result[ ['session', 'decode_for', 'only_correct_trials',
                           'areas', 'cortex', 'elec_type',
                           'frequency_band',
                           'interval',
                           'recall_macro', 'error_bar_th', 'error_bar_emp', 'n_test_per_class',
                           'recall_macro_train_per_fold', 'recall_macro_test_per_fold',
                           'seed', 'n_splits',
                           'data_size', 'n_chans', 'window_size',
                           'y_true_per_fold', 'y_predict_per_fold'] ]
    
    # Save to file
    file_name = base_path + 'results/summary/'
            + 'summary_all_sess_'+decode_for+'.csv'
    file_exists = os.path.isfile(file_name)
    if file_exists :
        with open(file_name, 'a') as f:
            df.to_csv(f, mode ='a', index=False, header=False)
    else:
        with open(file_name, 'w') as f:
            df.to_csv(f, mode ='w', index=False, header=True)
        
        
