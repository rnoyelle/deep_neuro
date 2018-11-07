 
import sys
import os
import os.path
import re

import pandas as pd
import numpy as np

from scipy.special import erf

def pval(n_test, r, classes):
    """
    returns the p_value for the null hypothesis
    """
    n_test = np.array(eval(n_test.replace('.',',')))
    
    # Pi = np.array( classes * [1/classes] )
    # Pi = 0.5 + n_test * (1 - classes/2)/np.sum(n_test)
    # std = np.sqrt( np.sum( Pi * (1 -Pi)/n_test ) )/classes
    std = np.sqrt (np.sum(1/n_test) - (classes-2)**2/np.sum(n_test) ) /( 2 * classes )
    
    return 1 - erf( abs( r - 1/classes)/ (np.sqrt(2) * std ) ) 

def sign(p):
    '''
    return the sign according of the p_value
    '''
    sign = '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else 0
    return(sign)

### PATH
base_path = '/home/rudy/Python/rnoyelle_github_test/'  # .../my_project/
#raw_path = '/media/rudy/disk2/lucy/' # .../my_project/data/raw/



# DECODER
decoders = ['stim', 'resp'] # ['stim']

for decode_for in decoders :
    if decode_for == 'stim':
        classes = 5
    else:
        classes = 2

    ### SESSION
    # this way or
    #session = os.listdir(raw_path)
    #session.remove('unique_recordings.mat')
    # or this : 
    available_file = os.listdir(base_path + 'results/training/')
    session = []
    for i in range(len(available_file)):
        txt = re.split('_|\.', available_file[i]) # = [sess_no, 'training', decode_for', 'csv']
        if txt[2] == decode_for :
            session.append(txt[0])

    ## merge files
    print(len(session))
    df_session = len(session) * [0]

    for count, sess_no in enumerate(session):
        file_name = base_path + 'results/training/'+ sess_no + '_training_'+decode_for+'.csv'
        df_session[count] = pd.read_csv(file_name)


    if len(session) > 1 :
        result = pd.concat(df_session, ignore_index=True)
    elif len(session) == 1 :
        result = df_session[0]
    else:
        print('No file for decode_for = ', decode_for)
        break

    
    result = result[ ['session', 'decode_for', 'only_correct_trials',
                      'areas', 'cortex', 'elec_type',
                      'frequency_band',
                      'interval',
                      'recall_macro', 'error_bar_th', 'error_bar_emp',
                      'seed', 'n_splits',
                      'n_test_per_class', 'data_size', 'n_chans', 'window_size'] ]
    
  
    
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
     
    # keep only these columns
    result = result[ ['session', 'decode_for', 'only_correct_trials',
                      'areas', 'cortex', 'elec_type',
                      'frequency_band',
                      'interval',
                      'recall_macro', 'error_bar_th', 'error_bar_emp',
                      'seed', 'n_splits',
                      'n_test_per_class', 'data_size', 'n_chans', 'window_size'] ]
    
    ## ADD pval and sign
    
    result['pval'] = result.apply(lambda row: pval(row.n_test_per_class, row.recall_macro, classes), axis=1)
    result['sign'] = result.apply(lambda row : sign(row.pval), axis=1)
    
    # Save file
    file_name = (base_path + 'results/pvals/'
            + 'pval_all_sess_'+decode_for+'.csv')
    file_exists = os.path.isfile(file_name)
    # if file already exist, just append the data.
    if file_exists :
        print(file_name," already exists. Summary file hasn't been saved.")
        # to append to the file
        #with open(file_name, 'a') as f:
            #df.to_csv(f, mode ='a', index=False, header=False)
    else:
        with open(file_name, 'w') as f:
            result.to_csv(f, mode ='w', index=False, header=True)
        print(file_name, " Succesfully saved")

        
