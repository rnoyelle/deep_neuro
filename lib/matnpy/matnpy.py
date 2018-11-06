import sys
import os 
import os.path

import numpy as np

from . import matnpyio as io# matnpyio as io
from . import preprocess as pp#import preprocess as pp



def get_preprocessed_from_raw(sess_no, raw_path, align_on, from_time, to_time, lowcut, highcut, order) :
    """Gets raw data and preprocess them.
    
    Args:
        sess_no: A str. num of the session
        raw_path: A str. Path to the trial_info file.
        align_on: A str. One of 'sample' , 'match'.
        from_time : A float. in ms
        to_time :  A float. in ms. cuts the trial between from_time and to_time. 0 correspond at the onset time of align_on.
        lowcut : A float. in Hz
        highcut : A float. in Hz. filters the trials between lowcut and highcu
        order : A float. order of the frequency filter.
        
        
    Returns:
        Ndarray of filtered data. 
    """
    
    #params
    sess = '01'
       
    trial_length = abs(from_time - to_time)

    # Paths
    #raw_path = base_path + 'data/raw/' + sess_no + '/session' + sess + '/'
    rinfo_path = raw_path + 'recording_info.mat'
    tinfo_path = raw_path + 'trial_info.mat'

    # Define and loop over intervals
    
    srate = io.get_sfreq(rinfo_path) # = 1 000
    n_trials = io.get_number_of_trials(tinfo_path) 
    last_trial = int(max(io.get_trial_ids(raw_path)))
    n_chans = io.get_number_of_channels(rinfo_path)
    channels = [ch for ch in range(n_chans)]

    # Pre-process data
    filtered = np.empty([n_trials,
                        len(channels),
                        int(trial_length * srate/1000)])

    trial_counter = 0; counter = 0
    while trial_counter < last_trial:
        n_zeros = 4-len(str(trial_counter+1))
        trial_str = '0' * n_zeros + str(trial_counter+1)  # fills leading 0s
        if sess == '01' :
            file_in = sess_no + '01.' + trial_str + '.mat'
        else :
            file_in = sess_no + '02.' + trial_str + '.mat'
            
        if align_on == 'sample' :        
            onset = io.get_sample_on(tinfo_path)[trial_counter].item()
        elif align_on == 'match' :
            onset = io.get_match_on(tinfo_path)[trial_counter].item()
        else :
            print("Petit problème avec align_on : 'sample' ou 'match' ")
            

        
        if np.isnan(onset):  # drop trials for which there is no onset info
            print('No onset for ' + file_in)
            trial_counter += 1
            if trial_counter == last_trial:
                break
            else:
                counter += 1
                continue
        print(file_in)
        try:
            raw = io.get_data(raw_path + file_in)
            temp = pp.strip_data(raw,
                                rinfo_path,
                                onset,
                                start=from_time,
                                length=trial_length)
            temp = pp.butter_bandpass_filter(temp,
                                            lowcut,
                                            highcut,
                                            srate,
                                            order)
            if temp.shape[1] == trial_length:  # drop trials shorter than length
                filtered[counter] = temp
            counter += 1
        except IOError:
            print('No file ' + file_in)
        trial_counter += 1

    # Return data

    filtered = np.array(filtered)
    return(filtered)


def get_subset_by_cortex(sess_no, raw_path, 
                         align_on, from_time, to_time,
                         lowcut, highcut, 
                         cortex,
                         epsillon = 100, order = 3,
                         only_correct_trials = True, renorm = True ):
    """Gets raw data and preprocess them. Select data trials and channels according to inputs 
    
    Args:
        sess_no: A str. num of the session
        raw_path: A str. Path to the trial_info file.
        align_on: A str. One of 'sample' , 'match'.
        from_time : A float. in ms
        to_time :  A float. in ms. cuts the trial between from_time and to_time. 0 correspond at the onset time of align_on.
        lowcut : A float. in Hz
        highcut : A float. in Hz. filters the trials between lowcut and highcu
        order : A float. order of the frequency filter.
        cortex : A str. One of 'Visual', 'Parietal', 'Prefrontal', 'Motor', 'Somatosensory'
        epsillon : A float. Load a little bit more in time to prevent zero padding of the frequency filters. then the excess is cut before return.
        only_correct_trials : A boolean. if True, only trials where the monkey succeed the task are selected.
        renorm : A boolean. If True, data are normed after time cut and frequency filters        
        
    Returns:
        Ndarray of filtered data. 
    """
    tinfo_path = raw_path + 'trial_info.mat'
    rinfo_path = raw_path + 'recording_info.mat'
    
    # get all data
    data_filtered = get_preprocessed_from_raw(sess_no, raw_path, 
                                                               align_on, from_time - epsillon, to_time + epsillon, 
                                                               lowcut, highcut, order)
        
    # don't keep missing data // keep only_correct_trials if True
    
    responses = io.get_responses(tinfo_path)
    if only_correct_trials == False:
        ind_to_keep = (responses == responses).flatten()
    else:
        ind_to_keep = (responses == 1).flatten()
        
    #data1 =data1[ind_to_keep, :, :] # in the same time
    #data2 =data2[ind_to_keep, :, :]
    
    data_filtered = data_filtered[ind_to_keep,:,:]

    
    # select electrode
    
    dico_area_to_cortex = io.get_dico_area_to_cortex()
    area_names = io.get_area_names(rinfo_path)
    
    dtype = [('name', '<U6'), ('index', int), ('cortex', '<U16')]
    values = []
    for count, area in enumerate(area_names):
        if area in dico_area_to_cortex: # if not, area isn't in Visual or Parietal or Prefontal or Motor or Somatosensory
            
            values.append( (area, count, dico_area_to_cortex[area])  )
        else:
            print('Unknow area')
                    
    s = np.array(values, dtype=dtype)
    
    elec = s[s['cortex'] == cortex]['index']
    
    data_filtered = data_filtered[:, elec, epsillon : -epsillon ]

                                    
    
    ### variable for shape
    #n_chans1 = len(elec1)
    #n_chans2 = len(elec2)
            
    #samples_per_trial1 = data1.shape[2] # = window_size1
    #samples_per_trial2 = data2.shape[2] # = window_size2
    
    # renorm data : mean = 0 and var = 1
    if renorm == True :
        data_filtered = pp.renorm(data_filtered)

    ## change type 
    data_filtered = data.astype(np.float32)

    
    return( data_filtered )

def get_subset_by_areas(sess_no, raw_path, 
                         align_on, from_time, to_time,
                         lowcut, highcut, 
                         target_areas,
                         epsillon = 100, order = 3,
                         only_correct_trials = True, renorm = True, elec_type = 'grid' ):

        """Gets raw data and preprocess them. Select data trials and channels according to inputs 
    
    Args:
        sess_no: A str. num of the session
        raw_path: A str. Path to the trial_info file.
        align_on: A str. One of 'sample' , 'match'.
        from_time : A float. in ms
        to_time :  A float. in ms. cuts the trial between from_time and to_time. 0 correspond at the onset time of align_on.
        lowcut : A float. in Hz
        highcut : A float. in Hz. filters the trials between lowcut and highcu
        order : A float. order of the frequency filter.
        target_areas : A list. list of areas to select.
        elec_type : A str. One of 'single' (use all electrodes within area as single trials), 
                                  'grid' (use whole electrode grid), 
                                  'average' (mean over all electrodes in area).
        epsillon : A float. Load a little bit more in time to prevent zero padding of the frequency filters. then the excess is cut before return.
        only_correct_trials : A boolean. if True, only trials where the monkey succeed the task are selected.
        renorm : A boolean. If True, data are normed after time cut and frequency filters        
        
    Returns:
        Ndarray of filtered data. 
    """
    tinfo_path = raw_path + 'trial_info.mat'
    rinfo_path = raw_path + 'recording_info.mat'
    
    # get all data
    data_filtered = get_preprocessed_from_raw(sess_no, raw_path, 
                                                               align_on, from_time - epsillon, to_time + epsillon, 
                                                               lowcut, highcut, order)
        
    # don't keep missing data // keep only_correct_trials if True
    
    responses = io.get_responses(tinfo_path)
    if only_correct_trials == False:
        ind_to_keep = (responses == responses).flatten()
    else:
        ind_to_keep = (responses == 1).flatten()
        
    #data1 =data1[ind_to_keep, :, :] # in the same time
    #data2 =data2[ind_to_keep, :, :]
    
    data_filtered = data_filtered[ind_to_keep,:,:]

    
    # select electrode and cut the additionnal time
    
    area_names = io.get_area_names(rinfo_path)
    
    idx = []
    for count, area in enumerate(area_names):
        if area in target_areas:
            idx.append(count)         
            
    if epsillon != 0 :    
        data_filtered = data_filtered[:, idx, epsillon : -epsillon ]
    else:
        data_filtered = data_filtered[:, idx, :]
                                      
    

    ## change type 
    data_filtered = data_filtered.astype(np.float32)
    
    if elec_type == 'single':
        data_filtered = data_filtered.reshape(data_filtered.shape[0]*data_filtered.shape[1], data_filtered.shape[2])
        data_filtered = np.expand_dims(data_filtered, axis=1)
               

    
    elif elec_type == 'average':
        data_filtered = np.mean(data_filtered, axis=1, keepdims=True)

            
    #elif elec_type == 'grid':
        #data_filtered = data_filtered

    elif elec_type != 'grid':
        raise ValueError('Type \'' + elec_type + '\' not supported. Please ' + 
                         'choose one of \'single\'|\'grid\'|\'average\'.')
    
    # renorm data : mean = 0 and var = 1
    if renorm == True :
        data_filtered = pp.renorm(data_filtered)
        
    ### variable for shape
    #n_chans1 = len(idx)
            
    #samples_per_trial = data_filtered.shape[2] 
    
    return( data_filtered )





    
    
