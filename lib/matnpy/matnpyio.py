 
import h5py
import numpy as np
import scipy.io

from os import listdir


# GET with session_path

def get_trial_ids(session_path):
    """get_trial_ids scans (raw) files in a session directory and thereby
    checks what trials are actually available"""
    # Directory will contain trials of format 'session.trial.mat' and rinfo,
    # tinfo. In order to get only trial fnames, split at '.' and select only
    # files that were split into 3 (vs. 2) parts.
    trial_ids = [f.split('.')[1].replace('_raw', '') for f in listdir(session_path) if len(f.split('.')) == 3]
    trial_ids.sort()
    return np.array(trial_ids)

# GET file .mat 

def get_data(file_path):
    """get_data gets raw data for a given trial from the Grey data set (*.mat).
    returns an ndarray."""
    with h5py.File(file_path, 'r') as f:
        #return np.transpose(np.array([sample for sample in f['raw_data']]))
        return np.transpose(f['lfp_data'][:])


# GET with rinfo_path


def get_sfreq(rinfo_path):
    f = scipy.io.loadmat(rinfo_path) 
    rinfo = f['recording_info']
    return( float( rinfo[0,0][8][0][0] ))


def get_number_of_channels(rinfo_path):
    f = scipy.io.loadmat(rinfo_path) 
    rinfo = f['recording_info']
    
    return( int( rinfo[0,0][3][0][0] ) )

def get_area_names(rinfo_path):
    #sess_no = '141014'
    #rinfo_path = base_path + sess_no + '/session01/recording_info.mat'

    f = scipy.io.loadmat(rinfo_path)
    rinfo = f['recording_info']

    area_names = []
    for a in rinfo[0,0][5][0] : 
        area_names.append(str(a[0]))
    return( np.array( area_names) )

def get_area_cortex(rinfo_path, cortex, unique = True):
    '''
    return areas in the cortex area.
    
    Args:
       rinfo_path : A str. path to recording_info.mat
       cortex : A str. cortex name
       unique: A boolean. if True, return only unique area.
    
    '''

    dico_area_to_cortex = get_dico_area_to_cortex()
    area_names = get_area_names(rinfo_path)
    
    areas = []
    
    for area in area_names :
        if dico_area_to_cortex[area] == cortex :
            areas.append(area)
    if unique == True :
        return( np.unique( areas ) )
    else:
        return( areas )

#     dico_area_to_cortex = get_dico_area_to_cortex()
#     area_names = get_area_names(rinfo_path)
    
#     dtype = [('name', '<U6'), ('index', int), ('cortex', '<U16')]
#     values = []
#     for count, area in enumerate(area_names):
#         if area in dico_area_to_cortex: # if not, area isn't in Visual or Parietal or Prefontal or Motor or Somatosensory
            
#             values.append( (area, count, dico_area_to_cortex[area])  )
#         else:
#             print('Unknow area')
                    
#     s = np.array(values, dtype=dtype)
    
#     areas = s[s['cortex'] == cortex]['name']
    
#     if unique == True :
#         return( np.unique( areas ) )
#     else:
#         return( np.array(areas) )
    
    
## not for the cnn

def get_image_names(rinfo_path):
    f = scipy.io.loadmat(rinfo_path)
    rinfo = f['recording_info']
    
    image_names = [ rinfo['image_names'][0,0][0,i][0] for i in range(5) ]
    
    return(image_names)

def get_image(rinfo_path):
    f = scipy.io.loadmat(rinfo_path) 
    rinfo = f['recording_info']
    image_data = [rinfo['image_data'][0,0][0,i] for i in range(5) ]
    return(image_data)

###########" GET with tinfo_path


def get_responses(tinfo_path):
    """Gets the responses for all trials in a given session."""
    with h5py.File(tinfo_path, 'r') as f:
        # 'trial_info.mat' holds only 1 structure
        tinfo = f['trial_info']
        
        # Extract and return responses
        return np.array(tinfo['behavioral_response'])
    
def get_samples(tinfo_path):
    """Gets sample image classes for all trials in a given session. """
    with h5py.File(tinfo_path, 'r') as f:
        # 'trial_info.mat' holds only 1 structure
        tinfo = f['trial_info']
        
        # Extract and return sample classes. substract 1 to have classes range
        # from 0 to 4 instead of 1 to 5
        return np.array([k-1 for k in tinfo['sample_image']])
    
#key
#[u'behavioral_response', u'match_image', u'match_location', u'match_on', u'nonmatch_image', u'nonmatch_location', u'num_trials', u'reaction_time', u'sample_image', u'sample_location', u'sample_off', u'sample_on', u'trial_type']
def get_array(tinfo_path, key):
    with h5py.File(tinfo_path, 'r') as f:
        # 'trial_info.mat' holds only 1 structure
        tinfo = f['trial_info']
        
        # Extract and return responses
        return np.array(tinfo[key])

def get_sample_on(tinfo_path):
    """Gets sample onset times for all trials in a session. """
    with h5py.File(tinfo_path, 'r') as f:
        # 'trial_info.mat' holds only 1 structure
        tinfo = f['trial_info']
        
        # Extract and return sample onset
        return np.array(tinfo['sample_on'])
    
    
def get_match_on(tinfo_path):
    """Gets match onset times for all trials in a session. """
    with h5py.File(tinfo_path, 'r') as f:
        # 'trial_info.mat' holds only 1 structure
        tinfo = f['trial_info']
        
        # Extract and return sample onset
        return np.array(tinfo['match_on'])
    
    
def get_number_of_trials(tinfo_path):
    with h5py.File(tinfo_path, 'r') as f:
        # 'trial_info.mat' holds only 1 structure
        tinfo = f['trial_info']
        
        return int(tinfo['num_trials'][0].item())
    
    
def get_targets(decode_for, raw_path, n_chans, elec_type='grid', 
                only_correct_trials=True, onehot=True):
    """Gets behavioral responses or stimulus classes from the trial_info file
    and encodes them as one-hot.
    
    Args:
        decode_for: A str. One of 'stim' (stimulus classes) or 'resp' (behav. 
            response), depending on which info to classify the data on.
            (Defines number of classes: stim -> 5 classes, resp -> 2 classes)
        raw_path: A str. Path to the trial_info file.
        elec_type: A str. One of 'single' (use all electrodes within area as
            single trials), 'grid' (use whole electrode grid), 'average' (mean
            over all electrodes in area).
        n_chans: An int. Number of channels in the target area.
        only_correct_trials: A boolean. Indicating whether to subset for 
            trials with correct behavioral response only.
        
    Returns:
        Ndarray of one-hot targets.
    """
    # Trial info holds behavioral responses and stimulus classes
    tinfo_path = raw_path + 'trial_info.mat'
    
    # Get behavioral responses or stimulus classes, depending on user input
    if decode_for == 'stim':
        classes = 5
        targets = get_samples(tinfo_path)
    elif decode_for == 'resp':
        classes = 2
        targets = get_responses(tinfo_path)
    else:
        print('Can decode for behavioral response ("resp") or stimulus ' +
              'identity ("stim"), you entered \"' + decode_for + '\". Please ' +
              'adapt your input.')
    
    # Only keep non-NA targets
    ind_to_keep = (targets == targets).flatten()
    
    # If only_correct_trials set to True, drop targets for all incorrect trials
    if only_correct_trials == True:
        responses = get_responses(tinfo_path)
        ind_to_keep = (responses == 1).flatten()
    
    targets = targets[ind_to_keep].astype(int)
    
    # If every electrode (in an area) shall be regarded as their own trials,
    # we need to reshape targets accordingly. Every target es repeated as many
    # times as there are electrodes in the area.
    if elec_type == 'single':
        targets = np.repeat(targets, n_chans, axis=0)

    if onehot == True:
        # Convert to one-hot, return
        return np.eye(classes)[targets].reshape(targets.shape[0], classes)
    else:
        return targets.flatten()
    
    
## GET dict

def get_dico_cortex():
    ''' dico_cortex['cortex']= list of areas in cortex '''

    dico_cortex = {'Parietal': ['AIP',
    'LIP',
    'MIP',
    'PIP',
    'TPt',
    'VIP',
    'a23',
    'a5',
    'a7A',
    'a7B',
    'a7M',
    'a7op'],
    'Subcortical': ['Caudate', 'Claustrum', 'Putamen', 'Thal'],
    'Auditory': ['Core', 'MB', 'PBr'],
    'Visual': ['DP',
    'FST',
    'MST',
    'MT',
    'TEpd',
    'V1',
    'V2',
    'V3',
    'V3A',
    'V4',
    'V4t',
    'V6A'],
    'Motor': ['F1', 'F2', 'F6', 'F7'],
    'Temporal': ['Ins', 'STPc'],
    'Prefrontal': ['OPRO',
    'a11',
    'a12',
    'a13',
    'a14',
    'a24D',
    'a24c',
    'a32',
    'a44',
    'a45A',
    'a45B',
    'a46D',
    'a46V',
    'a8B',
    'a8L',
    'a8M',
    'a8r',
    'a9/46D',
    'a9/46V'],
    'Somatosensory': ['SII', 'a1', 'a2', 'a3']}
    
    return( dico_cortex )

def get_dico_area_to_cortex():
    "dico[area] = cortex"
    
    dico_cortex = get_dico_cortex()
    
    dico_area_to_cortex = {}
    for c in dico_cortex.keys():
        areas = dico_cortex[c]
        for area in areas:
            dico_area_to_cortex[area] = c
            
    return(dico_area_to_cortex)
    

    
    
    
    



