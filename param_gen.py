 
import sys
import os
import os.path

import lib.cnn.matnpyio as io

# path

base_path =
raw_path = '/media/rudy/disk2/lucy/'
path_out = base_path + 'scripts/_params/training.txt'
# rinfo_path = raw_path +sess_no+'/session01/' + 'recording_info.mat'


# PARAMS


#session = [sess_no]
# to run on all session
session = os.listdir(raw_path)
session.remove('unique_recordings.mat')

decoders = ['stim', # decode the class of the stimulus
            'resp'] # decode the response of the monkey (succes or fail)

# intervals = [ [align_on, from_time, to_time] ]
intervals = [['sample', -500, 0], # pre-sample
             ['sample', 0, 500], # sample
             ['sample', 500, 1000], # early delay
             ['match', -500, 0], # late delay / pre-match
             ['match', 0, 500]] # match

# frequency_band = [[lowcut, highcut]]
frequency_band = [[4,8], # theta
                  [8,12], # alpha
                  [12,30], # beta
                  [30,100], # gamma 
                  [80,300]] 
order = 3 # order of the filter

#target_areas = [[area1, area2, ... ]] 
target_areas = []
# target_cortex = [[cortex1, cortex2, ...]] 
target_cortex = [['Visual'],
                 ['Motor'],
                 ['Somatosensory'],
                 ['Prefontal'],
                 ['Parietal'],
                 ['Visual', 'Motor','Somatosensory', 'Prefontal', 'Parietal'] ]

with open(path_out, 'w') as f:
    f.write('')
    
    
total_runs = 0
for decode_for in decoders :
    for lowcut, highcut in frequency_band :
        for sess_no in session :
            
            rinfo_path = raw_path +sess_no+'/session01/' + 'recording_info.mat'
            
            for cortex_list in target_cortex :
                areas = []
                for cortex in cortex_list :
                    areas_cortex = io.get_area_cortex(rinfo_path, cortex, unique = True)
                    areas.append(areas_cortex)
                
                if len(areas) !=0 :
                    for align_on, from_time, to_time in intervals :
                        
                         print_str = '{}, {}, {}, {}, {}, {}'.format(
                             sess_no, decode_for, str(areas), 
                             align_on+'_from'+str(from_time)+'_to'+str(to_time), 
                             'low'+str(lowcut)+'high'+str(highcut)+'order'+str(order),
                             str(cortex_list) )

                        
                        params = [sess_no, 
                                  decode_for, 
                                  areas, 
                                  align_on, from_time, to_time,
                                  lowcut, highcut, order,
                                  cortex_list,
                                  print_str]
                        
                        with open(path_out, 'a') as f:
                            f.write('\n' + str(params))
                            
                        total_runs +=1
            
            for areas in target_areas:
                areas_available = get_area_names(rinfo_path) # areas of the session
                
                if set(areas) < set(areas_available) : # if all areas are available 
                    cortex_list = 'None'
                    
                    for align_on, from_time, to_time in intervals :
                        
                        
                         print_str = '{}, {}, {}, {}, {}, {}'.format(
                             sess_no, decode_for, str(areas), 
                             align_on+'_from'+str(from_time)+'_to'+str(to_time), 
                             'low'+str(lowcut)+'high'+str(highcut)+'order'+str(order),
                             str(cortex_list) )

                        
                        params = [sess_no, 
                                  decode_for, 
                                  areas, 
                                  align_on, from_time, to_time,
                                  lowcut, highcut, order,
                                  cortex_list,
                                  print_str]
                        with open(path_out, 'a') as f:
                            f.write('\n' + str(params))
                            
                        total_runs +=1
                        
            #all_areas = get_area_names(rinfo_path)
            #for areas in all_areas:
                    #cortex_list = 'None'
                    
                    #for align_on, from_time, to_time in intervals :
                        
                        
                         #print_str = '{}, {}, {}, {}, {}, {}'.format(
                             #sess_no, decode_for, str(areas), 
                             #align_on+'_from'+str(from_time)+'_to'+str(to_time), 
                             #'low'+str(lowcut)+'high'+str(highcut)+'order'+str(order),
                             #str(cortex_list) )

                        
                        #params = [sess_no, 
                                  #decode_for, 
                                  #areas, 
                                  #align_on, from_time, to_time,
                                  #lowcut, highcut, order,
                                  #cortex_list,
                                  #print_str]
                        #with open(path_out, 'a') as f:
                            #f.write('\n' + str(params))
                            
                        #total_runs +=1
                    
                        
                    
                    
print(total_runs) # print to pass length of array to shell

            

        

            
    


