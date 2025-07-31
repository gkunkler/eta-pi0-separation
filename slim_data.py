import h5py
import numpy as np
from tqdm import tqdm

file_paths = ['../eta-pi-data/merged_NCPi0.h5', '../eta-pi-data/merged_Eta.h5']
# file_paths = ['../eta-pi-data/merged_Eta.h5', '../eta-pi-data/merged_NCPi0.h5']
# output_file_path = '../eta-pi-data/eta-pi0.h5'

# with h5py.File(output_file_path, 'w') as f_w:

for file_name in file_paths:

    with h5py.File(file_name, 'r') as f:

        print(f.keys())

        sp = f['spacepoint_table']

        try:
            # If it exists, the seq_cnt array contains the num_points precalculated
            num_points = f['spacepoint_table/event_id.seq_cnt'][:,1]
            starting_index = [0] # The first event starts at the beginning

            for i in range(len(num_points)-1):
                starting_index.append(starting_index[i] + num_points[i]) # Cumulatively add the number of points in each event
        except:
            # Otherwise make it manually
            num_points = []
            starting_index = [0]

            current_index = sp['event_id'][0]
            current_num_points = 0
            i = 0
            for a in tqdm(sp['event_id'][:]):
                
                if a[0] == current_index[0] and a[1] == current_index[1] and a[2] == current_index[2]:
                    current_num_points += 1
                else:
                    num_points.append(current_num_points)
                    starting_index.append(i)

                    current_num_points = 1
                    current_index = a

                i+=1
            num_points.append(current_num_points) # Add the final event as well even if there is no new event info to trigger it
        

        num_points = np.array(num_points)
        starting_index = np.array(starting_index)

        # event_label = sp['event_id'][starting_index]

        print(f'num_points: {num_points} ({len(num_points)} items)')
        print(f'starting_index: {starting_index} ({len(starting_index)} items)')

        metadata = np.stack((starting_index, num_points)).transpose()

    with h5py.File(file_name, 'r+') as f:

        sp = f.require_group('spacepoint_table')
    
        sp.create_dataset('metadata', data=metadata)

        # metadata = f.create_dataset('event_metadata', (10000,5), maxshape=(5,None))
        


        # f.create_dataset('event_metadata')




