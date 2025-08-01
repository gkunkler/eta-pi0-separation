import h5py
import numpy as np
from tqdm import tqdm

labels = ['NCPi0', 'Eta']
# file_paths = ['../eta-pi-data/merged_NCPi0.h5', '../eta-pi-data/merged_Eta.h5']
file_paths = ['../eta-pi-data/merged_Eta.h5', '../eta-pi-data/merged_NCPi0.h5']
output_file_path = '../eta-pi-data/eta-pi0.h5'

# Take the event_id info and adds an adjacent dataset that has the event index, starting index, and number of elements for the supplied groups
def create_sequence_info(file_path, groups=['spacepoint_table', 'hit_table'], rewrite=False):

    with h5py.File(file_path, 'r+') as f:

        for group in groups:

            
            grp = f[group]

            if 'sequence_info' in grp.keys():
                if rewrite:
                    print('sequence_info already found. Deleting...')
                    del grp['sequence_info']
                else:
                    print('sequence_info already found. Skipping...')
                    continue
           
            print(f'Creating sequence_info for {group} in {file_path}')
                
            if 'event_id' not in grp.keys():
                print(f'event_id not found in hdf5 group {group}')
                continue

            if 'event_id.seq_cnt' in grp.keys():
                print(f'Found seq_cnt, so using that to speed up calculation')

                event_index = np.array(grp['event_id.seq_cnt'][:,0], dtype=np.int32)
                num_points = np.array(grp['event_id.seq_cnt'][:,1], dtype=np.int32)
                starting_index = np.zeros(len(num_points), dtype=np.int32)
                starting_index[1:] = np.cumsum(num_points[:-1])

                sequence_info = np.stack((event_index, starting_index, num_points), dtype=np.int32).transpose()

                print(sequence_info)
                 
                grp.create_dataset('sequence_info', data=sequence_info)

                # grp['sequence_info'] = sequence_info

                continue

            else:

                print(f'Did not find seq_cnt. Skipping...')

            # num_elem = []
            # starting_index = [0]

            # current_index = grp['event_id'][0]
            # current_num_elem = 0
            # i = 0
            # for a in tqdm(grp['event_id'][:]):
                
            #     # Increment the number of points 
            #     if a[0] == current_index[0] and a[1] == current_index[1] and a[2] == current_index[2]:
            #         current_num_points += 1
            #     else:
            #         num_elem.append(current_num_points)
            #         starting_index.append(i)

            #         current_num_points = 1
            #         current_index = a

            #     i+=1
            # num_elem.append(current_num_points) # Add the final event as well even if there is no new event info to trigger it
            # event_index = range(len(num_elem)) # There is no event_index from seq_cnt, so we create another index


for file_name, label in zip(file_paths, labels):

    create_sequence_info(file_name, rewrite=False)
    # Now I can assume that there is a dataset called sequence_info with the information I need to easily interpret hit_id information

    with h5py.File(file_name, 'r') as f:

        # Get references to each table we will be using
        sp = f['spacepoint_table']
        h = f['hit_table']

        if 'sequence_info' not in sp.keys() or 'sequence_info' not in h.keys():
            print(f'Did not find sequence_info in {file_name}. Skipping...')
            continue

        # Get the rse values for all space point events
        starting_index = sp['sequence_info'][:,1]
        # rse = sp['event_id'][starting_index] # Takes a while

        # Numpy arrays to store the data from the hdf5 file
        sequence_info_sp = sp['sequence_info'][:]
        sequence_info_h = h['sequence_info'][:]
        hit_id_sp = sp['hit_id'][:,2]
        hit_id_h = h['hit_id'][:]
        hit_plane = h['local_plane'][:]
        hit_integral = h['integral'][:]
        sp_positions = sp['position'][:]

        # Lists to store the data that will be put into the new file
        sequence_info_new = [] # Contains event_id, starting_index, num_points (length is the number of events)
        point_info_new = [] # Contains x, y, z, integral
        point_info_all = [] # Contains point_info_new but for all space points (with potentially duplicate charge)
        point_info_centered_new = [] # Contains centered_x, centered_y, centered_z, integral

        j = 0 # Index to start looking event_index in sequence_index_h
        for i in tqdm(range(len(sequence_info_sp[:]))):

            # Get the event sequence_data in spacepoint_table
            event_index, starting_index, num_points = sequence_info_sp[i]
            hit_ids = hit_id_sp[starting_index:starting_index+num_points]
            # hit_ids = [id for id in hit_id_sp[starting_index:starting_index+num_points] if id != -1] # Filter out the -1 values ahead of time for more efficiency

            # Find the matching event index in hit_table
            #  Assumes that sequence_info_h[:,0] has only unique values
            #  Assumes that sequence_info_h[:,0] is a sorted subset of the sorted sequence_info_sp[:,0]
            found_j = False
            while j < len(sequence_info_h) and not found_j:
                if sequence_info_h[j, 0] == event_index:
                    found_j = True
                else:
                    j+=1
            if not found_j:
                raise IndexError(f'Reached the end of hit_table/hit_id without finding desired index from spacepoint_table/hit_id.\nAssumption of unique, sorted values may be wrong.')

            event_index_h, starting_index_h, num_points_h = sequence_info_h[j]
            hit_ids_h = hit_id_h[starting_index_h:starting_index_h+num_points_h]

            # print(np.all(hit_ids_h.transpose() == np.array(range(0,num_points_h)))) # Shows that the hit_ids are in order for each

            # The charge integral on y-plane associated with each hit from the current event
            #  Assumes the hit_ids in the hit_table are in order starting at zero
            integrals = hit_integral[starting_index_h+hit_ids][:,0] 
            planes = hit_plane[starting_index_h+hit_ids][:,0]
            included_hit_ids = []

            # Get the first space point for each hit_id_y and add it to the new list
            for k in range(len(hit_ids)):
                hit_id = hit_ids[k]
                if hit_id == -1:
                    continue

                if planes[k] != 2:
                    print(event_index)
                    print(hit_ids)
                    print(planes)
                    raise ValueError(f'Calculating a hit integral for the wrong plane. Check your assumptions.')

                # Add point info to the new lists
                x, y, z = sp_positions[starting_index+k,:]
                integral = integrals[k]
                point_info_all.append([x,y,z,integral])
                
                if hit_id not in included_hit_ids:
                    included_hit_ids.append(hit_id)

                    point_info_new.append([x,y,z,integral])
                    
            # Update the starting_index and num_points with the reduced size
            if len(sequence_info_new) > 0:
                starting_index_new = sequence_info_new[-1][1] + sequence_info_new[-1][2]
            else: 
                starting_index_new = 0
            num_points_new = len(included_hit_ids)
            sequence_info_new.append([event_index, starting_index_new, num_points_new])

            # Calculate the center of the points in the current event and subtract that from the positions to center them
            points_new = np.array(point_info_new[-num_points_new:])
            
            com = np.mean(points_new, axis=0)[0:3]
            points_centered_new = points_new[:,0:3] - com
            
            point_info_centered_new.extend(np.hstack((points_centered_new, points_new[:,3].reshape(-1,1))))

        # Convert to numpy arrays
        sequence_info_new = np.array(sequence_info_new)
        point_info_new = np.array(point_info_new)
        point_info_centered_new = np.array(point_info_centered_new)

        print(sequence_info_new)
        print(point_info_new)
        print(point_info_centered_new)
                # if hit_plane[starting_index_h+hit_index] != 2:
                # print(hit_plane[starting_index_h+hit_index])

    print(f'Writing {label} to {output_file_path}')

    with h5py.File(output_file_path, 'a') as f:

        if label in f.keys():
            del f[label]
        grp = f.create_group(label)

        grp.create_dataset('sequence_info_filtered', data=sequence_info_new)
        grp.create_dataset('sequence_info_all', data=sequence_info_sp)
        grp.create_dataset('point_info_filtered', data=point_info_new)
        grp.create_dataset('point_info_all', data=point_info_all)
        grp.create_dataset('point_info_centered', data=point_info_centered_new)

        print(f'Created {label} with the datasets: {list(grp.keys())}')
                

# for rse_tuple in tqdm(rse_tuples_to_preload, desc="Preloading data & Calc. Angles/Features"):
#     #Calculate angle for curr RSE
#     calculated_angle = calculate_e_p_angle(h5f_preload, rse_tuple) # Pass h5f_preload\
#     if calculated_angle is None:
#         self.all_preloaded_targets[rse_tuple] = 0.0 # Assign a default value for target if not found
#     else:
#         self.all_preloaded_targets[rse_tuple] = calculated_angle
#     #Count nPC/nSP for the current RSE
#     n_pc, n_sp = count_points_and_hits(h5f_preload, rse_tuple) # Pass h5f_preload, not path
#     self.all_preloaded_meta_n_counts[rse_tuple] = {'nPC': n_pc, 'nSP': n_sp}
#     #Get spacepoint position and integral
#     rse_matches_mask_sp = np.all(all_spacepoint_event_ids == np.array(rse_tuple, dtype=np.int32), axis=1)
#     if np.any(rse_matches_mask_sp):
#         filtered_sp_positions = all_spacepoint_positions[rse_matches_mask_sp]
#         filtered_sp_hit_ids = all_spacepoint_hit_ids[rse_matches_mask_sp] # For hit_id lookup
#         #Prep hit id for integral lookup
#         integral_y_values = np.zeros(len(filtered_sp_positions), dtype=np.float32) # Default to 0.0
#         if hit_table_exists and np.any(np.all(all_hit_table_event_ids == np.array(rse_tuple, dtype=np.int32), axis=1)):
#             # Filter hit data for the current RSE
#             rse_matches_mask_hit_current = np.all(all_hit_table_event_ids == np.array(rse_tuple, dtype=np.int32), axis=1)
#             filtered_hit_ids_current = all_hit_table_hit_ids[rse_matches_mask_hit_current].flatten()
#             filtered_hit_integrals_current = all_hit_table_integrals[rse_matches_mask_hit_current].flatten()
#             filtered_hit_planes_current = all_hit_table_planes[rse_matches_mask_hit_current].flatten()
#             df_hits_in_rse = pd.DataFrame({
#                 'hit_id': filtered_hit_ids_current,
#                 'integral': filtered_hit_integrals_current,
#                 'local_plane': filtered_hit_planes_current
#             })
#             hit_id_to_integral_map = df_hits_in_rse[df_hits_in_rse['local_plane'] == 2].set_index('hit_id')['integral'].to_dict()
#             integral_y_values = np.array([
#                 hit_id_to_integral_map.get(hit_id_arr[2], 0.0) # Index 2 for plane 2 hit_id (y)
#                 for hit_id_arr in filtered_sp_hit_ids
#             ], dtype=np.float32)
#         else:
#             pass # integral_y_values remains 0.0 if no hits or table missing for this event
#         self.all_preloaded_pc_data[rse_tuple] = {
#             'x': filtered_sp_positions[:, 0].copy(),
#             'y': filtered_sp_positions[:, 1].copy(),
#             'z': filtered_sp_positions[:, 2].copy(),
#             'integral_y': integral_y_values.copy()
#         }
#         if integral_y_values.size > 0:
#             temp_all_integral_y_values.extend(integral_y_values.flatten())
#     else: # No space points for this RSE
#         self.all_preloaded_pc_data[rse_tuple] = {
#             'x': np.array([]), 'y': np.array([]), 'z': np.array([]), 'integral_y': np.array([])
#         }
#         print(f"Warning: No space points found for RSE {rse_tuple}. Storing empty arrays.")
#     print(f"Finished preloading {len(self.all_preloaded_pc_data)} PC event data sets and angles/features into RAM.")