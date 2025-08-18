import h5py
import numpy as np
from tqdm import tqdm

from FileBuffer import FileBuffer

# Set the file paths and groups to place the events into
#  labels and file_paths must be the same length (or at least labels longer than file_paths)
#  May need to add numbers to properly order them for easy retrieval alphabetically

labels = ['0 NCPi0','1 Eta']
# file_paths = ['../eta-pi-data/merged_NCPi0.h5', '../eta-pi-data/merged_Eta.h5']
file_paths = ['../eta-pi-data/merged_NCPi0_update.h5', '../eta-pi-data/merged_Eta_update_sequenced.h5']
output_file_path = '../eta-pi-data/eta-pi0-update.h5'
description_filters = [[8,9,10,11,12], [0,1,2,3,4]]

# labels = ['Eta']
# file_paths = ['../eta-pi-data/NeutrinoML_TN_ts491857.h5']
# output_file_path = '../eta-pi-data/eta-small.h5'

# Set the codes for the different primary particle ids
#  Items earlier in the list will override those that are later
interaction_descriptions = [
                            ([22, 22, 13, 2212],0),
                            ([22, 22, 13, 2112],1),
                            ([22, 22, 13],2),
                            ([22, 22, 13],3),
                            ([22, 22], 4),
                            ([111, 211, -211], 5),
                            ([111, 111, 111], 6),
                            ([111, 111], 7),
                            ([111, 211], 8),
                            ([111, -211], 9),
                            ([111, 2112], 10),
                            ([111, 2212], 11),
                            ([111], 12),
                            ([13, 22], 13),
                            ([13], 14),
                            ([211], 15)]

# Take the event_id info and adds an adjacent dataset that has the event index, starting index, and number of elements for the supplied groups
def create_sequence_info(file_path, groups=['spacepoint_table', 'hit_table', 'particle_table'], rewrite=False):

    with h5py.File(file_path, 'r+') as f:

        for grp in groups:
            group = f[grp]

            # Check whether sequence_info has been generated already
            if 'sequence_info' in group.keys():
                if rewrite:
                    print(f'sequence_info already found for {grp}. Deleting...')
                    del group['sequence_info']
                else:
                    print(f'sequence_info already found for {grp}. Skipping...')
                    continue
                
            if 'event_id' not in group.keys():
                print(f'event_id not found in hdf5 group {grp}')
                continue

            if 'event_id.seq_cnt' in group.keys():
                # We rely on the seq_cnt object existing to cumulatively sum the num_points to get a starting_index

                print(f'Creating sequence_info for {grp} in {file_path}')
                print(f'Found seq_cnt, so using that to speed up calculation')

                event_index = np.array(group['event_id.seq_cnt'][:,0], dtype=np.int32)
                num_points = np.array(group['event_id.seq_cnt'][:,1], dtype=np.int32)
                starting_index = np.zeros(len(num_points), dtype=np.int32)
                starting_index[1:] = np.cumsum(num_points[:-1]) # Cumulative sum is offset to let the first index be 0

                sequence_info = np.stack((event_index, starting_index, num_points), dtype=np.int32).transpose() # Add the starting_index to the matrix
                # print(sequence_info)
                 
                group.create_dataset('sequence_info', data=sequence_info) # Creates the new dataset

            else:
                # There is a way to loop through the RSE data in event_id and find when it switches values, but this is much slower.

                print(f'Did not find seq_cnt. Skipping...')

                # num_elem = []
                # starting_index = [0]
                # current_index = grp['event_id'][0]
                # current_num_elem = 0
                # i = 0
                # for a in tqdm(grp['event_id'][:]):
                #     # Increment the number of points 
                #     if a[0] == current_index[0] and a[1] == current_index[1] and a[2] == current_index[2]:
                #         current_num_elem += 1
                #     else:
                #         num_elem.append(current_num_elem)
                #         starting_index.append(i)
                #         current_num_elem = 1
                #         current_index = a
                #     i+=1
                # num_elem.append(current_num_elem) # Add the final event as well even if there is no new event info to trigger it
                # event_index = range(len(num_elem)) # There is no event_index from seq_cnt, so we create another index

# This is where the main code starts 
# Looping through each file which will remain in separate groups named after labels
for file_name, label, description_filter in zip(file_paths, labels, description_filters):

    create_sequence_info(file_name, rewrite=False)
    # Now I can assume that there is a dataset called sequence_info with the information I need to easily interpret hit_id information

    with h5py.File(file_name, 'r') as f:

        # Get references to each table we will be using
        sp = f['spacepoint_table']
        h = f['hit_table']
        pt = f['particle_table']

        if 'sequence_info' not in sp.keys() or 'sequence_info' not in h.keys() or 'sequence_info' not in pt.keys():
            print(f'Did not find sequence_info in {file_name}. Skipping...')
            continue

        print(f'Loading data from {file_name}')

        # Numpy arrays to store the data from the hdf5 file
        sequence_info_sp = sp['sequence_info'][:]
        sequence_info_h = h['sequence_info'][:]
        sequence_info_pt = pt['sequence_info'][:]

        sp_positions = sp['position'][:] # 3D points
        hit_id_sp = sp['hit_id'][:,2] # The Y-plane hit id associated with each space point
        hit_plane = h['local_plane'][:] # The plane associated with each hit
        hit_integral = h['integral'][:] # The current integral for each hit
        
        rse = sp['event_id'][:] # The run, subrun, event data for each space point
        parent_id = pt['parent_id'][:] # The parent id (0=primary) for all particles in each event
        g4_pdg = pt['g4_pdg'][:] # The particle code for each particle (22=photon)

        # Create buffer objects if these arrays are taking up too much space
        # sp_positions_buffer = FileBuffer(sp['position'])
        # hit_plane_buffer = FileBuffer(h['local_plane'])
        # hit_integral_buffer = FileBuffer(h['integral'])
        # rse_buffer = FileBuffer(sp['event_id'])
        # parent_id_buffer = FileBuffer(pt['parent_id'])
        # g4_pdg_buffer = FileBuffer(pt['g4_pdg'])
        
        # Lists to store the data that will be put into the new file
        sequence_info_new = [] # Contains event_id, starting_index, num_points, run, subrun, event, description (length is the number of events)
        point_info_new = [] # Contains x, y, z, integral
        point_info_centered_new = [] # Contains centered_x, centered_y, centered_z, integral

        # Cap the number of events to limit RAM usage 
        event_cap = min(len(sequence_info_sp), 20000)
        print(f'Connecting spacepoint and hit data ({event_cap} events)')

        descriptions_found = np.zeros(len(interaction_descriptions) + 1)
        skipped = 0

        h_index = 0 # Index to start looking for event_index in sequence_index_h
        pt_index = 0 # Index to start looking for event_index in sequence_index_pt
        for i in tqdm(range(len(sequence_info_sp[:event_cap]))):

            # Get the event sequence_data in spacepoint_table
            event_index, starting_index, num_points = sequence_info_sp[i]
            hit_ids = hit_id_sp[starting_index:starting_index+num_points]
            run, subrun, event = rse[starting_index]
            # run, subrun, event = rse_buffer[starting_index]

            # Find the matching event index in particle_table
            #  Assumes that sequence_info_pt[:,0] has only unique values
            #  Assumes that sequence_info_pt[:,0] is a sorted subset of the sorted sequence_info_sp[:,0]
            found_pt_index = False
            while pt_index < len(sequence_info_pt) and not found_pt_index:
                if sequence_info_pt[pt_index, 0] == event_index:
                    found_pt_index = True
                else:
                    pt_index+=1
            if not found_pt_index:
                raise IndexError(f'Reached the end of particle_table/event_id without finding desired index from spacepoint_table/event_id.\nAssumption of unique, sorted values may be wrong.')

            event_index_pt, starting_index_pt, num_points_pt = sequence_info_pt[pt_index]

            # Find the matching event index in hit_table
            #  Assumes that sequence_info_h[:,0] has only unique values
            #  Assumes that sequence_info_h[:,0] is a sorted subset of the sorted sequence_info_sp[:,0]
            found_h_index = False
            while h_index < len(sequence_info_h) and not found_h_index:
                if sequence_info_h[h_index, 0] == event_index:
                    found_h_index = True
                else:
                    h_index+=1
            if not found_h_index:
                raise IndexError(f'Reached the end of hit_table/hit_id without finding desired index from spacepoint_table/hit_id.\nAssumption of unique, sorted values may be wrong.')

            event_index_h, starting_index_h, num_points_h = sequence_info_h[h_index]

            # Get the primary particles associated with each event to add descriptive code
            primary_particles = g4_pdg[starting_index_pt + np.where(parent_id[starting_index_pt: starting_index_pt+num_points_pt] == 0)[0]]
            # primary_particles = g4_pdg_buffer[np.where(parent_id_buffer[np.array(list(range(starting_index_pt, starting_index_pt+num_points_pt)))] == 0)[0]]
            particle_description = -1 # Default description
            for particles, description in interaction_descriptions:
                is_matching = True
                remaining_particles = particles.copy()
                for particle in primary_particles:
                    if particle[0] in remaining_particles:
                        remaining_particles.remove(particle[0])
                if len(remaining_particles) == 0:
                    particle_description = description
                    break
            # if particle_description in [-1]:
            #     print(primary_particles.squeeze(-1))

            # Filter so only 2 photon events make it through
            descriptions_found[particle_description] += 1 
            if particle_description not in description_filter:
                skipped += 1
                continue
            # else:
            #     print(primary_particles.squeeze(-1))

            # The charge integral on y-plane associated with each hit from the current event
            #  Assumes the hit_ids in the hit_table are in order starting at zero
            integrals = hit_integral[starting_index_h+hit_ids][:,0] 
            planes = hit_plane[starting_index_h+hit_ids][:,0]
            # integrals = hit_integral_buffer[starting_index_h+hit_ids][:,0] 
            # planes = hit_plane_buffer[starting_index_h+hit_ids][:,0]

            # Get the first space point for each hit_id_y and add it to the new list
            included_hit_ids = []
            for k in range(len(hit_ids)):
                hit_id = hit_ids[k]
                if hit_id == -1:
                    continue

                if planes[k] != 2: # Confirmation that all hits found are from plane Y
                    print(event_index)
                    print(hit_ids)
                    print(planes)
                    raise ValueError(f'Calculating a hit integral for the wrong plane. Check your assumptions.')
                
                if hit_id not in included_hit_ids:
                    included_hit_ids.append(hit_id)

                    # Add point info to the new lists
                    x, y, z = sp_positions[starting_index+k,:]
                    # x, y, z = sp_positions_buffer[starting_index+k]
                    integral = integrals[k]
                    point_info_new.append([x,y,z,integral])

            # Update the starting_index and num_points with the reduced size
            if len(sequence_info_new) > 0:
                starting_index_new = sequence_info_new[-1][1] + sequence_info_new[-1][2]
            else: 
                starting_index_new = 0
            num_points_new = len(included_hit_ids)
            sequence_info_new.append([event_index, starting_index_new, num_points_new, run, subrun, event, particle_description])

            # Calculate the center of the points in the current event and subtract that from the positions to center them
            points_new = np.array(point_info_new[-num_points_new:])
            
            com = np.mean(points_new, axis=0)[0:3]
            points_centered_new = points_new[:,0:3] - com
            
            point_info_centered_new.extend(np.hstack((points_centered_new, points_new[:,3].reshape(-1,1))))

        # Convert to numpy arrays
        sequence_info_new = np.array(sequence_info_new)
        point_info_new = np.array(point_info_new)
        point_info_centered_new = np.array(point_info_centered_new)
        # print(sequence_info_new)
        # print(point_info_new)
        # print(point_info_centered_new)

        print(f'Descriptions: {descriptions_found}')
        print(f'Percentage Skipped: {skipped/event_cap*100}%')

    # Add the new lists to the new file
    with h5py.File(output_file_path, 'a') as f:
        print(f'Writing {label} to {output_file_path} ({len(sequence_info_new)} events)')

        if label in f.keys():
            del f[label]
        grp = f.create_group(label)

        grp.create_dataset('sequence_info_filtered', data=sequence_info_new)
        grp.create_dataset('sequence_info_all', data=sequence_info_sp)
        grp.create_dataset('point_info_filtered', data=point_info_new)
        grp.create_dataset('point_info_centered', data=point_info_centered_new)

        print(f'Created {label} with the datasets: {list(grp.keys())}')
                
