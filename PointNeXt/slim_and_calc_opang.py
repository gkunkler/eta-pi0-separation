import h5py
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from typing import Optional, Tuple, List
import sys 


ORIGINAL_MERGED_H5_PATH = "merged_EpEm.h5" #Input file from the new lab
NEW_LEAN_INDEXED_H5_PATH = "lean_merged_EpEm_indexed.h5" #Output file name
PROGRESS_LOG_FILE = "slimming_merged_progress.log" #Log file


#Return angle between electron and positron 
def calculate_e_p_angle(h5_file_path: str, rse_tuple: Tuple[int, int, int]) -> Optional[float]:
    electron_vector = None
    positron_vector = None

    try:
        with h5py.File(h5_file_path, 'r') as h5f: #Open h5py.File
            particle_table_path = '/particle_table'
            if particle_table_path not in h5f or \
               'event_id' not in h5f[particle_table_path] or \
               'start_position_corr' not in h5f[particle_table_path] or \
               'end_position_corr' not in h5f[particle_table_path] or \
               'g4_pdg' not in h5f[particle_table_path]:
                return None 
            
            particle_event_ids = h5f[f'{particle_table_path}/event_id'][()] # (N_particles, 3)
            rse_matches_mask = np.all(particle_event_ids == np.array(rse_tuple, dtype=np.int32), axis=1)

            if not np.any(rse_matches_mask):
                return None 

            start_pos_corr_data = h5f[f'{particle_table_path}/start_position_corr'][rse_matches_mask]
            end_pos_corr_data = h5f[f'{particle_table_path}/end_position_corr'][rse_matches_mask]
            g4_pdg_data = h5f[f'{particle_table_path}/g4_pdg'][rse_matches_mask]

            for i in range(len(g4_pdg_data)):
                pdg = g4_pdg_data[i].item() # .item() to get scalar
                start_p = start_pos_corr_data[i]
                end_p = end_pos_corr_data[i]

                vector = end_p - start_p
                
                if np.linalg.norm(vector) < 1e-9: #Check for near-zero magnitude
                    continue 

                if pdg == -11 and electron_vector is None:
                    electron_vector = vector
                elif pdg == 11 and positron_vector is None:
                    positron_vector = vector
                
                if electron_vector is not None and positron_vector is not None:
                    break 

        if electron_vector is None or positron_vector is None:
            return None 

        dot_product = np.dot(electron_vector, positron_vector)
        magnitude_e = np.linalg.norm(electron_vector)
        magnitude_p = np.linalg.norm(positron_vector)

        if magnitude_e == 0 or magnitude_p == 0: 
            return None 

        cosine_angle = dot_product / (magnitude_e * magnitude_p)
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0) #Clip

        angle_radians = np.arccos(cosine_angle)
        angle_degrees = np.degrees(angle_radians)

        return angle_degrees
    except Exception as e:
        print(f"Error calculating angle for RSE {rse_tuple}: {e}") 
        return None 


#Function to count points/hits for an event ---
def count_points_and_hits(h5_file_path: str, rse_tuple: Tuple[int, int, int]) -> Tuple[int, int]:
    #Counts the number of space points (nPC) and hits (nSP) for a given RSE
    n_pc, n_sp = 0, 0
    try:
        with h5py.File(h5_file_path, 'r') as h5f:
            # Count space points
            if '/spacepoint_table/event_id' in h5f:
                spacepoint_event_ids = h5f['/spacepoint_table/event_id'][()]
                n_pc = np.sum(np.all(spacepoint_event_ids == np.array(rse_tuple, dtype=np.int32), axis=1))

            # Count hits
            if '/hit_table/event_id' in h5f:
                hit_event_ids = h5f['/hit_table/event_id'][()]
                n_sp = np.sum(np.all(hit_event_ids == np.array(rse_tuple, dtype=np.int32), axis=1))
    except Exception as e:
        print(f"Error counting points/hits for RSE {rse_tuple}: {e}")
        return 0, 0
    
    return int(n_pc), int(n_sp)


def create_lean_indexed_hdf5(
    input_h5_path: str,
    output_h5_path: str,
    max_events_to_process: Optional[int] = None, 
    chunk_size: int = 1000 #Num of events to process per chunk for resumability
):
    """
    Creates a new, smaller HDF5 file
	Event Level Data: ('runNo', 'subRunNo', 'eventNo', :opang_calulated', 'nPC', 'nSP'
	Point cloud level data: ('runNo', 'subRunNo', 'eventNO', 'x', 'y', 'z', 'integral_y')
	Allows for fast pd.HDFStore.seelect queries
    Supports resuming progress.
    """
    if not os.path.exists(input_h5_path):
        print(f"Error: Input HDF5 file '{input_h5_path}' not found.")
        return

    print(f"Start creation of lean HDF5 file: '{output_h5_path}' from '{input_h5_path}'")

    try:
        # Loads event_ids from /event_table/event_id for the list of all events
        print("Loading event_ids from /event_table/event_id for initial event list...")
        with h5py.File(input_h5_path, 'r') as h5f_initial:
            if '/event_table/event_id' not in h5f_initial:
                print("Error: Dataset '/event_table/event_id' not found in input HDF5. Cannot get event list.")
                return
            all_raw_event_ids = h5f_initial['/event_table/event_id'][()] # (N_total_events, 3)

        # Create a base DataFrame for all events to process
        event_metadata_base_df = pd.DataFrame(all_raw_event_ids, columns=['runNo', 'subRunNo', 'eventNo'])
        
        # Applys max event process filter (not necessary fi doing the whole file)
        if max_events_to_process is not None and max_events_to_process > 0 and max_events_to_process < len(event_metadata_base_df):
            event_metadata_base_df = event_metadata_base_df.head(max_events_to_process)
            print(f"Total processing limited to first {max_events_to_process} events.")
        """
		Group: /event_table/
  			Dataset: /event_table/event_id
				Shape: (24450, 3)  #tuple of runNo subRunNo and eventNo
				Dtype: int32
		"""
        all_rse_tuples = [
            (row['runNo'], row['subRunNo'], row['eventNo'])
            for index, row in event_metadata_base_df.iterrows()
        ]
        total_rse_to_process = len(all_rse_tuples)
        print(f"Identified {total_rse_to_process} events to process.")

        #Resumability (check log to see if resumable progess/ which chunks have been loaded )
        processed_rse_count = 0
        if os.path.exists(PROGRESS_LOG_FILE):
            with open(PROGRESS_LOG_FILE, 'r') as f:
                try:
                    #Reads the index of the last event processed
                    last_processed_event_idx = int(f.read().strip())
                    #restarts from the beginning of the next chunk
                    processed_rse_count = (last_processed_event_idx // chunk_size) * chunk_size 
                    print(f"Resuming from event index {processed_rse_count}.")
                except ValueError:
                    print("Progress log corrupted, starting from beginning.")
                    processed_rse_count = 0
        else:
            print("No previous progress found, starting from beginning.")

        #Writes meta data to new Hdf5 file 
        if processed_rse_count == 0:
            # Calculate all 'calculated_opang', 'nPC', 'nSP' for all events upfront.
            print("Pre-calculating angles and counting points for all events to be processed...")
            calculated_opangs = []
            n_pcs = [] 
            n_sps = [] 

            with h5py.File(input_h5_path, 'r') as h5f_data_prep:
                for rse_tuple in tqdm(all_rse_tuples, desc="Pre-calculating Angles & Point Counts"):
                    angle = calculate_e_p_angle(input_h5_path, rse_tuple)
                    calculated_opangs.append(angle if angle is not None else np.nan) # Store NaN if not found

                    n_pc, n_sp = count_points_and_hits(input_h5_path, rse_tuple)
                    n_pcs.append(n_pc)
                    n_sps.append(n_sp)

            #Add calculated columns to the event_metadata_base_df
            event_metadata_base_df['opang_calculated'] = calculated_opangs
            event_metadata_base_df['nPC'] = n_pcs
            event_metadata_base_df['nSP'] = n_sps

            # Save the prepared event_metadata_base_df to the new file in write mode
            with pd.HDFStore(output_h5_path, 'w', complevel=9, complib='zlib') as output_store_initial:
                output_store_initial.put(
                    'event_metadata',
                    event_metadata_base_df, 
                    format='table', 
                    data_columns=['runNo', 'subRunNo', 'eventNo', 'opang_calculated', 'nPC', 'nSP'], # Index all used columns (change for use)
                    index=False
                )
                # Create an empty pc_points table to ensure it's there for appending
                output_store_initial.put('pc_points', pd.DataFrame(columns=['runNo', 'subRunNo', 'eventNo', 'x', 'y', 'z', 'integral_y']),
                                         format='table', data_columns=['runNo', 'subRunNo', 'eventNo', 'x', 'y', 'z', 'integral_y'],
                                         index=False, append=False)
            print(f"Initialized new output HDF5 file: '{output_h5_path}' with event_metadata and empty pc_points.")
        else:
            print(f"Appending to existing output HDF5 file: '{output_h5_path}'. (event_metadata assumed to be already present)")

        #Process spacepoint data in chunks and append
        print(f"Processing spacepoint data in chunk of {chunk_size} events and appending")
        
        #Open the input HDF5 file (h5py)
        with h5py.File(input_h5_path, 'r') as h5f_input:
            # Read spacepoint data for the current chunk using h5py boolean indexing
            spacepoint_event_ids = h5f_input['/spacepoint_table/event_id'][()]
            spacepoint_positions = h5f_input['/spacepoint_table/position'][()]
            spacepoint_hit_ids = h5f_input['/spacepoint_table/hit_id'][()]
            hit_table_event_ids = h5f_input['/hit_table/event_id'][()]
            hit_table_hit_ids = h5f_input['/hit_table/hit_id'][()]
            hit_table_integrals = h5f_input['/hit_table/integral'][()]
            hit_table_planes = h5f_input['/hit_table/local_plane'][()]
                    			
            # Open in append mode
            with pd.HDFStore(output_h5_path, 'a', complevel=9, complib='zlib') as output_store_append:
                
                # Iterate through event RSEs in chunks
                for i in tqdm(range(processed_rse_count, total_rse_to_process, chunk_size), 
                              desc="Processing Space Point Chunks", initial=processed_rse_count // chunk_size):
                    
                    chunk_rse_tuples = all_rse_tuples[i : i + chunk_size]
                    

                    # Create a mask for all spacepoints that belong to RSE in the current chunk
                    mask_for_chunk = np.isin(
                        [tuple(row) for row in spacepoint_event_ids], 
                        chunk_rse_tuples
                    )
                    mask_for_chunk_hit = np.isin(
                        [tuple(row) for row in hit_table_event_ids], 
                        chunk_rse_tuples
                    )
                    
                    df_pc_points_chunk = pd.DataFrame() #init as empty
                    if np.any(mask_for_chunk):
                        filtered_event_ids = spacepoint_event_ids[mask_for_chunk]
                        filtered_positions = spacepoint_positions[mask_for_chunk]
                        filtered_sp_hit_ids = spacepoint_hit_ids[mask_for_chunk_sp]
                        filtered_hit_ids = hit_table_hit_ids[mask_for_chunk_hit].flatten()
                        filtered_hit_integrals = hit_table_integrals[mask_for_chunk_hit].flatten()
                        filtered_hit_planes = hit_table_planes[mask_for_chunk_hit].flatten()

                        df_hits_in_chunk = pd.DataFrame({
                            'hit_id': filtered_hit_ids,
                            'integral': filtered_hit_integrals,
                            'local_plane': filtered_hit_planes
                        })
                        
                        hit_id_to_integral_map = df_hits_in_chunk[df_hits_in_chunk['local_plane'] == 2].set_index('hit_id')['integral'].to_dict()
                        
                        df_pc_points_chunk = pd.DataFrame(filtered_positions, columns=['x', 'y', 'z'])
                        # Re-add RSE columns to the points DataFrame for indexing in the new file
                        df_pc_points_chunk['runNo'] = [r[0] for r in filtered_event_ids]
                        df_pc_points_chunk['subRunNo'] = [r[1] for r in filtered_event_ids]
                        df_pc_points_chunk['eventNo'] = [r[2] for r in filtered_event_ids]
                        df_pc_points_chunk['integral_y'] = [
                            hit_id_to_integral_map.get(hit_id_arr[2], 0.0)
                            for hit_id_arr in filtered_sp_hit_ids
                        ]
                        
                    # Append current chunk to the 'pc_points' table in the output file (puts data into new file)
                    output_store_append.append(
                        'pc_points',
                        df_pc_points_chunk,
                        format='table',
                        data_columns=['runNo', 'subRunNo', 'eventNo', 'x', 'y', 'z', 'integral_y'], # Index these columns
                        index=False
                    )
                    
                    #Log progress
                    with open(PROGRESS_LOG_FILE, 'w') as f_progress:
                        f_progress.write(str(i + chunk_size)) 
                
        print(f"\nSuccessfully created/updated lean HDF5 file at '{output_h5_path}'.")
        print(f"Final file size: {os.path.getsize(output_h5_path) / (1024**2):.2f} MB")

        #Clean log file
        if os.path.exists(PROGRESS_LOG_FILE):
            os.remove(PROGRESS_LOG_FILE)
            print("Progress log cleaned up.")

    except KeyError as e:
		#Error if can't find expected table in HDF5 fiel
        print(f"Error: Missing expected table or column in input HDF5: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during HDF5 creation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    original_h5_path = ORIGINAL_MERGED_H5_PATH
    new_lean_indexed_h5_path = NEW_LEAN_INDEXED_H5_PATH

    limit_events_for_conversion = None
    conversion_chunk_size = 1000 

    create_lean_indexed_hdf5(
        original_h5_path, new_lean_indexed_h5_path, 
        max_events_to_process=limit_events_for_conversion,
        chunk_size=conversion_chunk_size
    )
