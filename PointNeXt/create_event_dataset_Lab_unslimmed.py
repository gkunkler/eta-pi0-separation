import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset 
import h5py 
from typing import Optional, Tuple, List
import os 
from tqdm import tqdm # Used for progress bar during preloading
import time # For internal timing in preloading (optional)


from openpoints.transforms import build_transforms_from_cfg


def calculate_e_p_angle(h5f_obj: h5py.File, rse_tuple: Tuple[int, int, int]) -> Optional[float]:
    """
    Calculates the angle between electron and positron corrected trajectories for a given RSE.
    Reads data directly from the provided h5f_obj.
    Returns the angle in degrees (0-180), or None if particles are not found/do not move.
    """
    electron_vector = None
    positron_vector = None

    try:
        particle_table_path = '/particle_table'
        if particle_table_path not in h5f_obj or \
           'event_id' not in h5f_obj[particle_table_path] or \
           'start_position_corr' not in h5f_obj[particle_table_path] or \
           'end_position_corr' not in h5f_obj[particle_table_path] or \
           'g4_pdg' not in h5f_obj[particle_table_path]:
            return None 
        
        # Read data directly from the h5f_obj passed
        particle_event_ids = h5f_obj[f'{particle_table_path}/event_id'][()] # (N_particles, 3)
        rse_matches_mask = np.all(particle_event_ids == np.array(rse_tuple, dtype=np.int32), axis=1)

        if not np.any(rse_matches_mask):
            return None 

        start_pos_corr_data = h5f_obj[f'{particle_table_path}/start_position_corr'][rse_matches_mask]
        end_pos_corr_data = h5f_obj[f'{particle_table_path}/end_position_corr'][rse_matches_mask]
        g4_pdg_data = h5f_obj[f'{particle_table_path}/g4_pdg'][rse_matches_mask]

        for i in range(len(g4_pdg_data)):
            pdg = g4_pdg_data[i].item()
            start_p = start_pos_corr_data[i]
            end_p = end_pos_corr_data[i]

            vector = end_p - start_p
            
            if np.linalg.norm(vector) < 1e-9: 
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
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0) 

        angle_radians = np.arccos(cosine_angle)
        angle_degrees = np.degrees(angle_radians)

        return angle_degrees
    except Exception as e:
        print(f"Error calculating angle for RSE {rse_tuple}: {e}") 
        return None 


#Takes opened h5f to avoid repeate opeming
def count_points_and_hits(h5f_obj: h5py.File, rse_tuple: Tuple[int, int, int]) -> Tuple[int, int]:
    n_pc, n_sp = 0, 0
    try:
        #Read data directly from the h5f_obj passed
        if '/spacepoint_table/event_id' in h5f_obj:
            spacepoint_event_ids = h5f_obj['/spacepoint_table/event_id'][()]
            n_pc = np.sum(np.all(spacepoint_event_ids == np.array(rse_tuple, dtype=np.int32), axis=1))

        if '/hit_table/event_id' in h5f_obj:
            hit_event_ids = h5f_obj['/hit_table/event_id'][()]
            n_sp = np.sum(np.all(hit_event_ids == np.array(rse_tuple, dtype=np.int32), axis=1))
    except Exception as e:
        print(f"Error counting points/hits for RSE {rse_tuple}: {e}")
        return 0, 0
    
    return int(n_pc), int(n_sp)


class EventPointCloudDataset(Dataset):
    def __init__(self, h5_file_path, num_points=2048, use_transforms=None, max_samples=None): 
        self.h5_file_path = h5_file_path
        self.num_points = num_points
        self.transforms = build_transforms_from_cfg(use_transforms) if use_transforms else None

        self.h5_file = None 
        self.hdf5_store = None 

        #Load event_metadata_df (from /event_table/event_id
        print("Loading event_ids from /event_table/event_id for initial event list...")
        with h5py.File(h5_file_path, 'r') as h5f_initial:
            if '/event_table/event_id' not in h5f_initial:
                raise KeyError("Dataset '/event_table/event_id' not found in input HDF5.")
            all_raw_event_ids = h5f_initial['/event_table/event_id'][()] # (N_total_events, 3)

        # Create event_metadata_df from event_ids
        self.event_metadata_df = pd.DataFrame(all_raw_event_ids, columns=['runNo', 'subRunNo', 'eventNo'])
        
        #chooses max samples
        if max_samples is not None and max_samples > 0 and max_samples < len(self.event_metadata_df):
            self.event_metadata_df = self.event_metadata_df.head(max_samples) 
            print(f"Dataset truncated to {max_samples} samples.") 
        
        self.num_samples = len(self.event_metadata_df) 
        print(f"Dataset initialized with {self.num_samples} events from {h5_file_path}.") 

        #Preload 'spacepoint_table/position' data and calcualte angle for optimality
        self.all_preloaded_pc_data = {} #Stores positions (x,y,z) and integral_y 4d
        self.all_preloaded_targets = {} #Stores calculated angle (target)
        self.all_preloaded_meta_n_counts = {} #Stores nPC, nSP per event for metadata

        rse_tuples_to_preload = []
        for index, row in self.event_metadata_df.iterrows():
            rse_tuple = (row['runNo'], row['subRunNo'], row['eventNo'])
            rse_tuples_to_preload.append(rse_tuple)
        
        print(f"Preloading {len(rse_tuples_to_preload)} sets of data and calculating angles/intensity into RAM...")
        
        # Open h5py.File
        with h5py.File(h5_file_path, 'r') as h5f_preload:
            # Check necessary spacepoint datasets
            if '/spacepoint_table/position' not in h5f_preload or '/spacepoint_table/event_id' not in h5f_preload:
                raise KeyError("Required '/spacepoint_table/position' or '/spacepoint_table/event_id' not found for preloading.")
            
            #Read full arrays to create masks only 1 ime
            all_spacepoint_event_ids = h5f_preload['/spacepoint_table/event_id'][()]
            all_spacepoint_positions = h5f_preload['/spacepoint_table/position'][()]
            all_spacepoint_hit_ids = h5f_preload['/spacepoint_table/hit_id'][()]
            
            # Read all hit_table data once for integral_y calculation  and check to make sure ot actually exists 9bunch of ands for bool output
            hit_table_exists = '/hit_table/event_id' in h5f_preload and '/hit_table/hit_id' in h5f_preload and \
                               '/hit_table/integral' in h5f_preload and '/hit_table/local_plane' in h5f_preload

            if not hit_table_exists:
                print("Warning: Missing hit_table data for intensity calculation. Integral feature will be 0.0.")
                # Provide empty arrays so no crashes
                all_hit_table_event_ids = np.array([])
                all_hit_table_hit_ids = np.array([])
                all_hit_table_integrals = np.array([])
                all_hit_table_planes = np.array([])
            else:
                all_hit_table_event_ids = h5f_preload['/hit_table/event_id'][()]
                all_hit_table_hit_ids = h5f_preload['/hit_table/hit_id'][()]
                all_hit_table_integrals = h5f_preload['/hit_table/integral'][()]
                all_hit_table_planes = h5f_preload['/hit_table/local_plane'][()]


            for rse_tuple in tqdm(rse_tuples_to_preload, desc="Preloading data & Calc. Angles/Features"):
                
                #Calculate angle for curr RSE
                calculated_angle = calculate_e_p_angle(h5f_preload, rse_tuple) # Pass h5f_preload\
                
                if calculated_angle is None:
                    self.all_preloaded_targets[rse_tuple] = 0.0 # Assign a default value for target if not found
                else:
                    self.all_preloaded_targets[rse_tuple] = calculated_angle 

                #Count nPC/nSP for the current RSE
                n_pc, n_sp = count_points_and_hits(h5f_preload, rse_tuple) # Pass h5f_preload, not path
                self.all_preloaded_meta_n_counts[rse_tuple] = {'nPC': n_pc, 'nSP': n_sp}

                #Get spacepoint position and integral
                rse_matches_mask_sp = np.all(all_spacepoint_event_ids == np.array(rse_tuple, dtype=np.int32), axis=1)
                
                if np.any(rse_matches_mask_sp):
                    filtered_sp_positions = all_spacepoint_positions[rse_matches_mask_sp]
                    filtered_sp_hit_ids = all_spacepoint_hit_ids[rse_matches_mask_sp] # For hit_id lookup

                    #Prep hit id for integral lookup
                    integral_y_values = np.zeros(len(filtered_sp_positions), dtype=np.float32) # Default to 0.0
                    
                    if hit_table_exists and np.any(np.all(all_hit_table_event_ids == np.array(rse_tuple, dtype=np.int32), axis=1)): 
                        # Filter hit data for the current RSE
                        rse_matches_mask_hit_current = np.all(all_hit_table_event_ids == np.array(rse_tuple, dtype=np.int32), axis=1)
                        filtered_hit_ids_current = all_hit_table_hit_ids[rse_matches_mask_hit_current].flatten()
                        filtered_hit_integrals_current = all_hit_table_integrals[rse_matches_mask_hit_current].flatten()
                        filtered_hit_planes_current = all_hit_table_planes[rse_matches_mask_hit_current].flatten()

                        df_hits_in_rse = pd.DataFrame({
                            'hit_id': filtered_hit_ids_current,
                            'integral': filtered_hit_integrals_current,
                            'local_plane': filtered_hit_planes_current
                        })
                        
                        hit_id_to_integral_map = df_hits_in_rse[df_hits_in_rse['local_plane'] == 2].set_index('hit_id')['integral'].to_dict()
                        
                        integral_y_values = np.array([
                            hit_id_to_integral_map.get(hit_id_arr[2], 0.0) # Index 2 for plane 2 hit_id (y)
                            for hit_id_arr in filtered_sp_hit_ids
                        ], dtype=np.float32)
                    else:
                        pass # integral_y_values remains 0.0 if no hits or table missing for this event
                    
                    self.all_preloaded_pc_data[rse_tuple] = {
                        'x': filtered_sp_positions[:, 0].copy(), 
                        'y': filtered_sp_positions[:, 1].copy(),
                        'z': filtered_sp_positions[:, 2].copy(),
                        'integral_y': integral_y_values.copy()
                    }
                else: # No space points for this RSE
                    self.all_preloaded_pc_data[rse_tuple] = {
                        'x': np.array([]), 'y': np.array([]), 'z': np.array([]), 'integral_y': np.array([])
                    }
                    print(f"Warning: No space points found for RSE {rse_tuple}. Storing empty arrays.")
        
        print(f"Finished preloading {len(self.all_preloaded_pc_data)} PC event data sets and angles/features into RAM.")


    def __len__(self):
        return self.num_samples

    #Not necessary anymore since preloading
    @staticmethod
    def worker_init_fn(worker_id):
        pass 
    
    def __getitem__(self, indx):
        #Access the already preloaded metadata and data directly from RAM
        event_meta_row = self.event_metadata_df.iloc[indx] 
        rse_tuple = (event_meta_row['runNo'], event_meta_row['subRunNo'], event_meta_row['eventNo'])

        # Retrieve points and target directly from preloaded dictionaries
        pc_data_preloaded = self.all_preloaded_pc_data[rse_tuple]
        opang_preloaded = self.all_preloaded_targets[rse_tuple] 
        
        x_coords = pc_data_preloaded["x"]
        y_coords = pc_data_preloaded["y"]
        z_coords = pc_data_preloaded["z"] 
        integral_y_coords = pc_data_preloaded["integral_y"]

        #tack XYZ and integral_y into features tensor
        points_xyz = np.stack([x_coords, y_coords, z_coords], axis=-1) 
        features = np.stack([x_coords, y_coords, z_coords, integral_y_coords], axis=-1) # NEW: Features now include integral_y

        N_original = points_xyz.shape[0]
        if N_original > self.num_points:
            sample_idx = np.random.choice(N_original, self.num_points, replace=False)
            points_xyz = points_xyz[sample_idx]
            features = features[sample_idx]
        elif N_original < self.num_points:
            padded_points_xyz = np.zeros((self.num_points, points_xyz.shape[1]), dtype=points_xyz.dtype)
            padded_features = np.zeros((self.num_points, features.shape[1]), dtype=features.dtype) # Padded features has 4 channels
            padded_points_xyz[:N_original] = points_xyz
            padded_features[:N_original] = features
            points_xyz = padded_points_xyz
            features = padded_features

        pos_tensor = torch.from_numpy(points_xyz).float()
        feat_tensor = torch.from_numpy(features).float() 
        feat_tensor = feat_tensor.transpose(0, 1).contiguous() # feature tensor is now (4, N)
        
        target_tensor = torch.tensor(opang_preloaded, dtype=torch.float) 
        
        if self.transforms:
            data_dict_for_transform = {'pos': pos_tensor, 'x': feat_tensor, 'y': target_tensor}
            data_dict_for_transform = self.transforms(data_dict_for_transform)
            pos_tensor = data_dict_for_transform['pos']
            feat_tensor = data_dict_for_transform['x']
            target_tensor = data_dict_for_transform['y']

        data_dict = {'pos': pos_tensor, 'x': feat_tensor} 
        
        return data_dict, target_tensor