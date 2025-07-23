import pandas as pd
import numpy as np
import os
import sys
import gc 


pkl_file_path = "epem_sample.pkl"
hdf5_output_path = "epem_sample_restructured_chunked.h5" # New HDF5 file name for this method

#Columns in the original DataFrame that contain arrays/lists (point cloud data)
POINT_CLOUD_COLS = ['pc_x', 'pc_y', 'pc_z']
SCINTILLATION_POINT_COLS = ['sp_x', 'sp_y', 'sp_z', 'sp_q']

#Columns in the original DataFrame that might contain problematic mixed/object dtypes
PROBLEMATIC_METADATA_COLS = ['minmax_sp_q', 'robust_sp_q', 'std_sp_q'] # <--- ADDED 'std_sp_q'

#Define the batch size for processing events
BATCH_SIZE = 5000 # Process 5000 events at a time

if __name__ == "__main__":
    print(f"[{os.getpid()}] Starting memory-optimized conversion from '{pkl_file_path}' to '{hdf5_output_path}'...")

    try:
        #Load pkl file
        print(f"[{os.getpid()}] Loading PKL file '{pkl_file_path}' into memory. This may still take a while for 5GB...")
        large_df = pd.read_pickle(pkl_file_path)
        print(f"[{os.getpid()}] PKL file loaded successfully. Original DataFrame shape: {large_df.shape}")

        #Identify all columns that contain point data or are problematic object types
        all_cols_to_flatten_or_drop = POINT_CLOUD_COLS + SCINTILLATION_POINT_COLS

        event_metadata_df = large_df.drop(columns=all_cols_to_flatten_or_drop, errors='ignore')
        print(f"[{os.getpid()}] Extracted event metadata. Shape: {event_metadata_df.shape}")
--
        print(f"[{os.getpid()}] Checking and converting problematic metadata columns to string type...")
        for col in PROBLEMATIC_METADATA_COLS:
            if col in event_metadata_df.columns:
                original_dtype = event_metadata_df[col].dtype
                if original_dtype == 'object':
                    event_metadata_df[col] = event_metadata_df[col].astype(str)
                    print(f"  Converted column '{col}' from '{original_dtype}' to '{event_metadata_df[col].dtype}' (string).")
                else:
                    print(f"  Column '{col}' is already of type '{original_dtype}' (not object), skipping conversion.")
            else:
                print(f"  Warning: Problematic column '{col}' not found in event_metadata_df. Skipping.")
       
        with pd.HDFStore(hdf5_output_path, mode='w') as store:
            store.put('event_metadata', event_metadata_df, format='table',
                      data_columns=['runNo', 'subRunNo', 'eventNo']) 
            print(f"[{os.getpid()}] Saved 'event_metadata' table.")

        total_events = len(large_df)
        print(f"[{os.getpid()}] Starting batch processing for point data (Batch Size: {BATCH_SIZE})...")

        #Open HDFStore in append mode for point tables
        with pd.HDFStore(hdf5_output_path, mode='a') as store:
            for i in range(0, total_events, BATCH_SIZE):
                batch_start = i
                batch_end = min(i + BATCH_SIZE, total_events)
                current_batch_df = large_df.iloc[batch_start:batch_end]

                print(f"[{os.getpid()}] Processing batch {batch_start}-{batch_end} of {total_events} events...")

                batch_pc_data = []
                batch_sp_data = []

                for idx, row in current_batch_df.iterrows():
                    run_no = row['runNo']
                    sub_run_no = row['subRunNo']
                    event_no = row['eventNo']
                    opang = row.get('opang')
                    e_E = row.get('E1')
                    p_E = row.get('E2')

                    #Process Point Cloud data for this batch
                    pc_x = row.get('pc_x')
                    pc_y = row.get('pc_y')
                    pc_z = row.get('pc_z')
                    if isinstance(pc_x, (list, np.ndarray)) and len(pc_x) > 0:
                        for p_idx in range(len(pc_x)):
                            batch_pc_data.append({
                                'runNo': run_no, 'subRunNo': sub_run_no, 'eventNo': event_no,
                                'point_idx': p_idx, 'x': pc_x[p_idx], 'y': pc_y[p_idx], 'z': pc_z[p_idx],
                                'opang': opang, 'e_E': e_E, 'p_E': p_E
                            })

                    #Process Scintillation Point (sp) data for this batch
                    sp_x = row.get('sp_x')
                    sp_y = row.get('sp_y')
                    sp_z = row.get('sp_z')
                    sp_q = row.get('sp_q')
                    if isinstance(sp_x, (list, np.ndarray)) and len(sp_x) > 0:
                        for p_idx in range(len(sp_x)):
                            batch_sp_data.append({
                                'runNo': run_no, 'subRunNo': sub_run_no, 'eventNo': event_no,
                                'point_idx': p_idx, 'x': sp_x[p_idx], 'y': sp_y[p_idx], 'z': sp_z[p_idx],
                                'q': sp_q[p_idx], 'opang': opang, 'e_E': e_E, 'p_E': p_E
                            })

                #Convert batch data to DataFrames
                pc_df_batch = pd.DataFrame(batch_pc_data)
                sp_df_batch = pd.DataFrame(batch_sp_data)

                #Append batch DataFrames to HDF5 tables
                store.put('pc_points', pc_df_batch, format='table', append=True,
                          data_columns=['runNo', 'subRunNo', 'eventNo', 'point_idx']) # Use data_columns for filtering later
                store.put('sp_points', sp_df_batch, format='table', append=True,
                          data_columns=['runNo', 'subRunNo', 'eventNo', 'point_idx']) # Use data_columns for filtering later

                print(f"[{os.getpid()}] Saved batch {batch_start}-{batch_end} to HDF5.")

                #Explicitly delete batch data and force garbage collection
                del current_batch_df
                del batch_pc_data
                del batch_sp_data
                del pc_df_batch
                del sp_df_batch
                gc.collect()
                print(f"[{os.getpid()}] Cleared memory for batch {batch_start}-{batch_end}.")

        del large_df
        gc.collect()
        print(f"[{os.getpid()}] Original DataFrame cleared from memory.")

        print(f"[{os.getpid()}] Conversion complete! Restructured HDF5 file saved to '{hdf5_output_path}'")

    except FileNotFoundError:
        print(f"[{os.getpid()}] Error: The PKL file '{pkl_file_path}' was not found. Please ensure it exists.")
        sys.exit(1)
    except Exception as e:
        print(f"[{os.getpid()}] An unexpected and critical error occurred during conversion: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)