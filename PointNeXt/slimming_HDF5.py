import h5py
import pandas as pd
import numpy as np
import os
from tqdm import tqdm # For progress bar during data processing

def create_lean_indexed_hdf5(
    input_h5_path: str,
    output_h5_path: str,
    max_events_to_process: Optional[int] = None # Limits total events to read from original file
):
    """
    Creates a new, smaller HDF5 file containing only the 'event_metadata' and
    'pc_points' data required by the model, saved in an indexed HDFStore format.

    Args:
        input_h5_path: Path to the large original HDF5 file (e.g., 'epem_sample_restructured_chunked.h5').
        output_h5_path: Path for the new, smaller HDF5 file (e.g., 'lean_epem_data_indexed.h5').
        max_events_to_process: Optional. If set, processes only the first N events from
                               the input file's event_metadata table. If None, processes all events.
    """
    if not os.path.exists(input_h5_path):
        print(f"Error: Input HDF5 file '{input_h5_path}' not found.")
        return

    print(f"Starting to create lean HDF5 file: '{output_h5_path}' from '{input_h5_path}'")

    try:
        with pd.HDFStore(input_h5_path, 'r') as input_store:
            # 1. Load relevant event_metadata (and filter by max_events_to_process)
            print("Loading event_metadata from original HDF5...")
            # Select only the columns needed for event identification and target (opang)
            # You might need to add 'nPC', 'nSP' if your code relies on them later
            event_metadata_cols = ['runNo', 'subRunNo', 'eventNo', 'opang', 'nPC', 'nSP'] 
            event_metadata_df = input_store.select('event_metadata', columns=event_metadata_cols)

            if max_events_to_process is not None and max_events_to_process > 0 and max_events_to_process < len(event_metadata_df):
                event_metadata_df = event_metadata_df.head(max_events_to_process)
                print(f"Processing limited to first {max_events_to_process} events from metadata.")
            
            # Get RSEs (Run, SubRun, Event) for the selected events
            rse_tuples_to_process = [
                (row['runNo'], row['subRunNo'], row['eventNo'])
                for index, row in tqdm(event_metadata_df.iterrows(), total=len(event_metadata_df), desc="Collecting RSEs to process")
            ]
            
            # 2. Load pc_points data for ONLY the selected events (and only necessary columns)
            print(f"Loading pc_points data for {len(rse_tuples_to_process)} selected events (this will be slow if original is not indexed)...")
            all_pc_points_data_list = []
            
            # Select only columns needed for pc_points: runNo, subRunNo, eventNo, x, y, z
            pc_points_cols = ['runNo', 'subRunNo', 'eventNo', 'x', 'y', 'z']

            for rse_tuple in tqdm(rse_tuples_to_process, desc="Reading pc_points data"):
                run_no, sub_run_no, event_no = rse_tuple
                
                # Perform the select query for each event
                pc_data_for_event = input_store.select(
                    'pc_points', 
                    where=f"runNo == {run_no} and subRunNo == {sub_run_no} and eventNo == {event_no}",
                    columns=pc_points_cols # Select only necessary columns to reduce memory if original has more
                )
                all_pc_points_data_list.append(pc_data_for_event)
            
            # Concatenate all collected DataFrames into one large DataFrame
            df_pc_points_lean = pd.concat(all_pc_points_data_list, ignore_index=True)
            print(f"Collected {len(df_pc_points_lean)} total points for {len(rse_tuples_to_process)} events.")


        # 3. Save to New HDF5 File (Crucially, with indexing for efficiency)
        print(f"Saving collected data to '{output_h5_path}'...")
        # Use 'w' mode to create/overwrite the new file
        # complevel and complib are for compression (recommended for large files)
        with pd.HDFStore(output_h5_path, 'w', complevel=9, complib='zlib') as output_store: 
            # Save event_metadata table with indexing
            output_store.put(
                'event_metadata', 
                event_metadata_df, 
                format='table', 
                # --- CRITICAL: data_columns=True for all columns you'll query by ---
                data_columns=['runNo', 'subRunNo', 'eventNo', 'opang'], # Index all columns you'll select or filter by
                index=False # No pandas index written to HDF5
            )
            print("event_metadata saved to new file.")

            # Save pc_points table with indexing
            output_store.put(
                'pc_points', 
                df_pc_points_lean, 
                format='table', 
                # --- CRITICAL: data_columns=True for all columns you'll query by ---
                data_columns=['runNo', 'subRunNo', 'eventNo', 'x', 'y', 'z'], 
                index=False
            )
            print("pc_points saved to new file.")
        
        print(f"Successfully created lean, indexed HDF5 file at '{output_h5_path}'.")
        print(f"New file size: {os.path.getsize(output_h5_path) / (1024**2):.2f} MB")

    except KeyError as e:
        print(f"Error: Missing expected table or column in input HDF5: {e}")
        print("Please verify the structure of your input HDF5 file (e.g., 'event_metadata', 'pc_points').")
    except Exception as e:
        print(f"An unexpected error occurred during HDF5 creation: {e}")
        import traceback
        traceback.print_exc()

# --- Example Usage (Main Execution Block) ---
if __name__ == "__main__":
    # --- IMPORTANT: Adjust these paths ---
    original_h5_path = "epem_sample_restructured_chunked.h5" # Your 17GB file
    new_lean_indexed_h5_path = "lean_epem_data_indexed.h5" # Name for the new smaller, indexed file

    # Set this to a small number (e.g., 1000) for initial testing/debugging the conversion itself.
    # Set to None to process all events (will be slow if original is not indexed, but runs once).
    limit_events_for_conversion = None # Set to None for full 48877 events, or 1000 for quick test

    create_lean_indexed_hdf5(original_h5_path, new_lean_indexed_h5_path, 
                             max_events_to_process=limit_events_for_conversion)

    # --- AFTER THIS SCRIPT RUNS SUCCESSFULLY ---
    # 1. Update your my_YAML.yaml file:
    #    dataset:
    #      common:
    #        h5_file_path: 'lean_epem_data_indexed.h5' # <-- Point to your new file
    #        max_samples: null # <-- IMPORTANT: Remove or set to null/None if you want to use ALL events from the NEW lean file
    #                              (which would be max_events_to_process events in total).
    # 2. Run your training script as usual.