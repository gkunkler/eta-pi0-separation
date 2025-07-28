import torch
import math
import numpy as np
import pandas as pd
from torch_geometric.nn import fps, radius
from torch_cluster import knn_graph
import plotly
import plotly.graph_objects as go
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import h5py
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
from typing import Union, Tuple, List, Optional
import os 

#from create_event_dataset import EventPointCloudDataset DONT NEED

HDF5_FILE_PATH = "epem_sample_restructured_chunked.h5"
#HDF5_FILE_PATH = "NeutrinoML_TN_ts818547.h5"

#Get event_id based on runNo, subRunNo, eventNo
def get_event_id_from_rse(event_metadata_df: pd.DataFrame, rse: List[int]) -> Optional[int]:

    result = event_metadata_df.loc[(event_metadata_df["runNo"] == rse[0]) &
                                   (event_metadata_df["subRunNo"] == rse[1]) &
                                   (event_metadata_df["eventNo"] == rse[2])]
    if not result.empty:
        return result.index[0] 
    return None

#Modified wc_coords to work with HDF5 tables
def wc_coords_hdf5(
    hdf5_store: pd.HDFStore,
    event_id: int,
    coord_type: str = "pc"
):

    #Retrieves point cloud or scintillation point coordinates for a given event_id from the HDF5 store.
    event_meta = hdf5_store.select('event_metadata', where=f"index == {event_id}").iloc[0]

    run_no = event_meta['runNo']
    sub_run_no = event_meta['subRunNo']
    event_no = event_meta['eventNo']
    opang = event_meta.get('opang') 

    if coord_type == "pc":
    
        pc_data = hdf5_store.select('pc_points', where=f"runNo == {run_no} and subRunNo == {sub_run_no} and eventNo == {event_no}")
        x = pc_data["x"].values
        y = pc_data["y"].values
        z = pc_data["z"].values
        num_pts = len(x)
        return x, y, z, num_pts, opang

    elif coord_type == "sp":
        
        sp_data = hdf5_store.select('sp_points', where=f"runNo == {run_no} and subRunNo == {sub_run_no} and eventNo == {event_no}")
        x = sp_data["x"].values
        y = sp_data["y"].values
        z = sp_data["z"].values
        q = sp_data["q"].values
        num_pts = len(x)
        return x, y, z, q, num_pts, opang
    else:
        raise ValueError("Invalid coord_type. Must be 'pc' or 'sp'.")


# Modified epem_info to work with HDF5 tables
def epem_info_hdf5(hdf5_store: pd.HDFStore, event_id: int):
    
    #Retrieves electron-positron information for a given event_id from the HDF5 store.
    event_meta = hdf5_store.select('event_metadata', where=f"index == {event_id}").iloc[0]

    e_x = event_meta["x1"]
    e_y = event_meta["y1"]
    e_z = event_meta["z1"]
    e_pos = [e_x, e_y, e_z]

    e_px = event_meta["p1x"]
    e_py = event_meta["p1y"]
    e_pz = event_meta["p1z"]
    e_E = event_meta["E1"]
    e_mom = [e_px, e_py, e_pz, e_E]

    p_x = event_meta["x2"]
    p_y = event_meta["y2"]
    p_z = event_meta["z2"]
    p_pos = [p_x, p_y, p_z]

    p_px = event_meta["p2x"]
    p_py = event_meta["p2y"]
    p_pz = event_meta["p2z"]
    p_E = event_meta["E2"]
    p_mom = [p_px, p_py, p_pz, p_E]

    return e_pos, p_pos, e_mom, p_mom


# The god function to view an event in 3D interactive!
def view_event3d_hdf5(
    hdf5_file_path: str, # Pass the path to the HDF5 file
    index_or_rse: Union[int, List[int]], # Can be int or [run, subrun, event]
    coord_type: str = "all",
    len1: int = 100,
    len2: int = 500,
    figsize: List[int] = [800, 800],
    html_name: Optional[str] = None
):
    """
    Args:
        hdf5_file_path: Path to the HDF5 file containing the event data.
        index_or_rse: Can be int (row index in event_metadata) or [runNo, subRunNo, eventNo].
        coord_type: "pc", "sp", or "all". If "sp" or "all", will plot charge in cbar.
        len1: Length for electron momentum arrow.
        len2: Length for positron momentum arrow.
        figsize: Figure size [width, height].
        html_name: If provided, saves the plot as an HTML file.

    Returns: NA
        Displays a plotly figure.
    """
    with pd.HDFStore(hdf5_file_path, 'r') as store:
        event_metadata_df = store.select('event_metadata', columns=['runNo', 'subRunNo', 'eventNo', 'nPC', 'nSP', 'opang', 'x1', 'y1', 'z1', 'p1x', 'p1y', 'p1z', 'E1', 'x2', 'y2', 'z2', 'p2x', 'p2y', 'p2z', 'E2'])

        event_id = None
        if isinstance(index_or_rse, int):
            event_id = index_or_rse
            if event_id >= len(event_metadata_df):
                print(f"Error: Integer index {event_id} out of bounds for {len(event_metadata_df)} events.")
                return
        elif isinstance(index_or_rse, list) and len(index_or_rse) == 3:
            event_id = get_event_id_from_rse(event_metadata_df, index_or_rse)
            if event_id is None:
                print(f"Error: RSE {index_or_rse} not found in event metadata.")
                return
        else:
            print("Not a valid index or RSE format. Must be int or [run, subrun, event].")
            return None

        # Get event-specific metadata for the title and true particle info
        current_event_meta = event_metadata_df.iloc[event_id]
        run_no_val = current_event_meta['runNo']
        sub_run_no_val = current_event_meta['subRunNo']
        event_no_val = current_event_meta['eventNo']
        opang_val = current_event_meta['opang']
        num_pc_val = current_event_meta['nPC']
        num_sp_val = current_event_meta['nSP']

        # Get true epem info using the HDF5 store and event_id
        e_pos, p_pos, e_mom, p_mom = epem_info_hdf5(store, event_id)

        fig = go.Figure()

        # Plot wc coords
        if coord_type == "pc" or coord_type == "all":
            pc_x, pc_y, pc_z, num_pc, _ = wc_coords_hdf5(store, event_id, "pc")
            fig.add_trace(go.Scatter3d(x=pc_x, y=pc_y, z=pc_z, showlegend=True, name="pc",
                                       mode="markers", marker=dict(size=1, color="#1f77b4")))

        if coord_type == "sp" or coord_type == "all":
            sp_x, sp_y, sp_z, sp_q, num_sp, _ = wc_coords_hdf5(store, event_id, "sp")
			
            show_colorbar_sp = (coord_type == "sp" or coord_type == "all")
            fig.add_trace(go.Scatter3d(x=sp_x, y=sp_y, z=sp_z, showlegend=True, name="sp",
                                       mode="markers", marker=dict(size=3, color=sp_q, colorscale="jet",
                                                                  showscale=show_colorbar_sp,
                                                                  colorbar=dict(title="q [# e-]")),
                                       text=[f"x = {sp_x[i]:.2f} \n y = {sp_y[i]:.2f} \n z = {sp_z[i]:.2f} \n q = {sp_q[i]:.2f}" for i in range(len(sp_x))], hoverinfo="text"))


        # Plot true epem
        line_length1 = len1
        e_end = [e_pos[0] + e_mom[0] * line_length1, e_pos[1] + e_mom[1] * line_length1, e_pos[2] + e_mom[2] * line_length1]
        fig.add_trace(go.Scatter3d(
            x=[e_pos[0], e_end[0]], y=[e_pos[1], e_end[1]], z=[e_pos[2], e_end[2]],
            name="e-", mode="lines", line=dict(color="black", width=5, showscale=False)
        ))

        line_length2 = len2
        p_end = [p_pos[0] + p_mom[0] * line_length2, p_pos[1] + p_mom[1] * line_length2, p_pos[2] + p_mom[2] * line_length2]
        fig.add_trace(go.Scatter3d(
            x=[p_pos[0], p_end[0]], y=[p_pos[1], p_end[1]], z=[p_pos[2], p_end[2]],
            name="e+", mode="lines", line=dict(color="red", width=5, showscale=False)
        ))

        # Update the layout
        if coord_type == "all":
            title_text = f"True Angle: {opang_val:.2f} [Deg], # pc: {num_pc_val}, # sp: {num_sp_val} <br> Index: {event_id}, Run: {run_no_val}, Sub Run: {sub_run_no_val}, Event: {event_no_val} <br> e- Energy = {e_mom[3]:.2f} MeV, e+ Energy = {p_mom[3]:.2f} MeV"
        else:
            # Use appropriate count based on coord_type
            num_pts_val = num_pc_val if coord_type == "pc" else num_sp_val
            title_text = f"True Angle: {opang_val:.2f} [Deg], # {coord_type}: {num_pts_val} <br> Index: {event_id}, Run: {run_no_val}, Sub Run: {sub_run_no_val}, Event: {event_no_val} <br> e- Energy = {e_mom[3]:.2f} MeV, e+ Energy = {p_mom[3]:.2f} MeV"

        fig.update_layout(
            title=title_text,
            title_x=0.5,
            scene=dict(xaxis_title="x [cm]", yaxis_title="y [cm]", zaxis_title="z [cm]"),
            legend=dict(x=0, y=1, traceorder="reversed" if coord_type == "all" else "normal", itemsizing="constant"),
            autosize=False,
            width=figsize[0],
            height=figsize[1]
        )

    # Show the plot and write to html if specified
    fig.show()
    if html_name is not None:
        fig.write_html(f"{html_name}.html")

    return None

def print_hdf5_info(obj, indent=''):
    """
    Recursively prints information about an HDF5 object (group or dataset).
    """
    if isinstance(obj, h5py.Group):
        # It's a Group
        print(f"{indent}Group: {obj.name}/")
        print(f"{indent}  Attributes:")
        if obj.attrs:
            for key, val in obj.attrs.items():
                print(f"{indent}    {key}: {val}")
        else:
            print(f"{indent}    (No attributes)")

        # Recursively visit members
        for key in obj.keys():
            print_hdf5_info(obj[key], indent + '  ')
    elif isinstance(obj, h5py.Dataset):
        # It's a Dataset
        print(f"{indent}Dataset: {obj.name}")
        print(f"{indent}  Shape: {obj.shape}")
        print(f"{indent}  Dtype: {obj.dtype}")
        print(f"{indent}  Max/Min Value (sample): ", end='')
        try:
            # Attempt to get min/max for numerical data
            if np.issubdtype(obj.dtype, np.number):
                # Sample a small portion for large datasets to avoid loading all
                if obj.size > 100000: # Adjust threshold as needed
                    sample_data = obj[()] # Load a small sample, e.g., first 1000 elements
                    print(f"[{np.min(sample_data):.4g}, {np.max(sample_data):.4g}] (sampled)")
                else:
                    print(f"[{np.min(obj[()]):.4g}, {np.max(obj[()]):.4g}]")
            else:
                print("(Non-numeric data)")
        except Exception as e:
            print(f"(Could not get min/max: {e})")

        print(f"{indent}  Attributes:")
        if obj.attrs:
            for key, val in obj.attrs.items():
                print(f"{indent}    {key}: {val}")
        else:
            print(f"{indent}    (No attributes)")
    else:
        print(f"{indent}Unknown HDF5 object type: {type(obj)}")

def view_hdf5_categories(file_path):
    """
    Opens an HDF5 file and prints its hierarchical structure and properties.
    """
    try:
        with h5py.File(file_path, 'r') as f:
            print(f"\n--- HDF5 File Info: {file_path} ---")
            print(f"Root Attributes:")
            if f.attrs:
                for key, val in f.attrs.items():
                    print(f"  {key}: {val}")
            else:
                print("  (No root attributes)")
            
            print_hdf5_info(f) # Start recursion from the root group
            print(f"--- End HDF5 File Info ---")

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Ensure the HDF5 file exists from the conversion script
    if not os.path.exists(HDF5_FILE_PATH):
        print(f"Error: HDF5 file '{HDF5_FILE_PATH}' not found. Please run the conversion script first.")
        sys.exit(1)
    #view_hdf5_categories(HDF5_FILE_PATH)
    print(f"\nAttempting to view events from HDF5 file '{HDF5_FILE_PATH}'.")

    
    # 1. View an event using an integer index
    event_index_to_view = 49 # Example: View the first event
    
    print(f"\nViewing event by integer index: {event_index_to_view}")
    view_event3d_hdf5(HDF5_FILE_PATH, event_index_to_view, coord_type="all", html_name="event_view_int_index_from_hdf5")

    # 2. View an event using RSE (Run, SubRun, Event)
    # For demonstration, let's assume event_id=0 corresponds to [1001, 1, 10] from the dummy data structure.
    # Replace with an actual RSE from your converted data.
    '''
    try:
        with pd.HDFStore(HDF5_FILE_PATH, 'r') as store:
            first_event_rse_from_hdf5 = store.select('event_metadata', columns=['runNo', 'subRunNo', 'eventNo']).iloc[0][['runNo', 'subRunNo', 'eventNo']].tolist()
        event_rse_to_view = first_event_rse_from_hdf5
        print(f"\nViewing event by RSE: {event_rse_to_view}")
        view_event3d_hdf5(HDF5_FILE_PATH, event_rse_to_view, coord_type="all", html_name="event_view_rse_from_hdf5")
    except Exception as e:
        print(f"Could not determine an RSE for testing: {e}. Please manually provide a valid RSE from your HDF5 file.")
        '''
    
