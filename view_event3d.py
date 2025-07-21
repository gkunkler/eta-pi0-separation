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
from matplotlib import colors 
from matplotlib.ticker import PercentFormatter 
from typing import Union, Tuple, List, Optional

# View event with true info as interactive 3D plot!
def wc_coords(data, index, coord_type="pc"):
    if type(index) == int:
        if coord_type == "pc":
            x = data["pc_x"][index]
            y = data["pc_y"][index]
            z = data["pc_z"][index]
            num_pts = data["nPC"][index]
            opang = data["opang"][index]
            return x, y, z, num_pts, opang
        elif coord_type == "sp":
            x = data["sp_x"][index]
            y = data["sp_y"][index]
            z = data["sp_z"][index]
            q = data["sp_q"][index]
            num_pts = data["nSP"][index]
            opang = data["opang"][index]
            return x, y, z, q, num_pts, opang
    elif type(index) is list and all(isinstance(val, int) for val in index):
        idx = (data.loc[(data["runNo"] == index[0]) & (data["subRunNo"] == index[1]) & (data["eventNo"] == index[2])]).index[0]
        if coord_type == "pc":
            x = data["pc_x"][idx]
            y = data["pc_y"][idx]
            z = data["pc_z"][idx]
            num_pts = data["nPC"][idx]   
            opang = data["opang"][idx] 
            return x, y, z, num_pts, opang, idx
        elif coord_type == "sp":
            x = data["sp_x"][idx]
            y = data["sp_y"][idx]
            z = data["sp_z"][idx]
            q = data["sp_q"][idx]
            num_pts = data["nSP"][idx]
            opang = data["opang"][idx]
            return x, y, z, q, num_pts, opang, idx
        
def epem_info(data, index):
    e_x = data["x1"][index]
    e_y = data["y1"][index]
    e_z = data["z1"][index]
    e_pos = [e_x, e_y, e_z]

    e_px = data["p1x"][index]
    e_py = data["p1y"][index]
    e_pz = data["p1z"][index]
    e_E = data["E1"][index]
    e_mom = [e_px, e_py, e_pz, e_E]

    p_x = data["x2"][index]
    p_y = data["y2"][index]
    p_z = data["z2"][index]
    p_pos = [p_x, p_y, p_z]

    p_px = data["p2x"][index]
    p_py = data["p2y"][index]
    p_pz = data["p2z"][index]
    p_E = data["E2"][index]
    p_mom = [p_px, p_py, p_pz, p_E]

    return e_pos, p_pos, e_mom, p_mom


# The god function to view an event in 3D interactive!
# The god function to view an event in 3D interactive!
def view_event3d(data, index, coord_type="all", len1=100, len2=500, figsize=[800,800], html_name=None):
    """
    Args:
        data: df to view events of
        index: Can be int or rse
        coord_type: If sp, will plot charge in cbar
        remove_offset: Fix x offset of true epem (TO BE IMPLEMENTED)
        
    Returns:
        fig: Figure with sp/pc and epem momenta
    """

    # Get wc coords
    if coord_type=="pc" and type(index) == int: x, y, z, num_pts, opang = wc_coords(data, index, coord_type)
    elif coord_type=="pc" and type(index) is list: x, y, z, num_pts, opang, idx = wc_coords(data, index, coord_type)
    elif coord_type=="sp" and type(index) == int: x, y, z, q, num_pts, opang = wc_coords(data, index, coord_type)
    elif coord_type=="sp" and type(index) is list: x, y, z, q, num_pts, opang, idx = wc_coords(data, index, coord_type)
    elif coord_type=="all" and type(index) == int: 
        pc_x, pc_y, pc_z, num_pc, opang = wc_coords(data, index, "pc")
        sp_x, sp_y, sp_z, sp_q, num_sp, opang = wc_coords(data, index, "sp")
    elif coord_type=="all" and type(index) is list: 
        pc_x, pc_y, pc_z, num_pc, _, _ = wc_coords(data, index, "pc")
        sp_x, sp_y, sp_z, sp_q, num_sp, opang, idx = wc_coords(data, index, "sp")
    else:
        print("Not a valid index and/or coordinate types!!!")
        return None
    
    # Get true epem info
    if type(index) == int: e_pos, p_pos, e_mom, p_mom = epem_info(data, index)
    else: e_pos, p_pos, e_mom, p_mom = epem_info(data, idx)

    fig = go.Figure()

    # Plot wc coords
    if coord_type=="pc": fig.add_trace(go.Scatter3d(x=x, y=y, z=z, showlegend=False, name="pc",
        mode="markers", marker=dict(size=1, color="#1f77b4")))
    elif coord_type=="sp": fig.add_trace(go.Scatter3d(x=x, y=y, z=z, showlegend=False, name="sp",
        mode="markers", marker=dict(size=3, color=q, colorscale="jet", showscale=True, colorbar = dict(title="q [# e-]")),#))
        text=[f"x = {x[i]:.2f} \n y = {y[i]:.2f} \n z = {z[i]:.2f} \n q = {q[i]:.2f}" for i in range(num_pts)], hoverinfo="text"))
    elif coord_type=="all":
        fig.add_trace(go.Scatter3d(x=pc_x, y=pc_y, z=pc_z, showlegend=True, name="pc",
        mode="markers", marker=dict(size=1, color="#1f77b4")))
        fig.add_trace(go.Scatter3d(x=sp_x, y=sp_y, z=sp_z, showlegend=True, name="sp",
        mode="markers", marker=dict(size=3, color=sp_q, colorscale="jet", showscale=True, colorbar = dict(title="q [# e-]"))))
    
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
    
    # Update the layout to enhance interactivity
    if coord_type=="all":
        if type(index) is int:
            fig.update_layout(
                title=f"True Angle: {opang:.2f} [Deg], # pc: {num_pc}, # sp: {num_sp} <br> Index: {index}, Run: {data.iloc[index]['runNo']}, Sub Run: {data.iloc[index]['subRunNo']}, Event: {data.iloc[index]['eventNo']} <br> e- Energy = {e_mom[3]:.2f} MeV, e+ Energy = {p_mom[3]:.2f} MeV",
                title_x=0.5,
                scene=dict(xaxis_title="x [cm]", yaxis_title="y [cm]", zaxis_title="z [cm]"),
                legend=dict(x=0, y=1, traceorder="reversed", itemsizing="constant"),
                autosize=False,
                width=figsize[0],
                height=figsize[1]
            )
        else:
            fig.update_layout(
                title=f"True Angle: {opang:.2f} [Deg], # pc: {num_pc}, # sp: {num_sp} <br> Index: {idx}, Run: {data.iloc[idx]['runNo']}, Sub Run: {data.iloc[idx]['subRunNo']}, Event: {data.iloc[idx]['eventNo']} <br> e- Energy = {e_mom[3]:.2f} MeV, e+ Energy = {p_mom[3]:.2f} MeV",
                title_x=0.5,
                scene=dict(xaxis_title="x [cm]", yaxis_title="y [cm]", zaxis_title="z [cm]"),
                legend=dict(x=0, y=1, traceorder="reversed", itemsizing="constant"),
                autosize=False,
                width=figsize[0],
                height=figsize[1]
            )
    else:
        if type(index) is int:
            fig.update_layout(
                title=f"True Angle: {opang:.2f} [Deg], # {coord_type}: {num_pts} <br> Index: {index}, Run: {data.iloc[index]['runNo']}, Sub Run: {data.iloc[index]['subRunNo']}, Event: {data.iloc[index]['eventNo']} <br> e- Energy = {e_mom[3]:.2f} MeV, e+ Energy = {p_mom[3]:.2f} MeV",
                title_x=0.5,
                scene=dict(xaxis_title="x [cm]", yaxis_title="y [cm]", zaxis_title="z [cm]"),
                legend=dict(x=0, y=1, traceorder="normal"),
                autosize=False,
                width=figsize[0],
                height=figsize[1]
            )
        else:
            fig.update_layout(
                title=f"True Angle: {opang:.2f} [Deg], # {coord_type}: {num_pts} <br> Index: {idx}, Run: {data.iloc[idx]['runNo']}, Sub Run: {data.iloc[idx]['subRunNo']}, Event: {data.iloc[idx]['eventNo']} <br> e- Energy = {e_mom[3]:.2f} MeV, e+ Energy = {p_mom[3]:.2f} MeV",
                title_x=0.5,
                scene=dict(xaxis_title="x [cm]", yaxis_title="y [cm]", zaxis_title="z [cm]"),
                legend=dict(x=0, y=1, traceorder="normal"),
                autosize=False,
                width=figsize[0],
                height=figsize[1]
            )

    # Show the plot and write to html if specified
    fig.show()
    if html_name is not None: fig.write_html(f"{html_name}.html")

    return None

# Get sample index from any df index using rse info
def get_sample_idx(data, results, index):
    r = results["r"][index]
    s = results["s"][index]
    e = results["e"][index]

    sample_idx = data[(data["runNo"] == r) & (data["subRunNo"] == s) & (data["eventNo"] == e)].index[0]
    return int(sample_idx)      # view_event3d gets whiny if index isn't specified as int

# Create shareable link to view interactive 3D plot
def share_link(g_link):
    path = g_link.split("/")
    path = path[-2]
    shareable_link = f"https://drive.google.com/uc?id={path}"
    print(shareable_link)

def create_dummy_pkl_file(filename="dummy_events.pkl"):
    """
    Creates a dummy .pkl file with a structure similar to what view_event3d expects.
    """
    num_events = 2
    data = {
        "pc_x": [np.random.rand(100) * 100 for _ in range(num_events)],
        "pc_y": [np.random.rand(100) * 100 for _ in range(num_events)],
        "pc_z": [np.random.rand(100) * 100 for _ in range(num_events)],
        "nPC": [100 for _ in range(num_events)],
        "opang": [np.random.rand() * 180 for _ in range(num_events)],
        "sp_x": [np.random.rand(50) * 100 for _ in range(num_events)],
        "sp_y": [np.random.rand(50) * 100 for _ in range(num_events)],
        "sp_z": [np.random.rand(50) * 100 for _ in range(num_events)],
        "sp_q": [np.random.rand(50) * 10 for _ in range(num_events)],
        "nSP": [50 for _ in range(num_events)],
        "x1": [np.random.rand() * 10 for _ in range(num_events)],
        "y1": [np.random.rand() * 10 for _ in range(num_events)],
        "z1": [np.random.rand() * 10 for _ in range(num_events)],
        "p1x": [np.random.rand() * 5 for _ in range(num_events)],
        "p1y": [np.random.rand() * 5 for _ in range(num_events)],
        "p1z": [np.random.rand() * 5 for _ in range(num_events)],
        "E1": [np.random.rand() * 100 for _ in range(num_events)],
        "x2": [np.random.rand() * 10 for _ in range(num_events)],
        "y2": [np.random.rand() * 10 for _ in range(num_events)],
        "z2": [np.random.rand() * 10 for _ in range(num_events)],
        "p2x": [np.random.rand() * 5 for _ in range(num_events)],
        "p2y": [np.random.rand() * 5 for _ in range(num_events)],
        "p2z": [np.random.rand() * 5 for _ in range(num_events)],
        "E2": [np.random.rand() * 100 for _ in range(num_events)],
        "runNo": [1001, 1002],
        "subRunNo": [1, 1],
        "eventNo": [10, 11]
    }
    df = pd.DataFrame(data)
    df.to_pickle(filename)
    print(f"Dummy .pkl file '{filename}' created.")
    return filename


if __name__ == "__main__":
    pkl_file_path="epem_sample.pkl"
    try:
        loaded_data = pd.read_pickle(pkl_file_path)
        print(f"\nSuccessfully loaded data from '{pkl_file_path}'.")
        print(f"DataFrame shape: {loaded_data.shape}")
        print("First 5 rows of the loaded DataFrame:")
        print(loaded_data[['runNo', 'subRunNo', 'eventNo', 'nPC', 'nSP', 'opang']].head())

        # 3. View an event
        # You can choose an index:
        # - By integer index (e.g., 0 for the first event)
        event_index_int = 0
        print(f"\nViewing event by integer index: {event_index_int}")
        view_event3d(loaded_data, event_index_int, coord_type="all", html_name="event_view_int_index")

        # - By RSE (Run, SubRun, Event)
        # Make sure this RSE exists in your dummy data or actual data
        event_rse = [1002, 1, 11]
        print(f"\nViewing event by RSE: {event_rse}")
        view_event3d(loaded_data, event_rse, coord_type="all", html_name="event_view_rse")

    except FileNotFoundError:
        print(f"Error: The file '{pkl_file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
