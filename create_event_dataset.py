import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import pandas as pd
from typing import Union, Tuple, List, Optional
import os 
from PointNeXt.openpoints.transforms import build_transforms_from_cfg

#easilt fecthable
class EventPointCloudDataset(Dataset):
	def __init__ (self, h5_file_path, num_points=2048, use_transforms=None):
		self.h5_file_path=h5_file_path
		self.num_points=num_points
		with pd.HDFStore(h5_file_path, 'r') as store:
			self.event_metadata_df=store.select('event_metadata')

		self.num_samples =len(self.event_metadata_df)
		print(f"Dataset initialized with {self.num_samples} events from {h5_file_path}.")

		if use_transforms:
				self.transforms=build_transforms_from_cfg(use_transforms)

		else:
				self.transforms=None

	def __len__(self):
			return self.num_samples

	def __getitem__(self,indx):
		with pd.HDFStore(self.h5_file_path, 'r') as hdf5_store:
			current_event_meta = self.event_metadata_df.iloc[indx]
			run_no = current_event_meta['runNo']
			sub_run_no= current_event_meta['subRunNo']
			event_no = current_event_meta['eventNo']
			opang = current_event_meta['opang']
			num_pc= current_event_meta['nPC']
			num_sp = current_event_meta['nSP']

			pc_data = hdf5_store.select('pc_points', where=f"runNo == {run_no} and subRunNo == {sub_run_no} and eventNo == {event_no}")
			
			x_coords = pc_data["x"].values
			y_coords = pc_data["y"].values
			z_coords = pc_data["z"].values
			
			#combine into 3 numpt array
			
			points_xyz = np.stack([x_coords, y_coords, z_coords], axis=-1) 
			features = points_xyz.copy()
			
			N_original = points_xyz.shape[0]
			if N_original > self.num_points:
				#Randomly sample points
				sample_idx = np.random.choice(N_original, self.num_points, replace=False)
				points_xyz = points_xyz[sample_idx]
				features = features[sample_idx]
			elif N_original < self.num_points:
				#Padding
				padded_points_xyz = np.zeros((self.num_points, points_xyz.shape[1]), dtype=points_xyz.dtype)
				padded_features = np.zeros((self.num_points, features.shape[1]), dtype=features.dtype)
				padded_points_xyz[:N_original] = points_xyz
				padded_features[:N_original] = features
				points_xyz = padded_points_xyz
				features = padded_features

			pos_tensor = torch.from_numpy(points_xyz).float()
			feat_tensor = torch.from_numpy(features).float() 

				
			feat_tensor = feat_tensor.transpose(0, 1).contiguous() 
			
			target_tensor = torch.tensor(opang, dtype=torch.float) 
			
			if self.transforms:
				data_dict_for_transform = {'pos': pos_tensor, 'x': feat_tensor, 'y': target_tensor}
			
				data_dict_for_transform = self.transforms(data_dict_for_transform)
				pos_tensor = data_dict_for_transform['pos']
				feat_tensor = data_dict_for_transform['x']
				target_tensor = data_dict_for_transform['y']

			# Return as a dict for inputs, and the target
			#Keys 'pos' and 'x' match expected arguments for PointNextEncoder's forward
			data_dict = {'pos': pos_tensor, 'x': feat_tensor} 
			
			return data_dict, target_tensor

if __name__ == "__main__":
	HDF5_FILE_PATH = "epem_sample_restructured_chunked.h5"
	dataset = EventPointCloudDataset(HDF5_FILE_PATH, num_points=2048)
	data_dict, target = dataset[0]
	print(f"POS shape: {data_dict['pos'].shape}, X shape: {data_dict['x'].shape}, Target shape: {target.shape}")
