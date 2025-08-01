import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import pandas as pd
from typing import Union, Tuple, List, Optional
import os 
from openpoints.transforms import build_transforms_from_cfg
import time
from tqdm import tqdm

#easilt fecthable
class EventPointCloudDataset(Dataset):
	def __init__ (self, h5_file_path, num_points=2048, use_transforms=None, max_samples_per_category=None):
		self.h5_file_path=h5_file_path
		self.num_points=num_points
		if use_transforms:
			self.transforms=build_transforms_from_cfg(use_transforms)

		else:
			self.transforms=None
		self.h5_file = None 

		self.num_samples = 0

		self.h5_file_store = []

		self.preloaded_pc_data = []
		self.preloaded_pc_target = []
		self.preloaded_sequence_info = []
		
		with h5py.File(h5_file_path, 'r') as f:

			print(f.keys())

			label_index = 0 # Only works for binary groups
			for group in f.keys():

				grp = f[group]
				sequence_info = grp['sequence_info_filtered']
				
				if max_samples_per_category is not None and max_samples_per_category > 0 and max_samples_per_category < len(sequence_info):
					sequence_info = sequence_info[:max_samples_per_category] # Take the first N samples
					print(f"Dataset truncated to {max_samples_per_category} samples per category for testing.")

				self.num_samples += len(sequence_info)

				print(sequence_info)

				print(f"Preloading {len(sequence_info)} sets of PC data into RAM (this might take a moment if many samples)...")
				for event_index, starting_index, num_points, _, _, _ in sequence_info:

					points = grp['point_info_centered'][starting_index:starting_index+num_points]

					self.preloaded_sequence_info.extend(sequence_info)
					self.preloaded_pc_data.extend(points)
					self.preloaded_pc_target.extend(np.ones(num_points)*label_index)

				label_index+=1

			self.preloaded_sequence_info = np.array(self.preloaded_sequence_info)
			self.preloaded_pc_data = np.array(self.preloaded_pc_data)
			self.preloaded_pc_target = np.array(self.preloaded_pc_target)
			
			print(f"Dataset initialized with {self.num_samples} events from {h5_file_path}.")
			print(f"Finished preloading {len(self.preloaded_pc_data)} PC event data sets into RAM.")

			# self.all_preloaded_pc_data={}
			# self.all_preloaded_pc_targets={}
			# rse_tuples_to_preload = []
			# for index, row in self.event_metadata_df.iterrows():
			# 	rse_tuple = (row['runNo'], row['subRunNo'], row['eventNo'])
			# 	rse_tuples_to_preload.append(rse_tuple)
			# 	self.all_preloaded_pc_targets[rse_tuple] = row['opang']
			# print(f"Preloading {len(rse_tuples_to_preload)} sets of PC data into RAM (this might take a moment if many samples)...")
			# with pd.HDFStore(h5_file_path, 'r') as preload_store: 
			# 	for rse_tuple in tqdm(rse_tuples_to_preload, desc="Preloading PC data"):
			# 		run_no, sub_run_no, event_no = rse_tuple
			# 		pc_data = preload_store.select('pc_points', where=f"runNo == {run_no} and subRunNo == {sub_run_no} and eventNo == {event_no}")
			# 		self.all_preloaded_pc_data[rse_tuple] = {
			# 			'x': pc_data["x"].values,
			# 			'y': pc_data["y"].values,
			# 			'z': pc_data["z"].values
			# 		}

			# print(f"Dataset initialized with {self.num_samples} events from {h5_file_path}.")
			# print(f"Finished preloading {len(self.all_preloaded_pc_data)} PC event data sets into RAM.")



	def __len__(self):
			return self.num_samples

	#for workers but don't need anymore
	@staticmethod
	def worker_init_fn(worker_id):
		pass
		"""
		worker_info = torch.utils.data.get_worker_info()
		dataset_obj = worker_info.dataset
		if isinstance(dataset_obj, torch.utils.data.Subset):
			original_dataset_for_worker = dataset_obj.dataset
		else:
			original_dataset_for_worker = dataset_obj 
		original_dataset_for_worker.hdf5_store = pd.HDFStore(original_dataset_for_worker.h5_file_path, 'r')
		print(f"HDFStore opened in worker {worker_id}") 
		"""


	def __getitem__(self,indx):
		with pd.HDFStore(self.h5_file_path, 'r') as hdf5_store:

			hdf5_select_start = time.time()
			current_event_meta = self.event_metadata_df.iloc[indx]
			rse_tuple = (current_event_meta['runNo'], current_event_meta['subRunNo'], current_event_meta['eventNo'])
			

			pc_data = self.all_preloaded_pc_data[rse_tuple]
			opang =self.all_preloaded_pc_targets[rse_tuple]
			hdf5_select_end =  time.time()
			print(f"DEBUG DATASET: GetItem {indx} - HDF5 Select Duration: {hdf5_select_end - hdf5_select_start:.4f}s")


			x_coords = pc_data["x"]
			y_coords = pc_data["y"]
			z_coords = pc_data["z"]
		
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
			data_dict = {'pos': pos_tensor, 'x': feat_tensor} 
			
			return data_dict, target_tensor

	def __del__(self):
		if hasattr(self, 'hdf5_store') and self.hdf5_store is not None and self.hdf5_store.is_open:
			self.hdf5_store.close()
			print("HDFStore closed in __del__")

if __name__ == "__main__":
	HDF5_FILE_PATH = "../../eta-pi-data/eta-pi0.h5"
	dataset = EventPointCloudDataset(HDF5_FILE_PATH, num_points=2048, max_samples_per_category=10)
	data_dict, target = dataset[0]
	print(f"POS shape: {data_dict['pos'].shape}, X shape: {data_dict['x'].shape}, Target shape: {target.shape}")
