import h5py

file_path = "epem_sample_restructured_chunked.h5"
with h5py.File(file_path, 'r') as f:
    print("Keys (datasets/groups) in HDF5 file:", list(f.keys()))
    # If your data is nested, explore further
    # for key in f.keys():
    #     if isinstance(f[key], h5py.Group):
    #         print(f"  Subkeys in {key}:", list(f[key].keys()))

    # Example: print shape of a known dataset (replace 'points_data' with your actual key)
    if 'points_data' in f:
        print(f"Shape of 'points_data': {f['points_data'].shape}")
        print(f"Type of 'points_data': {f['points_data'].dtype}")
    # Identify how labels/classes are stored
    if 'labels' in f:
        print(f"Shape of 'labels': {f['labels'].shape}")