import torch
import torch_geometric
import os

from openpoints.models.backbone.pointnext import PointNextEncoder
from openpoints.models import build_model_from_cfg
from openpoints.utils.config import EasyConfig

print("Successfully imported necessary components!")

print(f"\n--- PyTorch & CUDA Status ---")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"CUDA current device: {torch.cuda.current_device()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: CUDA not available. PointNeXt will run on CPU, which may be slow.")

print(f"\n--- PyTorch Geometric Components ---")
try:
    import torch_scatter
    print("torch_scatter imported successfully.")
except ImportError as e:
    print(f"Error importing torch_scatter: {e}")

try:
    import torch_cluster
    print("torch_cluster imported successfully.")
except ImportError as e:
    print(f"Error importing torch_cluster: {e}")

print(f"\n--- PointNeXt Model Test ---")
try:
    batch_size = 2
    num_points = 2048
    feature_dim = 6

    model_cfg = EasyConfig({
        'NAME': 'PointNextEncoder',
        'in_channels': feature_dim,
        'width': 32,
        'blocks': [1, 4, 7, 4],
        'strides': [4, 4, 4, 4],
        'radius': 0.1,
        'nsample': 32,
        'expansion': 4,
        'sa_layers': 1,
        'sa_use_res': True,
        'use_res': True,
        'group_args': EasyConfig({'NAME': 'ballquery'}),
        'norm_args': EasyConfig({'norm': 'bn'}),
        'act_args': EasyConfig({'act': 'relu'}),
        'conv_args': EasyConfig({}),
        'sampler': 'fps',
        'radius_scaling': 2,
        'nsample_scaling': 1,
    })

    model = build_model_from_cfg(model_cfg)

    print("PointNext model instantiated successfully!")

    if torch.cuda.is_available():
        model.cuda()
        print("PointNext model moved to GPU.")
    else:
        print("PointNext model instantiated on CPU.")

    xyz_dummy = torch.randn(batch_size, num_points, 3) 
    x_dummy = torch.randn(batch_size, feature_dim, num_points)

    if torch.cuda.is_available():
        x_dummy = x_dummy.cuda()
        xyz_dummy = xyz_dummy.cuda()

    print(f"Running forward pass with input shapes: x={x_dummy.shape}, xyz={xyz_dummy.shape}")

    output_p, output_f = model(xyz_dummy, x_dummy)
    print(f"Forward pass successful.")
    print(f"Final output position shape: {output_p[-1].shape}") 
    print(f"Final output feature shape: {output_f[-1].shape}")

except Exception as e:
    print(f"An error occurred during PointNext model test: {e}") 
    import traceback
    traceback.print_exc()

print("\n--- Installation Verification Complete ---")