#Final layer to map features to single regression angle output

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
import os 

from PointNeXt.openpoints.models.build import MODELS
from PointNeXt.openpoints.models.layers import create_convblock1d

@MODELS.register_module()
class RegressionHead(nn.Module):
	def __init__(self, encoder_out_channnels: int, out_dim: int=1, mlp_channels: List[int]=[512,256], norm_args=None, act_args=None, **kwargs):
		super().__init__()
		self.mlp_channels= [encoder_out_channels]+mlp_channels
		mlp_layers=[]

		for i in range(len(self.mlp_channels)-1):
				mlp_layers.append(create_convblock1d(self.mlp_channels[i], self.mlp_channels[i+1], norm_args=norm_args, act_args=act_args))
		self.mlp=nn.Sequential(*mlp_layers)
		self.linear_out=nn.Linear(self.mlp_channels[-1],out_dim)

	def foward(self, features: torch.Tensor) -> torch.Tensor:
		#maxpool
		features_pooled=featruesm.max(dim=-1, keepdim=False)[0]

	#Multilayer perceptron 
	mlp_input = features_pooled.unsqueeze(-1)
	mlp_output = self.mlp(mlp_input)
	#squeeze
	mlp_output=mlp_output.squeeze(-1)

	output=self.linear_out(mlp_output)
	return output



