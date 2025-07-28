# File: openpoints/models/regression_model_wrapper.py (or similar new file)
import torch.nn as nn
from openpoints.models.build import MODELS, build_model_from_cfg

@MODELS.register_module()
class RegressionModelWrapper(nn.Module):
    def __init__(self, encoder_args, head_args):
        super().__init__()
        self.encoder = build_model_from_cfg(encoder_args)
        # Pass the encoder's output channels to the head, if not specified in head_args
        if 'encoder_out_channels' not in head_args:
            head_args['encoder_out_channels'] = self.encoder.out_channels
        self.head = build_model_from_cfg(head_args)

    def forward(self, data_dict):
        # Encoder takes (pos, features)
        # Make sure data_dict has 'pos' and 'x' keys
        pos = data_dict['pos']
        features = data_dict['x'] # Features are typically passed as 'x' in OpenPoints
        
        # Encoder returns a list of (p, f) pairs, with final output as last f
        # Or sometimes just the final feature tensor, depends on forward_seg_feat/cls_feat.
        # Assuming forward_seg_feat returns (p_list, f_list)
        p_list, f_list = self.encoder(pos, features) 
        
        # Get the final feature tensor from the encoder's output
        final_features = f_list[-1] 

        # Head takes features
        regression_output = self.head(final_features)
        return regression_output