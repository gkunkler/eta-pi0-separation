# File: openpoints/models/regression_model_wrapper.py (or similar new file)
import torch.nn as nn
from openpoints.models.build import MODELS, build_model_from_cfg
import numpy as np
import torch

@MODELS.register_module()
class ClassificationModelWrapper(nn.Module):
    def __init__(self, encoder_args, head_args):
        super().__init__()
        self.encoder = build_model_from_cfg(encoder_args)
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
        # f_list = self.encoder.forward_cls_feat(pos, features) # Switches to the classification forward function in PointNextDecoder

        # Get the final feature tensor from the encoder's output
        final_features = f_list[-1]
        final_feature = torch.max(final_features, dim=1)[0]
        # final_feature = torch.mean(final_features, dim=1)

        # print(f'final features: {np.shape(final_features)}')
        # print(f'final feature: {np.shape(final_feature)}')
        # print(f'{final_feature}')

        # Head takes features
        classification_output = self.head(final_feature)

        # Normalize to sum to one
        classification_output = nn.functional.normalize(classification_output, p=1, dim=1)

        # print(classification_output)

        return classification_output