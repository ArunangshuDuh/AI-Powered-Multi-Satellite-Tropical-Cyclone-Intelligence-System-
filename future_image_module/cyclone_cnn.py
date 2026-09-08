
import torch
import torch.nn as nn
import timm
import config_reference as config

class CycloneCNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Load pretrained EfficientNet-B0
        self.backbone = timm.create_model('efficientnet_b0', pretrained=True)
        
        # Replace the first conv layer for 5 channels
        old_conv = self.backbone.conv_stem
        new_conv = nn.Conv2d(
            in_channels=5,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None
        )
        
        # Initialize new weights by averaging the pretrained RGB weights
        with torch.no_grad():
            old_weight = old_conv.weight  # [out_channels, 3, k, k]
            avg_weight = old_weight.mean(dim=1, keepdim=True)  # [out, 1, k, k]
            new_conv.weight.copy_(avg_weight.repeat(1, 5, 1, 1))  # [out, 5, k, k]
            if old_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)
                
        self.backbone.conv_stem = new_conv
        
        self.feat_dim = config.BACKBONE_FEAT_DIM
        self.meta_dim = config.NUM_META_FEATURES
        self.concat_dim = self.feat_dim + self.meta_dim
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.concat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, config.NUM_CATEGORIES)
        )
        
        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(self.concat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def get_feature_extractor(self):
        """Returns a function to extract features up to global average pooling."""
        def extract(images):
            features = self.backbone.forward_features(images)
            return self.backbone.forward_head(features, pre_logits=True)
        return extract

    def forward(self, images, metadata):
        # Extract features [B, 1280]
        extractor = self.get_feature_extractor()
        features = extractor(images)
        
        # Concatenate metadata [B, 1284]
        combined = torch.cat([features, metadata], dim=1)
        
        # Output heads
        class_logits = self.classifier(combined)
        regression = self.regressor(combined)
        
        return class_logits, regression

    def get_param_groups(self, lr):
        """Returns parameter groups with differential learning rates."""
        early_params = []
        late_params = []
        head_params = []
        
        for name, param in self.backbone.named_parameters():
            if 'blocks.0.' in name or 'blocks.1.' in name or 'blocks.2.' in name or 'conv_stem' in name or 'bn1' in name:
                early_params.append(param)
            else:
                late_params.append(param)
                
        for param in self.classifier.parameters():
            head_params.append(param)
        for param in self.regressor.parameters():
            head_params.append(param)
            
        return [
            {'params': early_params, 'lr': lr * 0.01},
            {'params': late_params, 'lr': lr * 0.1},
            {'params': head_params, 'lr': lr * 1.0}
        ]
