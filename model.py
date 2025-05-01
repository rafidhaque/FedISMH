import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple

class FeatureExtractor:
    def __init__(self):
        self.features = {}
        self.hooks = []

    def get_activation(self, name):
        def hook(model, input, output):
            self.features[name] = output.detach()
        return hook

    def attach_hooks(self, model: nn.Module, layer_names: List[str]):
        for name, layer in model.named_modules():
            if name in layer_names:
                hook = layer.register_forward_hook(self.get_activation(name))
                self.hooks.append(hook)

    def clear_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.features.clear()

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = FeatureExtractor()
        self.feature_layers = []

    def get_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        self.feature_extractor.attach_hooks(self, self.feature_layers)
        _ = self(x)
        features = self.feature_extractor.features
        self.feature_extractor.clear_hooks()
        return features

    def get_flattened_features(self, x: torch.Tensor) -> torch.Tensor:
        self.feature_extractor.attach_hooks(self, self.feature_layers)
        _ = self(x)
        features = self.feature_extractor.features
        
        flattened_features = []
        for name in self.feature_layers:
            if name in features:
                flat = torch.flatten(features[name], start_dim=1)
                flattened_features.append(flat)
        
        self.feature_extractor.clear_hooks()
        
        if not flattened_features:
            raise ValueError("No valid features captured for flattening and concatenation.")
        return torch.cat(flattened_features, dim=1)

class SimpleConvNet(BaseModel):
    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
        self.feature_layers = ['conv2', 'fc1']

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ModelWithProjection(nn.Module):
    def __init__(self, base_model, feature_dim, projection_dim=128):
        super().__init__()
        self.base_model = base_model
        self.projection = nn.Linear(feature_dim, projection_dim)

    def forward(self, x):
        features = self.base_model.get_flattened_features(x)
        projected_features = self.projection(features)
        return projected_features

    def get_projected_features(self, x):
        return self.forward(x)

class ModelFactory:
    @staticmethod
    def create_model(model_name: str, in_channels: int = 1, num_classes: int = 10, projection_dim: int = 128) -> nn.Module:
        models = {
            'simple': SimpleConvNet,
            'resnet': SimpleConvNet,  # For simplicity, using SimpleConvNet for both
        }
        
        if model_name not in models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(models.keys())}")
        
        base_model = models[model_name](in_channels=in_channels, num_classes=num_classes)
        
        feature_dim_mapping = {
            'simple': 13056,
            'resnet': 13056,
        }
        feature_dim = feature_dim_mapping[model_name]
        base_model.expected_feature_dim = feature_dim

        return ModelWithProjection(base_model, feature_dim=feature_dim, projection_dim=projection_dim)