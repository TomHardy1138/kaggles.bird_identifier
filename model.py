import torch.nn as nn
import timm


class Network(nn.Module):
    def __init__(self, backbone, num_classes, metric_losses=False, **kwargs):
        super(Network, self).__init__()
        self.model = timm.create_model(
            backbone,
            num_classes=0,
            pretrained=False,
            **kwargs
        )
        self.classifier = nn.Linear(self.model.num_features, num_classes)
        self.metric_losses = metric_losses
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        features = self.model(x)
        preds = self.classifier(features)

        if self.training and self.metric_losses:
            return features, preds

        return preds
