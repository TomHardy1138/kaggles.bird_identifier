import torch.nn as nn
import timm


class Network(nn.Module):
    def __init__(self, backbone, num_classes, metric_losses=False, **kwargs):
        super(Network, self).__init__()
        self.model = timm.create_model(
            backbone,
            num_classes=0,
            pretrained=True,
            **kwargs
        )
        self.start_conv = nn.Conv2d(1, 3, (3, 3))
        self.classifier = nn.Linear(self.model.num_features, num_classes)
        self.metric_losses = metric_losses

    def forward(self, x):
        features = self.model(self.start_conv(x))
        preds = self.classifier(features)

        if self.training and self.metric_losses:
            return features, preds

        return preds
