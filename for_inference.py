import torch
import torch.nn as nn
import timm


NUM_CLASSES: int = 397
DEVICE: str = "cuda"


class Network(nn.Module):
    def __init__(self, backbone, num_classes, metric_losses=False, **kwargs):
        super(Network, self).__init__()
        self.model = timm.create_model(
            backbone,
            num_classes=0,
            pretrained=True,
            **kwargs
        )
        self.classifier = nn.Linear(
            self.model.num_features,
            num_classes
        )

    def forward(self, x):
        features = self.model(x)
        preds = self.classifier(features)

        if self.training and self.metric_losses:
            return features, preds

        return preds


def load_net(
    backbone: str,
    checkpoint_path: str,
    num_classes: int = NUM_CLASSES
):
    net = Network(backbone, num_classes)
    
    dummy_device = torch.device("cpu")
    state_dict = torch.load(checkpoint_path, map_location=dummy_device)
    
    net.load_state_dict(state_dict)
    net = net.to(DEVICE)
    net = net.eval()

    return net