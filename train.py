import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn

from dataset import BirdClefDataset, collate_fn
from model import Network


dataset = BirdClefDataset()
loader = DataLoader(dataset, collate_fn=collate_fn, batch_size=4, num_workers=4)
model = Network(backbone='resnet34', num_classes=397)
model.cuda()

criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=3e-2)
for i, (data, target, _) in enumerate(loader):
    # print(data.shape)
    x = F.conv2d(data, torch.ones(3,1,3,3))
    # print(x.shape)
    out = model(x.cuda())
    # print(out.shape)
    # print(target)
    # print(torch.stack(target).shape)
    target = torch.stack(target)
    loss = criterion(F.sigmoid(out), target.cuda())
    print(loss.item())
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
