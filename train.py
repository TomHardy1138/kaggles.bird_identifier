import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn

from dataset import BirdClefDataset, collate_fn
from model import Network


if __name__ == '__main__':

    dataset = BirdClefDataset()
    loader = DataLoader(dataset, collate_fn=collate_fn,
                        batch_size=16, num_workers=4,
                        sampler=torch.utils.data.RandomSampler(dataset))
    model = Network(backbone='resnest50d', num_classes=397)
    model.cuda()

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    for i, (data, target) in enumerate(loader):
        # print(data.shape)
        # print(x.shape)
        data = data.cuda()
        out = model(data)
        # print(out.shape)
        # print(target)
        # print(torch.stack(target).shape)
        target = target.cuda()
        loss = criterion(out, target.cuda())
        print(loss.item())
        print(torch.sum(torch.argmax(out, 1) == torch.argmax(target, 1)).item() / data.size(0))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()