import time

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import tqdm

from dataset import BirdClefDataset, collate_fn
from model import Network
from utils import AverageMeter


def evaluate(model, loader):
    print("\nStart validation")
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    for i, (data, target) in enumerate(tqdm.tqdm(loader)):
        data = data.cuda()
        with torch.no_grad():
            out = model(data)
            target = target.cuda()
            loss = criterion(out, target)
        accuracy = torch.sum(torch.argmax(out, 1) == torch.argmax(target, 1))
        loss_meter.update(loss.item())
        acc_meter.update(accuracy.item())
        print(loss.item())
        print(accuracy.item() / data.size(0))

    print(f"Validation loss {loss_meter.avg}")
    print(f"Validation accuracy {acc_meter.avg}")


def train(model, criterion, optimizer, train_loader, val_loader, epochs):
    print("Start training")
    for epoch in range(1, epochs + 1):
        start_epoch_time = time.time()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        for i, (data, target) in enumerate(tqdm.tqdm(train_loader)):
            data = data.cuda()
            out = model(data)
            target = target.cuda()
            loss = criterion(out, target)
            accuracy = torch.sum(torch.argmax(out, 1) == torch.argmax(target, 1)) / data.size(0)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_meter.update(loss.item())
            acc_meter.update(accuracy.item())
            print(f" Loss: {loss_meter.val:.5f} Acc: {acc_meter.val:.5f} Acc mean: {acc_meter.avg:.5f}")
        end_epoch_time = time.time()
        print(f"Epoch loss {loss_meter.avg}")
        print(f"Epoch accuracy (train) {acc_meter.avg}")
        print(f"{epoch} Epoch time (m): ", (end_epoch_time - start_epoch_time) / 60)

        evaluate(model, val_loader)


if __name__ == '__main__':
    dataset = BirdClefDataset()
    loader = DataLoader(dataset, collate_fn=collate_fn,
                        batch_size=32, num_workers=4,
                        sampler=torch.utils.data.RandomSampler(dataset))
    model = Network(backbone='resnest50d', num_classes=397)
    model.cuda()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    train(model, criterion, optimizer, loader, loader, 7)
