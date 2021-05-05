import time

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import tqdm
from sklearn.metrics import f1_score, accuracy_score

from dataset import BirdClefDataset, collate_fn
from model import Network
from model_stt import DeepSpeech
from utils import AverageMeter


def evaluate(model, loader):
    print("\nStart validation")
    model.eval()
    loss_meter = AverageMeter()
    pred_label = []
    target_label = []
    for i, (data, target) in enumerate(tqdm.tqdm(loader)):
        data = data.cuda()
        with torch.no_grad():
            out = model(data)
            target = target.cuda()
            loss = criterion(out, target)
        pred_label += list(torch.argmax(out, 1).cpu().numpy())
        target_label += list(torch.argmax(target, 1).cpu().numpy())
        loss_meter.update(loss.item())
        print(loss.item())

    print(f"Validation loss {loss_meter.avg}")
    print(f"Validation accuracy {accuracy_score(target_label, pred_label)}")
    print(f"Validation f1 {f1_score(target_label, pred_label, average='weighted')}")


def train(model, criterion, optimizer, scheduler, train_loader, val_loader, epochs):
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
            scheduler.step()
            loss_meter.update(loss.item())
            acc_meter.update(accuracy.item())
            print(f" Loss: {loss_meter.val:.5f} Acc: {acc_meter.val:.5f} Acc mean: {acc_meter.avg:.5f}")
        end_epoch_time = time.time()
        torch.save(model.state_dict(), f"model_{epoch}.pth")
        print(f"Epoch loss {loss_meter.avg}")
        print(f"Epoch accuracy (train) {acc_meter.avg}")
        print(f"{epoch} Epoch time (m): ", (end_epoch_time - start_epoch_time) / 60)

        evaluate(model, val_loader)


def get_model():
    model = DeepSpeech(
        rnn_hidden_size=256,
        cnn_width=256,
        nb_layers=15,
        # labels=labels,
        rnn_type='cnn_residual_repeat_sep_down8',
        # audio_conf=audio_conf,
        # bidirectional=args.bidirectional,
        # bnm=args.batch_norm_momentum,
        dropout=0.1,
        kernel_size=13,
    )
    return model


if __name__ == '__main__':
    train_dataset = BirdClefDataset(path_meta='data/train_90.csv')
    test_dataset = BirdClefDataset(path_meta='data/test_10.csv')
    loader = DataLoader(train_dataset, collate_fn=collate_fn,
                        batch_size=64, num_workers=8,
                        sampler=torch.utils.data.RandomSampler(train_dataset))
    test_loader = DataLoader(test_dataset, collate_fn=collate_fn,
                             batch_size=64, num_workers=8,
                             sampler=torch.utils.data.RandomSampler(test_dataset))
    model = Network(backbone='resnet34', num_classes=397)
    model.cuda()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)
    train(model, criterion, optimizer, scheduler, loader, test_loader, 15)
    torch.save(model.state_dict(), 'resnet34_model_last.pth')
