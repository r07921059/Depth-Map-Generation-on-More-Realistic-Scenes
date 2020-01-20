import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from dataset import ImageDataset
from models.classifier import ImageClassifier
from torch.utils.data import DataLoader


def main():
    syn_path = './data/Synthetic'
    real_path = './data/Real'
    
    n_batch = 4
    num_epochs = 20
    learning_rate = 0.005
    train_dataset = ImageDataset(syn_path, real_path)
    train_loader = DataLoader(train_dataset, batch_size=n_batch, shuffle=True, num_workers=12)
    valid_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=12)

    classifier = ImageClassifier()
    classifier = classifier.cuda()
    loss_fn = nn.CrossEntropyLoss()
    loss_fn = loss_fn.cuda()
    optimizer = torch.optim.Adam([{'params':classifier.classifier.parameters()}], lr=learning_rate, weight_decay=1e-4)

    for epoch in range(num_epochs):
        classifier.train()
        for i, (images, label) in enumerate(train_loader):
            label = torch.LongTensor(label)
            images, label = images.cuda(), label.cuda()
            pred = classifier(images)
            # print(pred.size(),label.size())
            loss = loss_fn(pred, label)
            print('eopch:', epoch,i + 1, loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            classifier.eval()
            correct=0
            total=0
            for i, (images, label) in enumerate(valid_loader):
                label = torch.LongTensor(label)
                images, label = images.cuda(), label.cuda()
                pred = classifier(images)
                pred = pred.argmax(dim=1, keepdim=True).cpu()
                print(pred, label)
                correct += pred.eq(label.view_as(pred).cpu()).cpu().sum()
                total += pred.size(0)
            accuracy = float(correct)/total
            print(accuracy)
            best_acc = 0
            if(accuracy>=best_acc):
                best_acc = accuracy
                best_epoch = epoch
                torch.save(classifier.state_dict(), './classifier.pth')


if __name__=='__main__':
	main()
