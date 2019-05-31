import time, os, json
import numpy as np
import matplotlib.pyplot as plt

from torch.nn import functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from pdb import set_trace as pause

from dataloader import Load_Dataset
from cnn import UNetVGG16
from loss import LossBinary
from utils import Data_View
from utils import View_Preds

X_img = './train/X_img.npy'
X_dpt = './train/X_dpt.npy'
Y = './train/Y_labels.npy'

train_dataset = Load_Dataset(X_img, X_dpt, Y, False, True)

train_loader = DataLoader(train_dataset, batch_size=3, shuffle=False, num_workers=0, drop_last=True)
model = UNetVGG16(num_classes=4)

lr = 0.0001
loss = LossBinary(jaccard_weight=0.3)
#init_optimizer = lambda lr: optim.Adam(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)
init_optimizer = lambda lr: optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)
optimizer = init_optimizer(lr)
epoch_best_loss = 1.0

for epoch in range(100) :  # loop over the dataset multiple times
    epoch_loss = 0.0
    running_loss = 0.0

    for idx, (inputs, targets) in enumerate(train_loader, 0) :

        # generate forward prediction
        outputs = model(inputs.cuda())
        
        # calculate the loss function
        loss_i = loss(outputs, targets.cuda())
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # backward + optimize
        loss_i.backward()
        optimizer.step()

        # print statistics
        running_loss += loss_i.item()
        epoch_loss += loss_i.item()

        # print every 100 mini-batches
        if idx % 100 == 0 :
            r_loss_m = running_loss/100

            print('[Epoch: {d}, Batch: {5d}] Loss: {.9f}'.format(epoch + 1, idx + 1, r_loss_m))
            
            running_loss = 0.0

    epoch_loss /= idx

    print('Epoch: {}, Global Loss: {}'.format(epoch, epoch_loss))
    
    if epoch_best_loss > epoch_loss :
        torch.save({'epoch':epoch,'epoch_loss':epoch_loss,'running_loss':running_loss,'state_dict':model.state_dict(),'optimizer':optimizer.state_dict()},'./model-best-loss-epoch-{}.pth'.format(epoch))        


torch.save({'epoch':epoch,'epoch_loss':epoch_loss,'running_loss':running_loss,'state_dict':model.state_dict(),'optimizer':optimizer.state_dict()},'./model-final-epoch-{}.pth'.format(epoch))

print('Finished Training')