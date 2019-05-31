import numpy as np
import torch, cv2, torchvision, time, os, json
from torch.utils.data import DataLoader

from dataset import SARDataset
from model import UNetVGG16
from loss import LossBinary
from torch.optim import Adam
from utils import dataview, gen_pred

from pdb import set_trace as pause

VV_train = '/media/pixforce/Armazenamento/shell/s2/spt/VV_train.npy'
VH_train = '/media/pixforce/Armazenamento/shell/s2/spt/VH_train.npy'
WIND_train = '/media/pixforce/Armazenamento/shell/s2/spt/WIND_train.npy'
OIL_train = '/media/pixforce/Armazenamento/shell/s2/spt/OIL_train.npy'
LAND_train = '/media/pixforce/Armazenamento/shell/s2/spt/LAND_train.npy' 
ND_train = '/media/pixforce/Armazenamento/shell/s2/spt/ND_train.npy'

VV_val = '/media/pixforce/Armazenamento/shell/s2/spt/VV_val.npy'
VH_val = '/media/pixforce/Armazenamento/shell/s2/spt/VH_val.npy'
WIND_val = '/media/pixforce/Armazenamento/shell/s2/spt/WIND_val.npy'
OIL_val = '/media/pixforce/Armazenamento/shell/s2/spt/OIL_val.npy'
LAND_val = '/media/pixforce/Armazenamento/shell/s2/spt/LAND_val.npy' 
ND_val = '/media/pixforce/Armazenamento/shell/s2/spt/ND_val.npy'

train_dataset = SARDataset(VV_train, VH_train, WIND_train, OIL_train, LAND_train, ND_train)
data_in = lambda data : data[..., 256:-256, 256:-256]
train_loader = DataLoader(train_dataset, batch_size=3, shuffle=False, num_workers=0, drop_last=True)

model = UNetVGG16(num_classes=4)
model.cuda()

lr = 0.0001
loss = LossBinary(jaccard_weight=0.3)
loss
init_optimizer = lambda lr: Adam(model.parameters(), lr=lr)
optimizer = init_optimizer(lr)
epoch_best_loss = 1.0

for epoch in range(65) :  # loop over the dataset multiple times
    epoch_loss = 0.0
    running_loss = 0.0

    for idx, (inputs, targets) in enumerate(train_loader, 0) :
        # data to compute
        inputs, targets = data_in(inputs), data_in(targets)

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

# #criterion = nn.CrossEntropyLoss().cuda()
# #optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)

# checkpoint = torch.load('./model-best-loss-epoch-64.pth')
# model.load_state_dict(checkpoint['state_dict'])
# #optimizer.load_state_dict(checkpoint['optimizer'])
# model.eval()
# #model.cuda()
# data_b = iter(train_loader)
# pause()

# #model.train()
# best_loss = 1 #checkpoint['running_loss']
# #epoch = checkpoint['epoch']

# epoch_loss = 0.0
# epoch_best_loss = 1

# if False :
    
#     data_in = lambda data : data[..., 256:-256, 256:-256]

#     for epoch in range(65) :  # loop over the dataset multiple times

#         running_loss = 0.0

#         for i, data in enumerate(train_loader, 0) :
#             # get the inputs; data is a list of [inputs, labels]
#             inputs, labels = data
#             inputs = data_in(inputs)
#             labels = data_in(labels)

#             # zero the parameter gradients
#             optimizer.zero_grad()

#             # forward + backward + optimize
#             outputs = model(inputs.cuda())

#             loss = criterion(outputs, labels.cuda())
#             loss.backward()
#             optimizer.step()

#             # print statistics
#             running_loss += loss.item()
#             epoch_loss += loss.item()

#             if i % 100 == 0:    # print every 100 mini-batches
#                 r_loss_m = running_loss/100

#                 print('[Epoch: {d}, Batch: {5d}] Loss: {.9f}'.format(epoch + 1, i + 1, r_loss_m))
                
#                 running_loss = 0.0

#         epoch_loss /= i

#         print('Epoch: {}, Global Loss: {}'.format(epoch, epoch_loss))
        
#         if epoch_best_loss > epoch_loss :
#             torch.save({'epoch':epoch,'running_loss':running_loss,'state_dict':model.state_dict(),'optimizer':optimizer.state_dict()},'./model-best-loss-epoch-{}.pth'.format(epoch))        


#     torch.save({'epoch':epoch,'running_loss':running_loss,'state_dict':model.state_dict(),'optimizer':optimizer.state_dict()},'./model-final-epoch-{}.pth'.format(epoch))

#     print('Finished Training')