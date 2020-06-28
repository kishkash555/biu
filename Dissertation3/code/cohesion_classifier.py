import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import loader

class ann_model(nn.Module):
  #conv1d(in_channels: int, out_channels: int, kernel_size, stride: ..., padding: ..., dilation: ..., groups: int, bias: bool, padding_mode: str) -> None
  #MaxPool1d(kernel_size, stride=None, padding=0, dilation=1,
  def __init__(self):
    super(ann_model,self).__init__()
    self.conv1 = nn.Conv1d(1, 64, 21, 5, 5)
    self.conv2 = nn.Conv1d(64, 64, 21, 1, 5)
    self.pool1 = nn.MaxPool1d(2,2)

    self.conv3 = nn.Conv1d(64, 128, 5, 1, 0)
    self.conv4 = nn.Conv1d(128, 128, 5, 1, 0)
    self.pool2 = nn.MaxPool1d(2,2)

    self.conv5 = nn.Conv1d(128, 256, 6, 1, 0)
    self.conv6 = nn.Conv1d(256, 256, 6, 1, 0)
    self.pool3 = nn.AvgPool1d(3)
    self.linear1 = nn.Linear(256,1)
#    self.linear2 = nn.Linear(6,1)
 
  def forward(self,x):
    x = self.conv1(x)
    x = F.relu(x)
    x = self.conv2(x)
    x = F.relu(x)
    x = self.pool1(x)

    x = self.conv3(x)
    x = F.relu(x)
    x = self.conv4(x)
    x = F.relu(x)
    x = self.pool2(x)
    x = self.conv5(x)
    x = F.relu(x)
    x = self.conv6(x)
    x = F.relu(x)
    x = self.pool3(x).view(-1)
    x = self.linear1(x)
#    x = F.relu(x)
#    x = self.linear2(x)
    x = torch.sigmoid(x)
    return x

def train(model, epochs):
  criterion = nn.MSELoss()
  losslist = []
  #running_loss = 
  epochloss = 0.
  optimizer = optim.SGD(model.parameters(),lr=0.1,weight_decay=1e-5)
  for epoch in range(epochs):
    l = 0
    print("Entering Epoch: ",epoch)
    for id, x, y in loader.loader(None,'disk'):
      optimizer.zero_grad()
      y_hat = model(x)
      loss = criterion(y_hat.view(1),torch.Tensor([y]))

      loss.backward()

      #if l%50==0:
      #  print(y, y_hat.item(), loss.item(), model.linear1.weight.grad)
      optimizer.step()
    
#      running_loss += loss.item()
      epochloss += loss.item()
      l += 1
    losslist.append(epochloss/l)
    print("======> epoch: {}/{}, Loss:{}, l: {}\n".format(epoch,epochs,epochloss, l))
    epochloss=0



if __name__ == "__main__":
    #con = loader.connect()
    dn = ann_model()
    train(dn, 120)
    1

if 0:
    dn = ann_model()
    for i,(k, v, klass) in enumerate(loader.loader(None,'disk')):
        y = dn.forward(v)
   
        print(v.shape, y)
        if i==10:
            break