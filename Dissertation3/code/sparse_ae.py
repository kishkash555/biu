import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import loader

class ann_model(nn.Module):
  #conv1d(in_channels: int, out_channels: int, kernel_size, stride: ..., padding: ..., dilation: ..., groups: int, bias: bool, padding_mode: str) -> None
  #MaxPool1d(kernel_size, stride=None, padding=0, dilation=1,

  # torch.nn.ConvTranspose1d(in_channels: int, out_channels: int, kernel_size, 
  # stride, padding, output_padding, groups, bias= True, dilation= 1, padding_mode = 'zeros')
  def __init__(self):
    super(ann_model,self).__init__()
    self.convE1 = nn.Conv1d(1, 20, 40, 36, 18)
    self.convE2 = nn.Conv1d(20, 40, 4, 3, 2)
    
    self.convD1 = nn.ConvTranspose1d(40,20,4,2)
    self.convD2 = nn.ConvTranspose1d(20,1,40,36, 18)
    
 
  def forward(self,x):
    # 1, 1, 400
    x = self.convE1(x) # 1 -> 64, ----21, ^$+5, ...5
    # 1, 20, 12
    x = F.relu(x)
    # 1, 20, 12
    x = self.convE2(x) # 64 -> 64, ----21, ^$+5, ...1
    # 1, 40, 5
    x = F.relu(x)
    encoded = x

    # 1, 40, 5
    x = self.convD1(x) # 64 -> 64, ----21, ^$+5, ...1
    # 1, 20, 12
    x = F.relu(x)
    # 1, 20, 12
    x = self.convD2(x)
    # 1, 1, 400

    decoded = x
    return encoded, decoded


def train(model, epochs):
  criterion = nn.MSELoss()
  losslist = []
  #running_loss = 
  epochloss = 0.
  optimizer = optim.Adam(model.parameters(),lr=0.0001,weight_decay=0.001)
  data = loader.loader(None, 'disk')

  for epoch in range(epochs):
    l = 0
    print("Entering Epoch: ",epoch)
    for id, x, y in data.generator(training=True):
      optimizer.zero_grad()
      encoded, decoded = model(x)
      loss = criterion(decoded,x)  + 0.01*torch.abs(torch.norm(decoded)-torch.norm(x))
      if torch.isnan(loss):
          print('a')
      loss.backward()

      #if l%50==0:
      #  print(y, y_hat.item(), loss.item(), model.linear1.weight.grad)
      optimizer.step()
    
#      running_loss += loss.item()
      epochloss += loss.item()
      l += 1
    losslist.append(epochloss/l)
    print("\n======> epoch: {}/{}, Loss:{:.4f}, l: {}".format(epoch,epochs,epochloss/l, l))
    print("\tlast loss: {:.4f} + {:.4f}".format(criterion(decoded,x),  0.01*torch.abs(torch.norm(decoded)-torch.norm(x))))
    epochloss=0

    if 0:
      validation_loss=0.
      m = 0
      for id, x, y in data.generator(training=False):
        m += 1
        with torch.no_grad():
          y_hat = model(x)
          loss = criterion(y_hat.view(1), torch.Tensor([y]))
          validation_loss += loss.item()
      print("-------> Validation Loss: {:.6f}, l: {}".format(100*validation_loss/m, m))


if __name__ == "__main__":
    #con = loader.connect()
    dn = ann_model()
    train(dn, 1200)
    1

if 0:
    dn = ann_model()
    for i,(k, v, klass) in enumerate(loader.loader(None,'disk')):
        y = dn.forward(v)
   
        print(v.shape, y)
        if i==10:
            break