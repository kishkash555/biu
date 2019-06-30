from gcommand_loader import GCommandLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gitutils.gitutils as gu
import time
from datetime import timedelta

def time2str(st):
    return str(timedelta(seconds=round(st)))


criterion = nn.CrossEntropyLoss()

IN_CHANNELS = 161
SIGNAL_LENGTH = 101
N_CLASSES = 31

def n_out(input_size, padding, kernel_size, stride, dilation):
    return int((input_size + 2*padding - dilation*(kernel_size - 1) -1)/stride +1 )

def conv_output_size(conv):
    return (
        conv.out_channels if hasattr(conv,'out_channels') else conv.input_size[0], 
        n_out(conv.input_size[1], conv.padding, conv.kernel_size, conv.stride, conv.dilation),
        n_out(conv.input_size[2], conv.padding, conv.kernel_size, conv.stride, conv.dilation)
        )

def mult(s):
    return s[0]*s[1]*s[2]

class conv_default:
    dilation = 1
    padding = 0
    stride = 1    

class pl_default:
    dilation = 1
    padding = 0

class cv1(conv_default):
    input_size = (1, SIGNAL_LENGTH, IN_CHANNELS)
    in_channels = 1
    out_channels = 32
    kernel_size = 6
    stride = 2
    padding = 1
cv1.output_size = conv_output_size(cv1)

print("cv1 output: {}".format(cv1.output_size))

class pl1(pl_default):
    input_size = cv1.output_size
    kernel_size = 2
    stride = kernel_size
pl1.output_size = conv_output_size(pl1)

print("pl1 output: {}".format(pl1.output_size))

class cv2(conv_default):
    input_size = pl1.output_size
    in_channels = pl1.output_size[0]
    out_channels = 16
    kernel_size = 6
    stride = 1
    padding = 1
cv2.output_size = conv_output_size(cv2)

print("cv2 output: {} ({})".format(cv2.output_size, mult(cv2.output_size)))


class pl2(pl_default):
    input_size = cv2.output_size
    kernel_size = 2
    stride = kernel_size
pl2.output_size = conv_output_size(pl2)

print("pl2 output: {} ({})".format(pl2.output_size, mult(pl2.output_size)))



class cv3:
    input_size = pl2.output_size
    in_channels = pl2.output_size[0]
    out_channels = 32
    kernel_size = 3
    stride = 1
    output_size = (out_channels,
        int((input_size[1] - kernel_size)/stride + 1),
        int((input_size[2] - kernel_size)/stride + 1))

# print("cv3 output: {}".format(cv3.output_size))


class pl3:
    input_size = cv3.output_size
    kernel_size = 3
    stride = kernel_size
    output_size = ( 
        input_size[0], 
        int((input_size[1] - kernel_size +1 )/stride - 1),
        int((input_size[2] - kernel_size)/stride + 1),
    )

# print("pl3 output: {}".format(pl3.output_size))


class fc1:
    input_size = mult(pl2.output_size) 
    output_size = 100

print("fc1 input {}".format(fc1.input_size))

class fc2:
    input_size = fc1.output_size
    output_size = N_CLASSES


class convnet(nn.Module):
    def __init__(self, min_acc=0.75, epochs=20, logging_interval=150, save_fname='model_file'):
        super().__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.conv1 = nn.Conv2d(cv1.in_channels, cv1.out_channels, cv1.kernel_size, cv1.stride, cv1.padding)
        self.pool1 = nn.MaxPool2d(pl1.kernel_size)
        self.bn1 = nn.BatchNorm2d(pl1.output_size[0])

        self.conv2 = nn.Conv2d(cv2.in_channels, cv2.out_channels, cv2.kernel_size, cv2.stride, cv2.padding)
        self.pool2 = nn.MaxPool2d(pl2.kernel_size)
        self.bn2 = nn.BatchNorm2d(pl2.output_size[0])
        
        #self.conv3 = nn.Conv2d(cv3.in_channels, cv3.out_channels, cv3.kernel_size)
        #self.pool3 = nn.MaxPool2d(pl3.kernel_size)
        
        self.fc1 = nn.Linear(fc1.input_size,fc1.output_size)
        self.fc2 = nn.Linear(fc2.input_size,fc2.output_size)
        
        self.revision = gu.get_sha()
        self.options = {
            'min_acc': min_acc,
            'epochs': epochs,
            'logging_interval': logging_interval,
            'save_fname': save_fname + '_' + self.revision[:6]
        }

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.bn1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.bn2(x)
        # print("after pool2 {}".format(x.shape))
        
        x = x.view(-1,fc1.input_size)

        # print("after flat {}".format(x.shape))

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


    def perform_training(self, trainloader, validloader):
        self.train()
        optimizer = optim.Adam(self.parameters())
        good = bad = 0

        ep = self.options['epochs']
        log_interval = self.options['logging_interval']
        for epoch in range(ep):
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self(inputs)
                guess = torch.argmax(outputs,1)

                # print("guess size: {}, label size: {}".format(guess.size(), labels.size()))            
                good += int(sum(guess==labels))
                bad += int(sum(guess!=labels))
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if i % log_interval == log_interval -1:    
                    valid_good = valid_bad = 0
                    self.eval()
                    for valid_input, valid_labels in validloader:
                        valid_guess = torch.argmax(self(valid_input),1)
                        valid_good += int(sum(valid_guess == valid_labels))
                        valid_bad += int(sum(valid_guess != valid_labels))
                    self.train()
                    valid_acc =  valid_good/(valid_good+valid_bad)
                    save = '*' if valid_acc > self.options['min_acc'] else ''
                    print('{} [{}, {:5}] loss: {:.3f} train acc: {}/{} ({:.1%}), valid acc:  {}/{} ({:.1%}){}'.format(
                        time2str(time.time()-start),
                        epoch + 1, i + 1, 
                        running_loss / log_interval,
                        good, good+bad, good/(good+bad),
                        valid_good, valid_good+valid_bad, valid_acc,
                        save
                        ))
                    good = bad = 0
                    running_loss = 0.0
                    if save == '*':
                        self.save(self.options['save_fname'])
                        self.options['min_acc'] = valid_acc

        print('Finished Training')


    def save(self, fname):
        torch.save(self,fname)


def main():

    train_set = GCommandLoader('./data/train')
    valid_set = GCommandLoader('./data/valid')
    
    train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=100, shuffle=True,
            num_workers=30, pin_memory=False, sampler=None)
    
    train_loader = list(train_loader)
    valid_loader = torch.utils.data.DataLoader(
            valid_set, batch_size=100, shuffle=None,
            num_workers=5, pin_memory=False, 
            sampler=None )
    
    net = convnet()
    net.perform_training(train_loader,valid_loader)

if __name__ == "__main__":
    start = time.time()
    main()
