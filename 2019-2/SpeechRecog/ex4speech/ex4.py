from cer import cer
from datetime import timedelta
from gcommand_loader import GCommandLoader, MAX_WORD_LEN
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

#import gitutils.gitutils as gu

def time2str(st):
    return str(timedelta(seconds=round(st)))


criterion = nn.CTCLoss()

BATCH_SIZE = 100
IN_CHANNELS = 161
SIGNAL_LENGTH = 101

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
    input_size = (1, SIGNAL_LENGTH, IN_CHANNELS) # ignoring the batch dimension
    in_channels = 1
    out_channels = 12
    kernel_size = 8
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
    out_channels = 7
    kernel_size = 4
    stride = 2
    padding = 1
cv2.output_size = conv_output_size(cv2)

print("cv2 output: {} ({})".format(cv2.output_size, mult(cv2.output_size)))


sequence_lengths = torch.full(size=(BATCH_SIZE,), fill_value = cv2.output_size[1], dtype=torch.long)

print("sequence length: {}".format(sequence_lengths[0]))


class lstm1:
    input_size = IN_CHANNELS
    seq_len = SIGNAL_LENGTH
    hidden_size = 80
    num_layers = 1
    batch_first = True
    bidi = True
    c0 = torch.zeros(num_layers * (2 if bidi else 1), BATCH_SIZE, hidden_size)
    h0 = torch.zeros(num_layers * (2 if bidi else 1), BATCH_SIZE, hidden_size)
    output_size = hidden_size * (2 if bidi else 1)

#print("lstm1 input: {}, sequence length: {}".format(lstm1.input_size, lstm1.seq_len))

class fc1:
    input_size = cv2.output_size[0]*cv2.output_size[2]
    output_size = 100

print("fc1 input {}".format(fc1.input_size))

class fc2:
    input_size = fc1.output_size 
    output_size = None


class convnet(nn.Module):
    def __init__(self, n_chars=None, min_acc=0.75, epochs=20, logging_interval=50, save_fname='model_file'):
        super().__init__()
        
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.conv1 = nn.Conv2d(cv1.in_channels, cv1.out_channels, cv1.kernel_size, cv1.stride, cv1.padding)
        self.pool1 = nn.MaxPool2d(pl1.kernel_size)
        self.conv2 = nn.Conv2d(cv2.in_channels, cv2.out_channels, cv2.kernel_size, cv2.stride, cv2.padding)

        """
        self.rnn = nn.LSTM(input_size=lstm1.input_size, 
            hidden_size=lstm1.hidden_size, 
            num_layers=lstm1.num_layers, 
            batch_first=lstm1.batch_first, 
            bidirectional=lstm1.bidi)
        """

        self.fc1 = nn.Linear(fc1.input_size, fc1.output_size)
        self.fc2 = nn.Linear(fc2.input_size, n_chars)
        
        self.revision = "0.0.1" #gu.get_sha()
        self.options = {
            'min_acc': min_acc,
            'epochs': epochs,
            'logging_interval': logging_interval,
            'save_fname': save_fname + '_' + self.revision[:6]
        }

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)        
        x = F.relu(self.conv2(x))
        
        x = torch.transpose(x,1,2)
#        x = x.reshape(BATCH_SIZE, lstm1.seq_len, lstm1.input_size)
        x = x.reshape(BATCH_SIZE, -1, fc1.input_size)

        # x, _ = self.rnn(x, (lstm1.h0, lstm1.c0))
        # output: torch.Size([100, 9, 20])
        
        x = self.fc1(F.relu(x))
        x = self.fc2(F.relu(x))
        char_seq = F.log_softmax(x, 2)

        return char_seq
            

#        print("output: {}\nhn: {}\ncn: {}".format(output.size(),hn.size(), cn.size()))


    def perform_training(self, trainloader, validloader, class_to_idx):
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=0.01)
        cum_cer_error = 0.
        batch_count = 0

        ep = self.options['epochs']
        log_interval = self.options['logging_interval']
        for epoch in range(ep):
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # unpack the batch. labels: BATCH_SIZE*MAX_WORD_LEN. word_lengths: BATCH_SIZE*1
                inputs, (labels, word_lengths) = data
                #print("label: {}".format(labels[0]))

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self(inputs)
                guess = torch.argmax(outputs,2)
                cer_error = calc_cer(guess, labels, word_lengths, class_to_idx)
                # print("guess size: {}, outputs: {}, label size: {}".format(guess.size(), outputs.size(), labels.size())) 
                cum_cer_error += cer_error
                batch_count += 1           
                loss = criterion(torch.transpose(outputs,0,1), labels, sequence_lengths, word_lengths)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if i % log_interval == log_interval -1:    
                    print('{} [{}, {:5}] loss: {:.3f} cer: {:.2%}'.format(
                        time2str(time.time()-start),
                        epoch + 1, i + 1, 
                        running_loss / log_interval,
                        cum_cer_error / batch_count
                        ), flush=True)
                    running_loss = 0.
                    cum_cer_error = 0.
                    batch_count = 0

        print('Finished Training')


    def save(self, fname):
        torch.save(self,fname)

def calc_cer(guess, labels, word_lengths, class_to_idx):
    idx_to_class = list(class_to_idx.keys())
    
    label_words = [
        ''.join(
            idx_to_class[labels[i,c]] 
            for c in range(word_lengths[i])
            )
        for i in range(BATCH_SIZE)
        ]

    cers = []

    len_guesses = 0
    for i in range(BATCH_SIZE):
        last_char = 0
        guess_word = []
        for c in range(guess.shape[1]):
            if guess[i,c] != last_char and guess[i,c] != 0:
                guess_word.append(idx_to_class[guess[i,c]])
            last_char = guess[i,c]
        len_guesses += len(guess_word)
        cers.append(cer(''.join(guess_word),label_words[i]))
    # if random.random() < 0.1: print(len_guesses)
    m = torch.mean(torch.Tensor(cers))
    return m



        


def main():
    train_path='./data/train'

    train_set = GCommandLoader(train_path)
    print("train_path: {}".format(train_path))
    valid_set = GCommandLoader('./data/valid')
    
    train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=10, pin_memory=False, sampler=None, drop_last=True)
    
    train_loader = list(train_loader)
    valid_loader = torch.utils.data.DataLoader(
            valid_set, batch_size=100, shuffle=None,
            num_workers=0, pin_memory=False, 
            sampler=None )
    
    net = convnet(train_set.n_chars)
    print("n chars {}".format(train_set.n_chars))
    net.perform_training(train_loader,valid_loader, train_set.class_to_idx)

if __name__ == "__main__":
    start = time.time()
    main()


'''
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
'''
