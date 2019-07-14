from cer import cer
from datetime import timedelta
from gcommand_loader import GCommandLoader, GCommandLoaderTest, MAX_WORD_LEN
from os import path
import sys
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

def conv_output_size(conv): # ignores the BATCH dimension (dimension 0)
    return (
        conv.out_channels if hasattr(conv,'out_channels') else conv.input_size[0], 
        n_out(conv.input_size[1], conv.padding, conv.kernel_size, conv.stride, conv.dilation),
        n_out(conv.input_size[2], conv.padding, conv.kernel_size, conv.stride, conv.dilation)
        )


def conv2d_output_size(conv): # ignores the BATCH dimension (dimension 0)
    one_or_twoD = lambda p: p if hasattr(p,'__len__') and len(p) > 1 else (p, p)
    padding = one_or_twoD(conv.padding) 
    kernel_size = one_or_twoD(conv.kernel_size)
    stride = one_or_twoD(conv.stride)
    dilation = one_or_twoD(conv.dilation)
    return (
        conv.out_channels if hasattr(conv,'out_channels') else conv.input_size[0], # captures pooling layer use case
        n_out(conv.input_size[1], padding[0], kernel_size[0], stride[0], dilation[0]),
        n_out(conv.input_size[2], padding[1], kernel_size[1], stride[1], dilation[1])
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
    input_size = (1, IN_CHANNELS, SIGNAL_LENGTH) # ignoring the batch dimension
    in_channels = 1
    out_channels = 15
    kernel_size = 10
    stride = 1
    padding = 0
cv1.output_size = conv2d_output_size(cv1)

print("cv1 output: {}".format(cv1.output_size))

class pl1(pl_default):
    input_size = cv1.output_size
    kernel_size = (2,2)
    stride = kernel_size
pl1.output_size = conv2d_output_size(pl1)

print("pl1 output: {}".format(pl1.output_size))

class cv2(conv_default):
    input_size = pl1.output_size
    in_channels = pl1.output_size[0]
    out_channels = 1
    kernel_size = 5
    stride = 1
    padding = 0
cv2.output_size = conv2d_output_size(cv2)

print("cv2 output: {} ({})".format(cv2.output_size, mult(cv2.output_size)))


sequence_lengths = torch.full(size=(BATCH_SIZE,), fill_value = cv2.output_size[2], dtype=torch.long)

print("sequence length: {}".format(sequence_lengths[0]))


class lstm1:
    input_size = cv2.output_size[0]*cv2.output_size[1]
    seq_len = sequence_lengths[0]
    hidden_size = 72
    num_layers = 3
    batch_first = True
    bidi = True
    c0 = torch.zeros(num_layers * (2 if bidi else 1), BATCH_SIZE, hidden_size)
    h0 = torch.zeros(num_layers * (2 if bidi else 1), BATCH_SIZE, hidden_size)
    output_size = hidden_size * (2 if bidi else 1)
    dropout = 0.25

print("lstm1 input: {}, sequence length: {}".format(lstm1.input_size, lstm1.seq_len))

class fc1:
    input_size = lstm1.output_size
    output_size = 144

print("fc1 input {}".format(fc1.input_size))

class fc2:
    input_size = fc1.output_size 
    output_size = None


class convnet(nn.Module):
    def __init__(self, n_chars=None, min_acc=0.99, epochs=60, logging_interval=50, save_fname='model_file'):
        super().__init__()
        
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.conv1 = nn.Conv2d(cv1.in_channels, cv1.out_channels, cv1.kernel_size, cv1.stride, cv1.padding)
        self.pool1 = nn.MaxPool2d(pl1.kernel_size)
        self.p1dropout = nn.Dropout(p=0.25)
        self.batch_norm1 = nn.BatchNorm2d(pl1.output_size[0])

        
        self.conv2 = nn.Conv2d(cv2.in_channels, cv2.out_channels, cv2.kernel_size, cv2.stride, cv2.padding)
        self.batch_norm2 = nn.BatchNorm2d(cv2.output_size[0])
        
        self.rnn = nn.LSTM(input_size=lstm1.input_size, 
            hidden_size=lstm1.hidden_size, 
            num_layers=lstm1.num_layers, 
            batch_first=lstm1.batch_first, 
            bidirectional=lstm1.bidi,
            dropout=lstm1.dropout)
    

        self.fc1 = nn.Linear(fc1.input_size, fc1.output_size)
        #self.dofc1 = nn.Dropout(p=0.25)
        
        self.fc2 = nn.Linear(fc2.input_size, n_chars)
        #self.dofc2 = nn.Dropout(p=0.25)
        

        self.revision = "0.0.1" #gu.get_sha()
        self.options = {
            'min_acc': min_acc,
            'epochs': epochs,
            'logging_interval': logging_interval,
            'save_fname': save_fname + '_' + self.revision[:6],
            'n_chars': n_chars
        }

    def forward(self, x):
        #tanh = nn.Tanh()
        x = self.batch_norm1(self.conv1(x))
        x = F.relu(x)
        x = self.p1dropout(self.pool1(x))
        x = self.conv2(x)
        x = F.relu(self.batch_norm2(x))

        x = x.permute(0, 3, 1, 2) 
        x = x.reshape([x.shape[0], x.shape[1], -1])
        
        # x = x.squeeze(dim=1).permute(0,2,1)
        #x = x.permute(0, 2, 1, 3) 
        #x = x.reshape((lstm1.seq_len, BATCH_SIZE, lstm1.input_size))

        #print("before lstm", x.shape)
        #assert(all([a==b for a,b in zip(x.shape[1:],[lstm1.seq_len, lstm1.input_size])]))        
        x, _ = self.rnn(x)

#        assert(all([a==b for a,b in zip(x.shape[1:],[pl1.output_size[2], pl1.output_size[0]*pl1.output_size[1] ])]))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        assert(x.shape[2] == self.options['n_chars'])
 #       x = self.dofc2(self.fc2(x))
        char_seq = F.log_softmax(x, dim=2)
        return char_seq
            



    def perform_training(self, trainloader, validloader, testloader, class_to_idx):
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=0.01)
        cum_cer_error = 0.
        batch_count = 0

        ep = self.options['epochs']
        log_interval = self.options['logging_interval']
        min_cer_error = self.options['min_acc'] + 1e-5
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
                    self.eval()
                    v_cum_cer_error = 0.
                    v_batch_count = 0
                    for v_inputs, (v_labels, v_word_lengths) in validloader:
                        v_outputs = self(v_inputs)
                        guess = torch.argmax(v_outputs,2)
                        cer_error = calc_cer(guess, v_labels, v_word_lengths, class_to_idx)
                        v_cum_cer_error += cer_error
                        v_batch_count += 1
                    eval_test = '*' if v_cum_cer_error/ v_batch_count < min_cer_error else ''
                    print('{} [{}, {:5}] loss: {:.3f} cer: {:.2%} validation cer: {:.2%}{}'.format(
                        time2str(time.time()-start),
                        epoch + 1, 
                        i + 1, 
                        running_loss / log_interval,
                        cum_cer_error / batch_count,
                        v_cum_cer_error / v_batch_count,
                        eval_test
                        ))
                    print("",end='',flush=True)

                    if eval_test=='*' and testloader is not None and len(test_file):
                        min_cer_error = v_cum_cer_error/ v_batch_count
                        with open(test_file,'wt') as a:
                            for t_inputs, fnames in testloader:
                                t_outputs = self(t_inputs)
                                guess = torch.argmax(t_outputs,2)
                                for f,g in zip(fnames, generate_guess_strings(guess,class_to_idx)):
                                    test_file.write("{}, {}\n".format(path.basename(f),g))


                    running_loss = 0.
                    cum_cer_error = 0.
                    batch_count = 0
                    self.train()

        print('Finished Training')


    def save(self, fname):
        torch.save(self,fname)

def calc_cer(guess, labels, word_lengths, class_to_idx):
    batch_size = guess.shape[0]
    idx_to_class = list(class_to_idx.keys())
    
    label_words = [
        ''.join(
            idx_to_class[labels[i,c]] 
            for c in range(word_lengths[i])
            )
        for i in range(batch_size)
        ]

    cers = [
        cer(guess_word,label_words[i]) 
        for i, guess_word 
        in enumerate(generate_guess_strings(guess, class_to_idx))
        ]

    m = torch.mean(torch.Tensor(cers))
    return m

def generate_guess_strings(guess, class_to_idx):
    batch_size = guess.shape[0]
    idx_to_class = list(class_to_idx.keys())
    for i in range(batch_size):
        last_char = 0
        guess_word = []
        for c in range(guess.shape[1]):
            if guess[i,c] != last_char and guess[i,c] != 0:
                guess_word.append(idx_to_class[guess[i,c]])
            last_char = guess[i,c]
        yield ''.join(guess_word)


        


def main():
    train_path='./data/train'
    valid_set = GCommandLoader('./data/valid')
    test_set = GCommandLoaderTest('./data/test')
    train_set = GCommandLoader(train_path)

    print("train_path: {}".format(train_path))
    
    train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=10, pin_memory=False, sampler=None, drop_last=True)
    
    train_loader = list(train_loader)
    valid_loader = torch.utils.data.DataLoader(
            valid_set, batch_size=BATCH_SIZE, shuffle=None,
            num_workers=0, pin_memory=False, 
            sampler=None )
    
    test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=BATCH_SIZE, shuffle=None,
            num_workers=0, pin_memory=False, 
            sampler=None )
    net = convnet(train_set.n_chars)
    print("n chars {}".format(train_set.n_chars))
    net.perform_training(train_loader, valid_loader, test_loader, train_set.class_to_idx)

if __name__ == "__main__":
    start = time.time()
    test_file = sys.argv[1] if len(sys.argv) >= 2 else ''
    main()


