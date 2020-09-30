
import torch.nn as nn
import torch

mse = nn.MSELoss()

class ind_lstm(nn.Module):
    def __init__(self, block_size, hidden_size, signal_size):
        super().__init__()
        self.block_size = block_size
        self.hidden_size = hidden_size
        self.signal_size = signal_size
        self.encoder_input_size = self.block_size 
        self.encoder_output_size = self.signal_size
        self.decoder_input_size = self.signal_size
        self.decoder_output_size = self.block_size
        
        self.encoder = nn.LSTM(
            input_size=self.encoder_input_size, 
            hidden_size=self.hidden_size,
            )
        self.decoder = nn.LSTM(
            input_size=self.decoder_input_size, 
            hidden_size=self.hidden_size
            )

        self.signal = None
        self.inputs = None
        self.decoded = None

        self.lam = 0.1
        
    def encode(self,inputs):
        self.inputs = torch.cat(inputs).view( -1, 1, self.encoder_input_size)
        # hidden = torch.zeros(1,1,self.hidden_size)
        # sig = torch.zeros(1,1,self.signal_size)
        
#        for inp in self.inputs:
#            sig, hidden = self.encoder(inp.view(1,1,-1),hidden)
        self.signal,_ = self.encoder(self.inputs)
        return self.signal

    def decode(self):
        decode_vec = torch.cat((self.inputs,self.signal[1:]))
        return self.decoded


    def forward(self, inputs):

        self.encode(inputs)
        self.decode()
        
        return self.loss()


    def loss(self):
        if not len(self.inputs): 
            return 0
        mse_loss = sum(mse(x,y) for x,y in zip(self.inputs[1:], self.decoded))
        l1_reg_loss = sum(torch.norm(s) for s in self.signal)
        loss = mse_loss + self.lam * l1_reg_loss
        return loss




