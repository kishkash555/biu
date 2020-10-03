
import torch.nn as nn
import torch

mse = nn.MSELoss()

class ind_lstm(nn.Module):
    def __init__(self, block_size, signal_size):
        super().__init__()
        self.block_size = block_size
        self.signal_size = signal_size
        # self.encoder_hidden_size = hidden_size
        # self.signal_size = signal_size
        # self.encoder_input_size = self.block_size 
        # self.encoder_output_size = self.signal_size
        # self.decoder_input_size = self.signal_size + self.block_size
        # self.decoder_output_size = self.block_size
 #       self.decoder_hidden_Siz
        
        self.encoder_input_size = self.block_size
        self.encoder_hidden_size = self.signal_size

        self.encoder = nn.LSTM(
            input_size=self.encoder_input_size, 
            hidden_size=self.encoder_hidden_size,
            )

        self.decoder = nn.LSTM(
            input_size=self.signal_size,
            hidden_size=self.block_size
        ) 

        

        self.signal = None
        self.inputs = None
        self.decoded = None

        self.lam = 0.25
        
    def encode(self,inputs):
        self.inputs = torch.cat(inputs).view( -1, 1, self.encoder_input_size)
        # hidden = torch.zeros(1,1,self.hidden_size)
        # sig = torch.zeros(1,1,self.signal_size)
        
#        for inp in self.inputs:
#            sig, hidden = self.encoder(inp.view(1,1,-1),hidden)
        self.signal,_ = self.encoder(self.inputs)
        return self.signal

    def decode(self):
        state = torch.zeros(1,1,self.block_size)
        decoded_blocks = []
        for i in range(self.inputs.shape[0]-1):
            _, (decoded_block, state) = self.decoder(
                self.signal[i+1:i+2,:,:], # the encoded vector
                (self.inputs[i:i+1,:,:], # the block
                state) # propagate internal state
                ) 
            decoded_blocks.append(decoded_block)
        self.decoded = torch.cat(decoded_blocks) 
        return self.decoded


    def forward(self, inputs):

        self.encode(inputs)
        self.decode()
        
        return self.loss()


    def loss(self):
        if not len(self.inputs): 
            return 0
        outcome = self.decoded
        target = self.inputs[1:,:,:]
        mse_loss = mse(outcome, target)
        l1_reg_loss = torch.norm(self.signal,1)/self.signal.nelement()
        loss = mse_loss + self.lam * l1_reg_loss
        return loss, mse_loss.item(), l1_reg_loss.item()




