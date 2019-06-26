import torch
from gcommand_loader import GCommandLoaderTest
import numpy as np
from ex4 import convnet
from os import path

model_fname = 'model_file_f54077'
labels_fname = 'test_y'

def main():
    net = torch.load(model_fname)
    test_set = GCommandLoaderTest('./data/test')
    
    test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=100, shuffle=None,
            num_workers=20, pin_memory=True, sampler=None)
    
    net.eval()
    
    with open(labels_fname, 'wt') as f:
        for inputs, fnames in test_loader:
            guess = torch.argmax(net.forward(inputs),dim=1).numpy()
            print(inputs.shape, guess.shape)
            result = zip(fnames,guess)
            result_str = '\n'.join(['{}, {}'.format(path.basename(f),g) for f, g in result])
            f.write(result_str+'\n')
    print('finished writing output')


if __name__ == "__main__":
    main()
