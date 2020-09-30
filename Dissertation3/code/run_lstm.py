import lstm_for_independence as lfi
import loader
import torch.optim as optim

def train_lstm():
    ld = loader.loader(loader.connect(),base_len=20)
    lstm = lfi.ind_lstm(20,20,20)
    optimizer = optim.SGD(lstm.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(20):
        for data in ld.generator(standardize=True):
            blocks = [block for (_, block, _) in data]
            optimizer.zero_grad()
            loss = lstm(blocks)
            loss.backward()
            optimizer.step()
                
if __name__ == "__main__":
    train_lstm()