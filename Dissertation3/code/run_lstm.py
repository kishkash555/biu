import lstm_for_independence as lfi
import loader
import torch.optim as optim
import torch
def train_lstm():
    ld = loader.loader(None,base_len=20)
    lstm = lfi.ind_lstm(20,200)
    optimizer = optim.Adagrad(lstm.parameters(), lr=0.1)

    for epoch in range(2000):
        total_mse = total_l1reg = 0.
        for data in ld.generator(standardize=True):
            blocks = [block for (_, block, _) in data]
            optimizer.zero_grad()
            loss, mse_loss, l1reg_loss = lstm(blocks)
            loss.backward()
            optimizer.step()
            total_mse += mse_loss
            total_l1reg += l1reg_loss
        print(epoch, total_mse, total_l1reg)
        if epoch % 50 == 0:
            print("saving")
            torch.save(lstm,'lstm_model.trch')

if __name__ == "__main__":
    train_lstm()