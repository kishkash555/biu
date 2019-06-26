from gcommand_loader import GCommandLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gitutils.gitutils as gu

criterion = nn.CrossEntropyLoss()

class convnet(nn.Module):
    def __init__(self, min_acc=0.8, epochs=20, logging_interval=50, save_fname='model_file'):
        super().__init__()
        self.conv1 = nn.Conv1d(161,40,12)
        self.pool = nn.MaxPool1d(10)
        self.fc1 = nn.Linear(360,31)
        self.revision = gu.get_sha()
        self.options = {
            'min_acc': min_acc,
            'epochs': epochs,
            'logging_interval': logging_interval,
            'save_fname': save_fname + self.revision[:6]
        }

    def forward(self, x):
#        print("input size: {}".format(x.size()))
        x = x.squeeze()
#        print("after squeeze: {}".format(x.size()))
        x = self.conv1(x)
#        print("after conv1: {}".format(x.size()))
        x = self.pool(F.relu(x))
#        print("after pool: {}".format(x.size()))
        x = x.view(-1,360)
#        print("after view: {}".format(x.size()))
        x = self.fc1(x)
#        print("final: {}".format(x.size()))
        return x


    def train(self, trainloader, validloader):
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

                # print statistics
                running_loss += loss.item()
                if i % log_interval == log_interval -1:    
                    valid_good = valid_bad = 0
                    for valid_input, valid_labels in validloader:
                        valid_guess = torch.argmax(self(valid_input),1)
                        valid_good += int(sum(valid_guess == valid_labels))
                        valid_bad += int(sum(valid_guess != valid_labels))
                    valid_acc =  valid_good/(valid_good+valid_bad)
                    print('[{}, {:5}] loss: {:.3f} train acc: {}/{} ({:.1%}), valid acc:  {}/{} ({:.1%})'.format(
                        epoch + 1, i + 1, 
                        running_loss / 50,
                        good, good+bad, good/(good+bad),
                        valid_good, valid_good+valid_bad, valid_acc
                        ))
                    good = bad = 0
                    running_loss = 0.0
                    if valid_acc > self.options['min_acc']:
                        print("saving")
                        self.save()
                        self.options['min_acc'] = valid_acc

        print('Finished Training')


        def save(self, fname):
            torch.save(self,fname)


def main():

    train_set = GCommandLoader('./data/train')
    valid_set = GCommandLoader('./data/valid')
    
    train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=100, shuffle=True,
            num_workers=20, pin_memory=True, sampler=None)
    
    valid_loader = torch.utils.data.DataLoader(
            valid_set, batch_size=100, shuffle=None,
            num_workers=20, pin_memory=True, 
            sampler=None )
    
    net = convnet()
    net.train(train_loader,valid_loader)

if __name__ == "__main__":
    main()
