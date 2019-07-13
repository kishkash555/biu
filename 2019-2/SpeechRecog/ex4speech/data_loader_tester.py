from gcommand_loader import GCommandLoader
import torch

dataset = GCommandLoader('./data/valid')

test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=20, shuffle=True,
        num_workers=0, pin_memory=True, sampler=None)

#test_loader = list(test_loader)

for k, (input, label) in enumerate(test_loader):
    print( label[0].dtype, label[1].dtype)
    print(input.size())
    if k == 10:
        break

