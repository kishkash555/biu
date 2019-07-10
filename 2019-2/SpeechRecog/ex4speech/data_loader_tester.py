from gcommand_loader import GCommandLoader
import torch

dataset = GCommandLoader('./data/validation')

test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=100, shuffle=True,
        num_workers=0, pin_memory=True, sampler=None)

test_loader = list(test_loader)

for k, (input,label) in enumerate(test_loader):
    print(input.size(), len(label))
    if k == 100:
            break

