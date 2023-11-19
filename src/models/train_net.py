
import torch
from src.models.cost_function import cost_function


def train_one_step(
    epoch_index,  
    optimizer: torch.optim = None,
    data_loader: torch.utils.data.DataLoader = None,
    model: torch.nn = None
):
    running_loss = 0.
    last_loss = 0.

    for i, data in enumerate(data_loader):
        inputs, labels = data

        optimizer.zero_grad()

        loss = cost_function(inputs, labels, model=model)
        loss.backward()

        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            running_loss = 0.

    return last_loss
