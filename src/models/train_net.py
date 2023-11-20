
import torch
from torch.nn import MSELoss, L1Loss
from src.utils import plot_function_and_model, plot_kernel_and_model


def train_one_step(
    epoch_index,  
    optimizer: torch.optim = None,
    data_loader: torch.utils.data.DataLoader = None,
    model: torch.nn = None,
    kernel=None
):
    running_loss = 0.
    last_loss = 0.

    for i, data in enumerate(data_loader):
        inputs, labels = data

        optimizer.zero_grad()

        output = model.evaluate(inputs, kernel=kernel)

        loss = L1Loss()
        mse_error = loss(output, labels)
        print(f"Loss ({i}) - {mse_error}")
        mse_error.backward()

        optimizer.step()
        if i % 100 == 0:  # maybe break here?
            plot_function_and_model(i, model=model, epoch=epoch_index)
            plot_kernel_and_model(i, model=model, kernel=kernel, epoch=epoch_index)
        
        if i == 300:
            last_loss = mse_error.item()
            break

        # Gather data and report
        running_loss += mse_error.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            running_loss = 0.
            break

    return last_loss
