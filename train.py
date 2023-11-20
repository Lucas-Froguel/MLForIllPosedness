import torch
from torchquad import set_up_backend 

from src.dataset import CustomImageDataset, load_dataset
from src.models.model import FunctionNet
from src.models.controller import NeuralNetController
from src.models.train_net import train_one_step
from src.settings import MONGO_DATABASE_URL, MONGODB_NAME, MONGO_DATA_COLLECTION, kernels

num_kernels = len(kernels.keys())

print("Setting up device...")
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
set_up_backend("torch", data_type="float32")
generator = torch.Generator(device=device)


print("Loading dataset...")
kernel_loaders = [
    load_dataset(
        database=MONGODB_NAME, 
        collection=MONGO_DATA_COLLECTION, 
        db_url=MONGO_DATABASE_URL,
        kernel=f"x{i+1}",
        generator=generator,
        batch_size=64
    )
    for i in range(num_kernels)
]
loaders = {
    f"kernel_{i+1}": kernel_loader for i, kernel_loader in enumerate(kernel_loaders)
}


model = FunctionNet(hidden_size=60).to(device)
model_controller = NeuralNetController(
    model=model, model_path="src/models/models/simple_model.pt"
)

# I think we need a bigger lr
# It seems a better way to estimate errors is needed as well (not only RMSE) - but what?
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

EPOCHS = 2 * num_kernels

best_vloss = 1_000_000.

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch + 1))
    kernel_idx = epoch % num_kernels + 1
    kernel_name = f"kernel_{kernel_idx}"

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_step(
        epoch, 
        optimizer=optimizer, 
        data_loader=loaders[kernel_name], 
        model=model, 
        kernel=kernels[kernel_name]
    )

    model_controller.save(f"src/models/models/simple_model_epoch_{epoch}.pt")

    print(f'LOSS train {avg_loss}')

    EPOCHS += 1

model_controller.save()
