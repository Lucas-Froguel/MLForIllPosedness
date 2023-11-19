import torch
from torch.utils.data import DataLoader
from torchquad import set_up_backend 

from src.dataset import CustomImageDataset
from src.models.model import FunctionNet
from src.models.controller import NeuralNetController
from src.models.train_net import train_one_step
from src.settings import MONGO_DATABASE_URL, MONGODB_NAME, MONGO_DATA_COLLECTION

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
kernel1_data = CustomImageDataset(
    kernel="x1", database=MONGODB_NAME, collection=MONGO_DATA_COLLECTION, db_url=MONGO_DATABASE_URL
)

kernel1_loader = DataLoader(kernel1_data, batch_size=16, shuffle=True, generator=generator)
train_features, train_labels = next(iter(kernel1_loader))


model = FunctionNet(hidden_size=2).to(device)
model_controller = NeuralNetController(
    model=model, model_path="src/models/models/simple1"
)

optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

EPOCHS = 5

best_vloss = 1_000_000.

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_step(
        epoch, optimizer=optimizer, data_loader=kernel1_loader, model=model
    )

    model.eval()

    print(f'LOSS train {avg_loss}')

    EPOCHS += 1

model_controller.save()
