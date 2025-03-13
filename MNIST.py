
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Subset, random_split
import random
import copy
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLIENTS = 5
CLIENTS_PER_ROUND = 3
EPOCHS = 10
LOCAL_EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.0001


class LowRankAdapter_fc(nn.Module):
    def __init__(self, input_dim, output_dim, rank):
        super(LowRankAdapter_fc, self).__init__()
        self.low_rank1 = nn.Linear(input_dim, rank, bias=False)
        self.low_rank2 = nn.Linear(rank, output_dim, bias=False)

    def forward(self, x):
        return self.low_rank2(self.low_rank1(x))

class LoRaSiameseNetwork(nn.Module):
    def __init__(self, rank):
        super(LoRaSiameseNetwork, self).__init__()

        self.backbone = torch.hub.load('pytorch/vision', 'squeezenet1_1', pretrained=True)
        self.backbone.classifier = nn.Identity()
        for param in self.backbone.parameters():
            param.requires_grad = False


        self.fc_lora1 = LowRankAdapter_fc(25088, 256, rank=rank)
        self.fc_lora2 = LowRankAdapter_fc(256, 128, rank=rank)
        self.similarity_lora = LowRankAdapter_fc(128, 1, rank=rank)

        self.fc = nn.Sequential(
            nn.Linear(25088, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.similarity = nn.Sequential(
            nn.Linear(128, 1)
        )

    def forward_once(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.fc_lora1(x)
        x = nn.ReLU()(x)
        x = self.fc_lora2(x)
        return x

    def forward(self, img1, img2):
        output1 = self.forward_once(img1)
        output2 = self.forward_once(img2)
        diff = torch.abs(output1 - output2)
        similarity_score = self.similarity_lora(diff)
        return torch.sigmoid(similarity_score)


class SiameseDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        anchor_img, anchor_label = self.dataset[index]
        if isinstance(anchor_img, torch.Tensor):
            anchor_img = to_pil_image(anchor_img)

        if random.random() > 0.5:
            while True:
                idx = random.randint(0, len(self.dataset) - 1)
                img, label = self.dataset[idx]
                if label == anchor_label:
                    break
            pair_img, pair_label = img, 1.0
        else:
            while True:
                idx = random.randint(0, len(self.dataset) - 1)
                img, label = self.dataset[idx]
                if label != anchor_label:
                    break
            pair_img, pair_label = img, 0.0

        if isinstance(pair_img, torch.Tensor):
            pair_img = to_pil_image(pair_img)

        if self.transform:
            anchor_img = self.transform(anchor_img)
            pair_img = self.transform(pair_img)

        return anchor_img, pair_img, torch.tensor(pair_label, dtype=torch.float32)


transform = transforms.Compose([
    transforms.Grayscale(3),
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])


full_dataset = datasets.MNIST(root='./data', train=True, download=True)
val_size = int(0.1 * len(full_dataset))
test_size = int(0.2 * len(full_dataset))
train_size = len(full_dataset) - val_size - test_size
train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

val_loader = DataLoader(SiameseDataset(val_dataset, transform=transform), batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(SiameseDataset(test_dataset, transform=transform), batch_size=BATCH_SIZE, shuffle=False)

# Create IID splits for clients
def create_iid_splits(dataset, num_clients):
    indices_by_digit = {digit: [] for digit in range(10)}
    for idx, (image, label) in enumerate(dataset):
        label = label if isinstance(label, int) else label.item()
        indices_by_digit[label].append(idx)
    for digit in indices_by_digit:
        random.shuffle(indices_by_digit[digit])
    client_indices = [[] for _ in range(num_clients)]
    for digit in range(10):
        digit_indices = indices_by_digit[digit]
        total_digit = len(digit_indices)
        base_size = total_digit // num_clients
        remainder = total_digit % num_clients
        start_idx = 0
        for client_id in range(num_clients):
            extra = 1 if client_id < remainder else 0
            end_idx = start_idx + base_size + extra
            client_indices[client_id].extend(digit_indices[start_idx:end_idx])
            start_idx = end_idx
    return [Subset(dataset, idxs) for idxs in client_indices]

client_subsets = create_iid_splits(train_dataset, num_clients=NUM_CLIENTS)
client_datasets = [SiameseDataset(subset, transform=transform) for subset in client_subsets]
client_loaders = [DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True) for ds in client_datasets]

def train_local_model(model, dataloader, device, lr=LEARNING_RATE, local_epochs=LOCAL_EPOCHS):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    criterion = nn.BCELoss()
    for epoch in range(local_epochs):
        total_loss = 0.0
        for img1, img2, label in dataloader:
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            optimizer.zero_grad()
            outputs = model(img1, img2)
            loss = criterion(outputs.squeeze(), label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"    Local Epoch [{epoch + 1}/{local_epochs}], Loss: {total_loss / len(dataloader):.4f}")

def aggregate_lora_weights(client_models, global_model):
    global_dict = global_model.state_dict()
    for key in global_dict.keys():
        if "low_rank" in key:

            global_dict[key] = sum(client.state_dict()[key] for client in client_models) / len(client_models)
    global_model.load_state_dict(global_dict)

def evaluate_siamese_model(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for img1, img2, label in dataloader:
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            outputs = model(img1, img2)
            predicted = (outputs.squeeze() > 0.5).float()
            correct += (predicted == label).sum().item()
            total += label.size(0)
    return correct / total


ranks = [1, 2, 4, 8, 16, 32]
num_trials = 3

for rank in ranks:
    for trial in range(num_trials):
        print("\n" + "=" * 50)
        print(f"Starting simulation for LoRA Rank = {rank} | Trial {trial + 1}/{num_trials}")
        print("=" * 50)

        global_model = LoRaSiameseNetwork(rank=rank).to(device)

        best_global_model_weights = copy.deepcopy(global_model.state_dict())
        prev_val_accuracy = 0.0
        val_accuracies = []


        for global_round in range(EPOCHS):
            print(f"\n--- Global Round {global_round + 1}/{EPOCHS} ---")
            selected_clients = random.sample(range(NUM_CLIENTS), k=CLIENTS_PER_ROUND)
            print(f"Selected clients: {selected_clients}")
            client_models = []


            for client_id in selected_clients:
                local_model = copy.deepcopy(global_model).to(device)
                print(f"  Training on Client {client_id}...")
                train_local_model(local_model, client_loaders[client_id], device)
                client_models.append(local_model)


            if client_models:
                aggregate_lora_weights(client_models, global_model)


            val_accuracy = evaluate_siamese_model(global_model, val_loader, device)
            val_accuracies.append(val_accuracy)
            print(f"Validation Accuracy after Global Round {global_round + 1}: {val_accuracy * 100:.2f}%")


            if val_accuracy > prev_val_accuracy:
                prev_val_accuracy = val_accuracy
                best_global_model_weights = copy.deepcopy(global_model.state_dict())
            else:
                global_model.load_state_dict(best_global_model_weights)


        test_accuracy = evaluate_siamese_model(global_model, test_loader, device)
        print(f"\nFinal Test Accuracy for Rank {rank} | Trial {trial + 1}: {test_accuracy * 100:.2f}%")


        plt.figure(figsize=(8, 5))
        plt.plot(range(1, EPOCHS + 1), val_accuracies, marker='o', linestyle='-')
        plt.title(f"Validation Accuracy Over Global Rounds\n(LoRA Rank = {rank}, Trial = {trial + 1})")
        plt.xlabel("Global Round")
        plt.ylabel("Validation Accuracy")
        plt.text(EPOCHS, val_accuracies[-1],
                 f"Test Acc: {test_accuracy * 100:.2f}%", fontsize=10,
                 ha='right', va='bottom', color='blue')


        params_text = (
            f"NUM_CLIENTS={NUM_CLIENTS}, "
            f"CLIENTS_PER_ROUND={CLIENTS_PER_ROUND}, "
            f"EPOCHS={EPOCHS}, "
            f"LOCAL_EPOCHS={LOCAL_EPOCHS}, "
            f"BATCH_SIZE={BATCH_SIZE}, "
            f"LEARNING_RATE={LEARNING_RATE}, "
            f"RANK={rank}"
        )
        plt.gcf().text(0.5, 0.01, params_text, ha='center', fontsize=10)
        plt.grid(True)
        plt.show()

