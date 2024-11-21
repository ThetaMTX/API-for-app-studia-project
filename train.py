import time
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import SimpleCNN
import torch.nn as nn
import matplotlib.pyplot as plt

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print("Using GPU for training.")
else:
    print("Using CPU for training.")

# Load data
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

dataset = datasets.ImageFolder('dataset', transform=transform)
label_mapping = dataset.class_to_idx

train_size = int(0.8 * len(dataset))  # 80% for training
val_size = len(dataset) - train_size  # 20% for validation
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])


train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False, pin_memory=True)

# Initialize model
num_classes = len(dataset.classes)
model = SimpleCNN(num_classes=num_classes).to(device)  # Move model to GPU
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()


# Function to calculate accuracy
def calculate_accuracy(y_pred, y_true):
    _, predicted = torch.max(y_pred, 1)
    correct = (predicted == y_true).sum().item()
    return correct / y_true.size(0)


# Training loop with accuracy and loss tracking
def train(model, train_loader, val_loader, optimizer, criterion, num_epochs=10):
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    with open("training_output.txt", "w") as f:  # Open a file to save output results
        for epoch in range(num_epochs):
            times = [time.time()]
            model.train()
            train_loss = 0.0
            train_correct = 0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)  # Move data to GPU
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_correct += (outputs.argmax(1) == labels).sum().item()

            times.append(time.time())
            train_loss /= len(train_loader)
            train_acc = train_correct / len(train_loader.dataset)

            # Validation
            times.append(time.time())
            model.eval()
            val_loss = 0.0
            val_correct = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    val_correct += (outputs.argmax(1) == labels).sum().item()

            times.append(time.time())
            val_loss /= len(val_loader)
            val_acc = val_correct / len(val_loader.dataset)

            train_loss_history.append(train_loss)
            train_acc_history.append(train_acc)
            val_loss_history.append(val_loss)
            val_acc_history.append(val_acc)
            output_text = (
                f"Epoch [{epoch + 1}/{num_epochs}], "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}\n"
            )
            for idx, (x, y) in enumerate(zip(times[:-1], times[1:])):
                print(f"Time {idx}: {y - x} ")
            print(output_text)
            f.write(output_text)  # Write to file

    return train_loss_history, train_acc_history, val_loss_history, val_acc_history



def plot_model_training_curve(train_loss, train_acc, val_loss, val_acc, save_path="training_curves.png"):
    epochs = range(1, len(train_loss) + 1)
    plt.figure(figsize=(14, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, label='Training Accuracy')
    plt.plot(epochs, val_acc, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)  # Save the figure as PNG
    plt.show()


train_loss_history, train_acc_history, val_loss_history, val_acc_history = train(
    model, train_loader, val_loader, optimizer, criterion, num_epochs=20
)
plot_model_training_curve(train_loss_history, train_acc_history, val_loss_history, val_acc_history)

# Save the model
torch.save(model.state_dict(), 'testmodel.pth')
