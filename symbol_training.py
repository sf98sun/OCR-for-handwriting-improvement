import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import json

DATA_DIR = 'symbol-dataset'
BATCH_SIZE = 8
EPOCHS = 35
LEARNING_RATE = 0.001

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.RandomRotation(15, fill=255), 
    # transforms.RandomResizedCrop(224, scale=(0.8, 1.0)), 
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

print("Loading dataset...")
dataset = datasets.ImageFolder(DATA_DIR, transform=data_transforms)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
class_names = dataset.classes
num_classes = len(class_names)
print(f"Classes found: {class_names}")

print("Loading MobileNetV2...")
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

for param in model.parameters():
    param.requires_grad = False

model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

if torch.cuda.is_available():
    device = torch.device("cuda")     
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu") 

print(f"Using hardware accelerator: {device}")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier[1].parameters(), lr=LEARNING_RATE)

print("Starting training...")
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.4f}")

torch.save(model.state_dict(), 'symbol_classifier.pth')


with open('class_mapping.json', 'w') as f:
    json.dump(class_names, f)
print("Training complete. Model saved to 'symbol_classifier.pth'")