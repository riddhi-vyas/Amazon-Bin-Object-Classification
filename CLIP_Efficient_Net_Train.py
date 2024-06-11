"""
Using CLIP embeddings from the model, and then using efficient net based CNN to train a ensemble module that 
combines clip embeddings + efficient net 

Training script 
"""


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from efficientnet_pytorch import EfficientNet
import torchvision.transforms as transforms
import pickle 
import pandas as pd 
from sklearn.model_selection import train_test_split
import os 
from PIL import Image

root_path = '/home/riddhi/Desktop/riddhi_workplace/GWAR_project'

# Data prepare 
feature_file = f"{root_path}/processed_image_features.pkl"
data_file = f"{root_path}/clean_dataset.csv"
image_dir = f"{root_path}/processed_images"

with open(feature_file, 'rb') as f:
    img_data = pickle.load(f)
    
clip_embeddings = img_data['image_feature']
img_files = img_data['files']
file_index_map = {file: idx for idx, file in enumerate(img_files)}
df = pd.read_csv(data_file)
df = df[['filename', 'quantity']]


# Split data into train+val and test
train_val, test = train_test_split(df, test_size=0.1, random_state=42)

# Split train+val into train and val
train, val = train_test_split(train_val, test_size=0.11, random_state=42)  # 0.11 x 0.9 ≈ 0.1 of the original data

# Generate train, test and validation tensors
clip_embedding_dim = 512

train_x, train_y, train_clip_embeddings = [], [], []
val_x, val_y, val_clip_embeddings = [], [], []
test_x, test_y, test_clip_embeddings = [], [], []


# Define image transformations
image_size = 224  # Adjust this based on the input size required by your model
mean = [0.485, 0.456, 0.406]  # ImageNet mean values
std = [0.229, 0.224, 0.225]  # ImageNet standard deviation values

image_transforms = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

for _, row in val.iterrows():
    image_path = os.path.join(image_dir, row['filename'])
    image = Image.open(image_path).convert('RGB')
    image = image_transforms(image)
    val_x.append(image)
    embedding_idx = file_index_map[row['filename']]
    val_clip_embeddings.append(clip_embeddings[embedding_idx, :])
    val_y.append(row['quantity'])

val_x = torch.stack(val_x)
val_y = torch.tensor(val_y, dtype=torch.long)
val_clip_embeddings = torch.stack(val_clip_embeddings)


for _, row in train.iterrows():
    image_path = os.path.join(image_dir, row['filename'])
    image = Image.open(image_path).convert('RGB')
    image = image_transforms(image)
    train_x.append(image)
    embedding_idx = file_index_map[row['filename']]
    train_clip_embeddings.append(clip_embeddings[embedding_idx, :])
    train_y.append(row['quantity'])

train_x = torch.stack(train_x)
train_y = torch.tensor(train_y, dtype=torch.long)
train_clip_embeddings = torch.stack(train_clip_embeddings)


for _, row in test.iterrows():
    image_path = os.path.join(image_dir, row['filename'])
    image = Image.open(image_path).convert('RGB')
    image = image_transforms(image)
    test_x.append(image)
    embedding_idx = file_index_map[row['filename']]
    test_clip_embeddings.append(clip_embeddings[embedding_idx, :])
    test_y.append(row['quantity'])

test_x = torch.stack(test_x)
test_y = torch.tensor(test_y, dtype=torch.long)
test_clip_embeddings = torch.stack(test_clip_embeddings)


#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define the model
class ObjectCountingModel(nn.Module):
    def __init__(self, num_classes, clip_embedding_dim=512, efficient_net_model='efficientnet-b0'):
        super(ObjectCountingModel, self).__init__()
        self.efficient_net = EfficientNet.from_pretrained(efficient_net_model)
        self.clip_embedding_branch = nn.Sequential(
            nn.Linear(clip_embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Get the output feature map size from the EfficientNet model
        dummy_input = torch.zeros(1, 3, 224, 224)
        efficient_net_feature_dim = self.efficient_net.extract_features(dummy_input).view(1, -1).shape[1]
        
        fusion_input_dim = efficient_net_feature_dim + 256
        fusion_output_dim = 512
        
        self.fusion = nn.Linear(fusion_input_dim, fusion_output_dim)
        self.classifier = nn.Linear(fusion_output_dim, num_classes)

    def forward(self, images, clip_embeddings):
        image_features = self.efficient_net.extract_features(images)
        image_features = image_features.view(image_features.size(0), -1)  # Flatten the image features
        clip_features = self.clip_embedding_branch(clip_embeddings)
        fused_features = torch.cat((image_features, clip_features), dim=1)
        fused_features = self.fusion(fused_features)
        logits = self.classifier(fused_features)
        return logits

# Instantiate the model
num_classes = 6  # Number of object count classes (0, 1, 2, 3, 4, 5)
clip_embedding_dim = 512  # CLIP embedding dimensions
model = ObjectCountingModel(num_classes, clip_embedding_dim)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Data Loaders 
train_dataset = TensorDataset(train_x, train_clip_embeddings, train_y)
val_dataset = TensorDataset(val_x, val_clip_embeddings, val_y)
test_dataset = TensorDataset(test_x, test_clip_embeddings, test_y)

batch_size = 1024
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Specify the target accuracy threshold
target_accuracy = 35.0
best_model_path = 'best_model'
best_accuracy = 0.0

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, clip_embeddings, labels in train_loader:
        optimizer.zero_grad()
        #inputs = inputs.to(device)
        #clip_embeddings = clip_embeddings.to(device)
        #lables = labels.to(device)
        outputs = model(inputs, clip_embeddings)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        print(f"Loss : {running_loss}")
        
    epoch_loss = running_loss / len(train_loader)
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}')

    # Evaluate on validation set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, clip_embeddings, labels in val_loader:
            outputs = model(inputs, clip_embeddings)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = 100 * correct / total
    print(f'Validation Accuracy: {val_accuracy:.2f}%')

    
    # Save the model if the validation accuracy is higher than the target and best so far
    if val_accuracy > target_accuracy:
        print(f'Saving model with validation accuracy {val_accuracy:.2f}%')
        torch.save(model.state_dict(), f"{best_model_path}_{val_accuracy}.pth")
    
    
# Evaluate on test set
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, clip_embeddings, labels in test_loader:
        outputs = model(inputs, clip_embeddings)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = 100 * correct / total
print(f'Test Accuracy: {test_accuracy:.2f}%')