import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2 
import os 


class BounceCNN(nn.Module):
    def __init__(self):
        super(BounceCNN, self).__init__()


        # TODO :maybe avg pooling due to finding patterns ? 
        # 1st block
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 2nd
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 3rd
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 4th
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        

        # TODO : do we go for fixed input sizes ?
        # Adaptive pooling to ensure fixed size regardless of input dimensions
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # FCNN
        self.fc1 = nn.Linear(256, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 1)  # Binary output
        
    def forward(self, x):
        # Apply convolutional blocks
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        
        # Global pooling and flatten
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return torch.sigmoid(x)  # Sigmoid for binary classification



class BounceDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
        
        if self.transform:
            image = self.transform(image)
            
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label

    
def train_model(model, train_loader, val_loader, num_epochs=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Define loss and optimizer
    criterion = nn.BCELoss() # TODO : ADD WEIGHTING FOR CLASS IMBALANCE (if we still have it)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predicted = (outputs > 0.5).float() 
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = correct / total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = correct / total
        
        # Update learning rate
        scheduler.step(val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'models/best_bounce_model.pth')
    
    return model



if __name__ == "__main__":
    # Define transformations
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)), # TODO : do we want to resize or do we lose too much information ? check !
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #TODO : Replace with mean/std of our data (? how to find that)
    ])


    bounce_circles_train_dir = 'data/trajectory_model_dataset/circles/train/bounce'
    no_bounce_circles_train_dir = 'data/trajectory_model_dataset/circles/train/no_bounce'

    bounce_circles_val_dir = 'data/trajectory_model_dataset/circles/val/bounce'
    no_bounce_circles_val_dir = 'data/trajectory_model_dataset/circles/val/no_bounce'

    bounce_circles_test_dir = 'data/trajectory_model_dataset/circles/test/bounce'
    no_bounce_circles_test_dir = 'data/trajectory_model_dataset/circles/test/no_bounce'

    # Get file names
    bounce_train_files = os.listdir(bounce_circles_train_dir)
    no_bounce_train_files = os.listdir(no_bounce_circles_train_dir)

    bounce_val_files = os.listdir(bounce_circles_val_dir)
    no_bounce_val_files = os.listdir(no_bounce_circles_val_dir)

    bounce_test_files = os.listdir(bounce_circles_test_dir)
    no_bounce_test_files = os.listdir(no_bounce_circles_test_dir)

    # Create full paths for each file
    bounce_train = [os.path.join(bounce_circles_train_dir, file) for file in bounce_train_files]
    no_bounce_train = [os.path.join(no_bounce_circles_train_dir, file) for file in no_bounce_train_files]

    bounce_val = [os.path.join(bounce_circles_val_dir, file) for file in bounce_val_files]
    no_bounce_val = [os.path.join(no_bounce_circles_val_dir, file) for file in no_bounce_val_files]

    bounce_test = [os.path.join(bounce_circles_test_dir, file) for file in bounce_test_files]
    no_bounce_test = [os.path.join(no_bounce_circles_test_dir, file) for file in no_bounce_test_files]


    image_paths_train = bounce_train + no_bounce_train
    labels_train = [1 for x in range(len(bounce_train))] + [0 for x in range(len(no_bounce_train))]


    image_paths_val = bounce_val + no_bounce_val
    labels_val = [1 for x in range(len(bounce_val))] + [0 for x in range(len(no_bounce_val))]


    image_paths_test = bounce_test + no_bounce_test
    labels_test = [1 for x in range(len(bounce_test))] + [0 for x in range(len(no_bounce_test))]
    
    # Create datasets
    train_dataset = BounceDataset(
        image_paths= image_paths_train,
        labels= labels_train,
        transform=transform
    )
    
    val_dataset = BounceDataset(
        image_paths=image_paths_val,
        labels=labels_val,
        transform=transform
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Initialize and train model
    model = BounceCNN()
    trained_model = train_model(model, train_loader, val_loader)