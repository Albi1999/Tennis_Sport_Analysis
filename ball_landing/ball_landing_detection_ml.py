import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2 
import os 
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import sys 
import pickle 
sys.path.append('../')



class BounceCNN(nn.Module):
    def __init__(self):
        super(BounceCNN, self).__init__()


        
        # 1st block
        self.conv1 = nn.Conv2d(1, 8, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(8)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 2nd
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 3rd
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 4th
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        

       
        # Adaptive pooling to ensure fixed size regardless of input dimensions 
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # FCNN
        self.fc1 = nn.Linear(64, 16)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(16, 1)  # Binary output
        
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
        
        return x



class BounceDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
      #  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
        
        if self.transform:
            image = self.transform(image)
            
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label

    
def train_model(model, train_loader, val_loader, num_epochs=20, factor = 4.0, patience=7):
    print("Training Model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Define loss and optimizer
    pos_weight = torch.tensor(factor).to(device)  # Weighted loss for class imbalance (calculated on train set)
  #  criterion = nn.BCEWithLogitsLoss()
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
  #  optimizer = optim.Adam(model.parameters(), lr=0.01)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    best_val_loss = float('inf')
    early_stopping_counter = 0  # For early stopping
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).float() 
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = correct / total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.inference_mode():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = correct / total
        
        # Update learning rate
        scheduler.step(val_loss)  # Reduce LR when validation loss stagnates
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'models/best_bounce_model.pth')  # Save best model
            early_stopping_counter = 0  # Reset counter
        else:
            early_stopping_counter += 1
        
        if early_stopping_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs!")
            break  # Stop training if no improvement
    
    return model

def compute_mean_std(dataset):
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    mean = 0.0
    std = 0.0
    total_samples = 0

    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, -1)
        mean += images.mean(1).sum().item()
        std += images.std(1).sum().item()
        total_samples += batch_samples

    mean /= total_samples
    std /= total_samples
    return torch.tensor([mean]), torch.tensor([std])

def evaluate_model(model, test_loader):
    print("Evaluating Model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    all_predictions = []
    all_labels = []
    
    with torch.inference_mode():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).squeeze()
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Convert lists to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)
    f1_beta = fbeta_score(all_labels, all_predictions, beta=2) # F2 score to give more weight to recall
    
    print(f"Test Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"F2 Score (Recall-oriented): {f1_beta:.4f}")
    
    # Create and plot confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)

    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Bounce', 'Bounce'],
                yticklabels=['No Bounce', 'Bounce'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    # Save the plot
    os.makedirs('output/cnn_training', exist_ok=True)
    plt.savefig('output/cnn_training/confusion_matrix.png')
    plt.show()
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'f1_beta': f1_beta,
        'confusion_matrix': cm
    }

def make_prediction(model, best_model_path, input_frames_directory, transform, device):
    """Uses BounceDataset for consistent processing"""
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device)
    model.eval()

    import re



    
    # Get sorted list of image paths
    directory_files = sorted(os.listdir(input_frames_directory))
    img_paths = [os.path.join(input_frames_directory, file) for file in directory_files]
    img_idxs = []
    # Pattern to extract the number before .jpg
    pattern = r'(\d+)\.jpg$'

    for i in img_paths:
        # Search for the pattern in the filename
        match = re.search(pattern, str(i))

        if match:
            number = match.group(1)
            img_idxs.append(number)
    
    # Create dummy labels
    dummy_labels = [0] * len(img_paths)
    
    # Use your existing dataset class
    dataset = BounceDataset(image_paths=img_paths, labels=dummy_labels, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    predictions = []
    confidences = []
    
    with torch.inference_mode():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs).squeeze()
            
            confidence = torch.sigmoid(outputs).item()
            predicted_label = 1 if confidence > 0.5 else 0
            
            predictions.append(predicted_label)
            confidences.append(confidence)
    
    return predictions, confidences, img_idxs


# TODO : need it in the final .py file (main.py) , so we have to down below clean up the code a bit (put into a utils file the transformations, mean calculation etc.)
def evaluation_transform(mean, std):

    eval_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),  
        transforms.ToTensor(),  
        transforms.Normalize(mean=mean[0].item(), std=std[0].item())
    ])

    return eval_transform





        
if __name__ == "__main__":
    
    print("Loading Data...")

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
    
    # Define transformations
    # Define transformations (WITHOUT Normalization)
    transform_no_norm = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),  # Resize to fixed input size
        transforms.ToTensor(),  # Convert to tensor
    ])

    # Create Dataset Object without Normalization
    train_dataset_no_norm = BounceDataset(
        image_paths=image_paths_train,
        labels=labels_train,
        transform=transform_no_norm  # No normalization applied here
    )

    # Compute Mean & Std (on the training dataset)
    mean, std = compute_mean_std(train_dataset_no_norm)
    print(f"Dataset Mean: {mean}, Std: {std}")

    data_mean_std = [mean, std]

    with open('data_bounce_stubs/data_mean_std.pkl', 'wb') as f:
            pickle.dump(data_mean_std, f)



    # Define transformations with Correct Mean & Std
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),  
        transforms.RandomHorizontalFlip(p=0.5),  
        transforms.RandomRotation(10),  
     #   transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  
        transforms.ToTensor(),  
        transforms.Normalize(mean=mean[0].item(), std=std[0].item())  # Apply computed values (just first value, as we have greyscale)
    ])


    eval_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),  
        transforms.ToTensor(),  
        transforms.Normalize(mean=mean[0].item(), std=std[0].item())
    ])

    # Create datasets
    train_dataset = BounceDataset(
        image_paths= image_paths_train,
        labels= labels_train,
        transform=transform
    )
    
    val_dataset = BounceDataset(
        image_paths=image_paths_val,
        labels=labels_val,
        transform=eval_transform
    )
    
    test_dataset = BounceDataset(
        image_paths=image_paths_test,
        labels=labels_test,
        transform=eval_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    test_loader = DataLoader(test_dataset, batch_size=16)

    # Calculate the imbalance factor between bounce & no bounce in the train set

    factor = len(no_bounce_train)/len(bounce_train)


    # Initialize and train model
    model = BounceCNN()
    trained_model = train_model(model, train_loader, val_loader, num_epochs = 50, factor = factor) # Comment this line if you want to skip training
    
    # Load best model
    model.load_state_dict(torch.load('models/best_bounce_model.pth'))
    metrics = evaluate_model(model, test_loader)
    
    
    

    