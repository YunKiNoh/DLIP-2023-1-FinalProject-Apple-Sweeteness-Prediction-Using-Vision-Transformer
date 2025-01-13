import torch
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import json
from PIL import Image
import os
from PIL import ImageDraw
import numpy as np
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
from PIL.ExifTags import TAGS
from PIL import ExifTags
import timm

# Define transformations for the train set
train_transforms = transforms.Compose([
     transforms. RandomRotation(30),
     transforms. RandomHorizontalFlip(),
     transforms.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.)),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
 ])

# Define transformations for the validation set
val_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class AppleDataset(Dataset):
    def __init__(self, img_dir, json_dir):

        # serch images and labels 
        self.img_dir = img_dir
        self.json_dir = json_dir
        # self.transform = transform
        self.data_info = []
        for root, dirs, files in os.walk(json_dir):
            for file in files:
                if file.endswith('.json'):
                    json_path = os.path.join(root, file)
                    img_name = os.path.splitext(file)[0]
                    try:
                        with open(json_path, "r", encoding='utf-8') as f:
                            data = json.load(f)
                            img_file_name = os.path.splitext(data["images"]["img_file_name"])[0] + '.jpg'
                            img_path = os.path.join(self.img_dir, img_file_name)
                            if os.path.exists(img_path): # Check if the image file exists
                                self.data_info.append(data)
                            else:
                                print(f"File {img_path} does not exist. Skipping.")
                    except UnicodeDecodeError:
                        print(f'UnicodeDecodeError: Cannot decode file {json_path}. Skipping this file.')

    def __len__(self):
        return len(self. data_info)
    

    def __getitem__(self, idx):
        # get image and label
        img_file_name = self.data_info[idx]["images"]["img_file_name"]
        img_folder = self.data_info[idx]["info"]["img_path"]
        img_file_name = os.path.splitext(img_file_name)[0] + '.jpg'
        img_path = os.path.join(self.img_dir, img_file_name)
        if not os.path.exists(img_path):
            print(f"File {img_path} does not exist. Skipping.")
            return None
        try:
            
            image = open_and_rotate_image(img_path).convert('RGB')
  
            get_camera_model(img_path)
        

            original_size = image.size
            
            image = image. resize((224, 224)) # Resize the image before converting to an array
            
        except OSError:
            print(f"OSError: Can't read file {img_path}. Skipping.")
            return None
        
        sugar_content = self.data_info[idx]["collection"]["sugar_content_nir"]
        segmentation = self.data_info[idx]["annotations"]["segmentation"]

        if segmentation is None or not isinstance(segmentation, list):
            print(f"Invalid segmentation data for image {img_path}. Skipping.")
            return None

        # Update segmentation according to new image size
        new_size = (224, 224)
        scale_x = new_size[0] / original_size[0]
        scale_y = new_size[1] / original_size[1]
        scaled_segmentation = []
        for i in range(0, len(segmentation), 2):
            x = segmentation[i] * scale_x
            y = segmentation[i + 1] * scale_y
            scaled_segmentation. append((x, y))

        # match image with segmentaion 
        mask = Image.new('L', new_size, 0)
        ImageDraw.Draw(mask).polygon(scaled_segmentation, outline=1, fill=1)
        mask = np.array(mask)
        
        image = np.array(image)
        mask = np.expand_dims(mask, axis=2)
        image = image * mask

        # Get bounding box of the mask
        y_indices, x_indices, _ = np.where(mask)
        xmin, xmax = np.min(x_indices), np.max(x_indices)
        ymin, ymax = np.min(y_indices), np.max(y_indices)

        # Crop the image using bounding box of the mask
        cropped_image = image[ymin:ymax, xmin:xmax]
        image = Image.fromarray(cropped_image)

        # Resize the image to a fixed size (e.g., (224, 224))
        image = image.resize((224, 224))

        # Transform the PIL image to a PyTorch tensor
        image = transforms.ToTensor()(image)

        return image, sugar_content

    
def open_and_rotate_image(image_path):
    # if image is rotated correct the image rotation
    image = Image. open(image_path)
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = dict(image._getexif().items())
        if exif[orientation] == 3:
            image = image. rotate(180, expand=True)
        elif exif[orientation] == 6:
            image = image. rotate(270, expand=True)
        elif exif[orientation] == 8:
            image = image. rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError):
        # If there's no EXIF ??data or the orientation data isn't set, just return the original image
        pass
    return image


def get_camera_model(image_path):
    # get camera information 
    global value
    image = Image. open(image_path)
    exif_data = image._getexif()

    if exif_data is not None:
        for tag_id, value in exif_data.items():
            tag_name = TAGS.get(tag_id, tag_id)
            if tag_name == 'Model':
                return value

    return None

def get_image_paths(directory):
    image_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_paths.append(os.path.join(root, file))
    return image_paths

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

if __name__ == '__main__':
    # Retrieve image paths for training and validation datasets
    train_img_paths = get_image_paths("/home/poi0724/new_apple/train/image/")
    val_img_paths = get_image_paths("/home/poi0724/new_apple/val/image/")
    
    # Retrieve image and brix label for training and validation datasets
    train_dataset = AppleDataset(img_dir="/home/poi0724/new_apple/train/image/", json_dir="/home/poi0724/new_apple/train/label/")
    val_dataset = AppleDataset(img_dir="/home/poi0724/new_apple/val/image/", json_dir="/home/poi0724/new_apple/val/label/")
    
    # Initialize data loaders for training and validation datasets
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=64)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn, num_workers=64)
    
    num_models = 3
    models_list = []
    for _ in range(num_models):
        # Train using vit_base_patch16_224 model and freeze parameters
        model = timm.create_model('vit_base_patch16_224', pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        
        # Update the model's head to output a single value
        model.head = nn.Linear(model.head.in_features, 1)
        
        # Move the model to the GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # Add the model to the models list
        models_list.append(model)
    
    # Use the MSE loss
    criterion = nn.MSELoss()
    
    # Initialize best validation loss for each model
    best_loss = [float('inf')] * num_models  
    
    for i, model in enumerate(models_list):
        print(f"Training model {i+1}")
        
        # Initialize Adam optimizer 
        optimizer = optim.Adam(model.head.parameters())
    
        num_epochs = 10
        
        for i, model in enumerate(models_list):
            print(f"Training model {i+1}")
    
            # Create lists to store loss per epoch for training and validation
            train_losses = []
            val_losses = []
    
            for epoch in range(num_epochs):
                model.train()
                running_loss = 0.0
                pbar = tqdm(train_dataloader)

                for inputs, labels in pbar:
                    # Move inputs and labels to the GPU if available
                    inputs, labels = inputs.to(device), labels.to(device).float()
                    
                    # Reshape the labels
                    labels = labels.view(-1, 1)
                    
                    # Zero out the gradients
                    optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = model(inputs)
                    
                    # Calculate the loss
                    loss = criterion(outputs, labels)
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
    
                    running_loss += loss.item() * inputs.size(0)
                    pbar.set_description(f"Train Epoch: {epoch+1}, Loss: {loss:.4f}")
                
                # Calculate and print epoch loss for training
                epoch_loss = running_loss / len(train_dataset)
                train_losses.append(epoch_loss)
                print(f'Train Loss: {epoch_loss:.4f}')
    
                # Switch to evaluation mode
                model.eval()
                running_loss = 0.0
                pbar = tqdm(val_dataloader)
                
                # Evaluate the model on validation data
                with torch.no_grad():
                    for inputs, labels in pbar:
                        # Move inputs and labels to the GPU if available
                        inputs, labels = inputs.to(device), labels.to(device).float()
                        
                        # Forward pass
                        outputs = model(inputs)
                        
                        # Calculate the loss
                        loss = criterion(outputs, labels)
                        running_loss += loss.item() * inputs.size(0)
                        pbar.set_description(f"Val Epoch: {epoch+1}, Loss: {loss:.4f}")
                    
                    # Calculate and print epoch loss for validation
                    epoch_loss = running_loss / len(val_dataset)
                    val_losses.append(epoch_loss)
                    print(f'Validation Loss: {epoch_loss:.4f}')
    
                # Save the model if validation loss has improved
                if epoch_loss < best_loss[i]:
                    print(f"Validation loss decreased ({best_loss[i]:.6f} --> {epoch_loss:.6f}). Saving model...")
                    torch.save(model.state_dict(), f'model_{i+1}_best.pth')
                    best_loss[i] = epoch_loss
    
            # Save losses to a file for later plotting
            with open(f'model_{i+1}_losses.txt', 'w') as file:
                for train_loss, val_loss in zip(train_losses, val_losses):
                    file.write(f'{train_loss},{val_loss}\n')
