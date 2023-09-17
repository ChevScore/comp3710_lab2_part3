
import time
from PIL import Image
import torch
import os
import torchvision.transforms as transforms
import torchvision
import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from unet_model import *

PRINT=True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print('CUDA not available. Running on CPU.')

# Parameters
num_epochs = 10
depth = 5
lr = 2e-4

# Paths to the directories containing the training, validation, and test data
# with images of the brain slices via the Preprocessed OASIS dataset
base_directory = '/home/chevscore/comp3710/lab2/comp3710_lab2_part3'
images_directory = base_directory + '/keras_png_slices_data'

train_directory = images_directory + '/keras_png_slices_train'
train_masks_directory = images_directory + '/keras_png_slices_seg_train'

test_directory = images_directory + '/keras_png_slices_test'
test_masks_directory = images_directory + '/keras_png_slices_seg_test'

model_name='unet'
output_directory = './output_torch_' + model_name

if not os.path.isdir(output_directory):
    os.mkdir(output_directory)

# Dataset class

class BrainSlicesDataset(Dataset):
    def __init__(self, img_directory, masks_directory, transform=None):
        self.img_directory = img_directory
        self.masks_directory = masks_directory
        self.transform = transform
        self.images = []
        for filename in os.listdir(img_directory):
            self.images.append({
                'image': img_directory + '/' + filename,
                'mask': masks_directory +  '/' + filename.replace('case', 'seg', 1)
            })
            if PRINT:
                print('Loaded ' + self.images[-1]['image'] + ' and ' + self.images[-1]['mask'])
        self.length = len(self.images)
        if PRINT:
            print('Loaded ' + str(self.length) + ' images from ' + images_directory)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        image, mask = self.images[index]['image'], self.images[index]['mask']
        image = Image.open(image)
        mask = Image.open(mask)
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        return image, mask

# 1. Preprocess the data with transforms and load data with DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(1,))
])

train_set = BrainSlicesDataset(train_directory, train_masks_directory, transform=transform)
train_loader = DataLoader(train_set, batch_size=len(train_set), shuffle=True)

test_set = BrainSlicesDataset(test_directory, test_masks_directory, transform=transform)
test_loader = DataLoader(test_set, batch_size=len(test_set), shuffle=True)

# 2. Setup training

def train_model(model, criteria, optimizer, num_epochs):
    device = next(model.parameters()).device
    model.train()
    start = time.time()
    for epoch in range(num_epochs):
        for image, mask in train_loader:
            image = image.to(device)
            mask = mask.to(device)
            
            # Forward pass
            output = model(image)
            loss = criteria(output, mask)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
    
    
    end = time.time()
    elapsed = end - start
    print('Training completed in {:.0f}m {:.0f}s'.format(elapsed // 60, elapsed % 60))
    
def test_model(model):
    start = time.time()
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for image, mask in test_loader:
            image = image.to(device)
            mask = mask.to(device)
            
            output = model(image)
            _, predicted = torch.max(output.data, 1)
            total += mask.size(0)
            correct += (predicted == mask).sum().item()
        print('Accuracy of the network on the test images: {} %'.format(100 * correct / total))
        
    end = time.time()
    elapsed = end - start
    print('Testing completed in {:.0f}m {:.0f}s'.format(elapsed // 60, elapsed % 60))
    
# 3. Train the model

model = UNet()
model.to(device)
criteria = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
train_model(model, criteria, optimizer, num_epochs)
test_model(model)
        