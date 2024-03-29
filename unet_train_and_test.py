import time
from PIL import Image
from matplotlib import pyplot as plt
import torch
import os
import torchvision.transforms as transforms
import torchvision
import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from unet_model import *
import numpy as np

# MACRO for printing
PRINT=True

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print('CUDA not available. Running on CPU.')

# Parameters
num_epochs = 10
num_classes = 4
batchsize = 8
lr = 2e-4

# Paths to the directories containing the training, validation, and test data
# with images of the brain slices via the Preprocessed OASIS dataset
base_directory = '.'
images_directory = base_directory + '\keras_png_slices_data'

train_directory = images_directory + '\keras_png_slices_train'
train_masks_directory = images_directory + '\keras_png_slices_seg_train'

test_directory = images_directory + '\keras_png_slices_test'
test_masks_directory = images_directory + '\keras_png_slices_seg_test'

validation_directory = images_directory + '\keras_png_slices_validate'
validation_masks_directory = images_directory + '\keras_png_slices_seg_validate'

model_name='unet'
output_directory = './output_torch_' + model_name

if not os.path.isdir(output_directory):
    os.mkdir(output_directory)


# Dataset class
class BrainSlicesDataset(Dataset):
    """
    Dataset class for storing the brain slices data
    """
    def __init__(self, img_directory, masks_directory, transform=None):
        """
        @param img_directory: the directory containing the images
        @param masks_directory: the directory containing the masks
        @param transform: the transform to apply to the images and masks
        """
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
        """
        Return the length of the dataset
        @return: the length of the dataset
        """
        return self.length
    
    def __getitem__(self, index):
        """
        Get the image and mask at the given index
        @param index: the index of the image to get
        @return: the image and mask at the given index
        """
        image, mask = self.images[index]['image'], self.images[index]['mask']
        image = Image.open(image)
        mask = Image.open(mask)

        # print(np.array(image).shape)
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
            print(mask)

            # one_hot_mask = torch.nn.functional.one_hot(mask, num_classes=num_classes)
            # desired_shape = (one_hot_mask.shape[0], num_classes, one_hot_mask.shape[1], one_hot_mask.shape[2])
            # one_hot_mask = one_hot_mask.permute(0, 3, 1, 2)  # Permute dimensions to match the desired shape
            # one_hot_mask = one_hot_mask.view(*desired_shape)

        return image, mask

# Function for generating a mask for the given image

def generate_and_save_images(model, epoch, test_input):
    """
    Generates and save images
    @param model: the model to use for generating the images
    @param epoch: the epoch number
    @param test_input: the test input
    
    Reference: https://github.com/shakes76/pattern-analysis-2023/blob/experimental/generative/test_vae_mnist.py#L243
    """
    with torch.no_grad():
        predictions = model(test_input)

    fig = plt.figure(figsize=(256,256))

    for i in range(predictions.shape[0]):
        plt.subplot(256, 256, i+1)
        plt.imshow(predictions[i, 0, :, :].cpu().detach().numpy(), cmap='gray')
        plt.axis('off')

    plt.savefig(output_directory+'/image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()


# 1. Preprocess the data with transforms and load data with DataLoader
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(1,))
])

train_set = BrainSlicesDataset(train_directory, train_masks_directory, transform=transform)
train_loader = DataLoader(train_set, batch_size=batchsize, shuffle=True)

test_set = BrainSlicesDataset(test_directory, test_masks_directory, transform=transform)
test_loader = DataLoader(test_set, batch_size=batchsize, shuffle=False)

validate_set = BrainSlicesDataset(validation_directory, validation_masks_directory, transform=transform)
validate_loader = DataLoader(validate_set, batch_size=batchsize, shuffle=False)

def train_model(model, criteria, optimizer, num_epochs, validation_interval=1):
    """
    Train the model
    @param model: the model to train
    @param criteria: the loss function
    @param optimizer: the optimizer
    @param train_loader: the training data loader
    @param val_loader: the validation data loader
    @param num_epochs: the number of epochs to train for
    @param validation_interval: interval (in epochs) at which to perform validation
    """
    print("> Commence Training")
    # device = next(model.parameters()).device
    model.train()
    start = time.time()
    
    train_losses = []  # Store training losses
    val_losses = []    # Store validation losses
    
    for epoch in range(num_epochs):
        # Training phase
        train_loss_sum = 0.0
        for image, mask in train_loader:
            image = image.to(device)
            mask = mask.to(device)
            
            # Flatten the mask to 2D
            mask = mask.view(-1, mask.shape[2], mask.shape[3])

            # Forward pass
            output = model(image)
            loss = criteria(output, mask)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss_sum += loss.item()
        
        # Calculate average training loss for the epoch
        avg_train_loss = train_loss_sum / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Print training loss
        print('Epoch [{}/{}], Training Loss: {:.4f}'.format(epoch + 1, num_epochs, avg_train_loss))
        
        # Validation phase
        if (epoch + 1) % validation_interval == 0:
            model.eval()  # Set model to evaluation mode
            val_loss_sum = 0.0
            with torch.no_grad():
                for val_image, val_mask in validate_loader:
                    val_image = val_image.to(device)
                    val_mask = val_mask.to(device)
                    
                    val_output = model(val_image)
                    val_loss = criteria(val_output, val_mask)
                    
                    val_loss_sum += val_loss.item()
            
            # Calculate average validation loss for the epoch
            avg_val_loss = val_loss_sum / len(validate_loader)
            val_losses.append(avg_val_loss)
            
            # Print validation loss
            print('Epoch [{}/{}], Validation Loss: {:.4f}'.format(epoch + 1, num_epochs, avg_val_loss))
            
            model.train()  # Set model back to training mode
    
    end = time.time()
    print(f"Training completed in {end - start} seconds.")
    
    return train_losses, val_losses
    
def test_model(model):
    """
    Test the model
    @param model: the model to test
    """
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
    
# 3. Train and test the model
model = UNet()
model.to(device)
criteria = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
train_model(model, criteria, optimizer, num_epochs)
test_model(model)
generate_and_save_images(model, num_epochs, test_set[0][0].unsqueeze(0).to(device))        