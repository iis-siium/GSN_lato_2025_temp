# SIMPLEST YOLO DATASET (VOC based)

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import VOCDetection
from torchvision import transforms
from PIL import Image

# VOC class names for reference
VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
    'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

class YOLOVOCDataset(Dataset):
    def __init__(self, root, year='2007', image_set='train', grid_size=7, num_boxes=2, num_classes=20):
        """
        Custom YOLO dataset for Pascal VOC
        
        Args:
            root: Root directory of VOC dataset
            year: Year of dataset ('2007' or '2012')
            image_set: 'train', 'val', or 'test'
            grid_size: Grid size for YOLO (S×S grid)
            num_boxes: Number of bounding boxes per grid cell (B)
            num_classes: Number of classes (C)
        """
        self.voc_dataset = VOCDetection(
            root=root,
            year=year,
            image_set=image_set,
            download=True
        )
        
        self.grid_size = grid_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        
        # For VOC dataset
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(VOC_CLASSES)}
        
        # Basic transforms
        self.transform = transforms.Compose([
            transforms.Resize((448, 448)),  # Original YOLO used 448×448
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.voc_dataset)
    
    def __getitem__(self, idx):
        img, annotation = self.voc_dataset[idx]
        
        # Get original image dimensions
        width, height = img.size
        
        # Initialize target tensor
        # Format: [S, S, 5*B + C]
        # For each cell: [confidence1, x1, y1, w1, h1, confidence2, x2, y2, w2, h2, class_probs...]
        target = torch.zeros((self.grid_size, self.grid_size, 5 * self.num_boxes + self.num_classes))
        
        # Process all objects in the image
        for obj in annotation['annotation']['object']:
            # Get class index
            class_idx = self.class_to_idx[obj['name']]
            
            # Get bounding box coordinates (in original image scale)
            bbox = obj['bndbox']
            x_min = float(bbox['xmin'])
            y_min = float(bbox['ymin'])
            x_max = float(bbox['xmax'])
            y_max = float(bbox['ymax'])
            
            # Convert to center format and normalize
            x_center = ((x_min + x_max) / 2) / width
            y_center = ((y_min + y_max) / 2) / height
            bbox_width = (x_max - x_min) / width
            bbox_height = (y_max - y_min) / height
            
            # Find the grid cell this bounding box belongs to
            grid_x = int(self.grid_size * x_center)
            grid_y = int(self.grid_size * y_center)
            
            # Handle edge case where coordinates are exactly at the boundary
            grid_x = min(grid_x, self.grid_size - 1)
            grid_y = min(grid_y, self.grid_size - 1)
            
            # Convert box coordinates relative to the grid cell
            x_cell = self.grid_size * x_center - grid_x
            y_cell = self.grid_size * y_center - grid_y
            
            # Width and height relative to the whole image
            width_cell = bbox_width
            height_cell = bbox_height
            
            # Check if there's already an object in this cell
            # In the original YOLO paper, if multiple objects have center points in the same
            # grid cell, only one object can be detected. For training, we'll use the first
            # box position if it's empty, otherwise the second box position.
            if target[grid_y, grid_x, 0] == 0:
                # First box is empty, use it
                box_idx = 0
            else:
                # First box is occupied, use second box
                # Note: This is a simple approach; original YOLO assigns the box
                # with the highest IOU to the ground truth
                box_idx = 1
                
                # If both boxes are already assigned, this will overwrite the second box.
                # This is a limitation of the original YOLO that's reduced in later versions.
            
            # Box index in the target tensor
            box_offset = box_idx * 5
            
            # Set confidence (objectness) to 1
            target[grid_y, grid_x, box_offset] = 1
            
            # Set box coordinates
            target[grid_y, grid_x, box_offset + 1] = x_cell
            target[grid_y, grid_x, box_offset + 2] = y_cell
            target[grid_y, grid_x, box_offset + 3] = width_cell
            target[grid_y, grid_x, box_offset + 4] = height_cell
            
            # Set class probability
            target[grid_y, grid_x, 5 * self.num_boxes + class_idx] = 1
        
        # Apply transforms to image
        img_tensor = self.transform(img)
        
        return img_tensor, target, (img, annotation)  # Return original data for visualization
    
    def visualize_sample(self, idx):
        """Visualize a sample with ground truth boxes"""
        img_tensor, target, (original_img, annotation) = self[idx]
        
        # Convert normalized tensor back to image for visualization
        img = img_tensor.permute(1, 2, 0).numpy()
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        
        # Create figure and axis
        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.imshow(img)
        
        # Get grid cell size in pixels
        cell_size_h = 448 / self.grid_size
        cell_size_w = 448 / self.grid_size
        
        # Iterate through grid cells
        for grid_y in range(self.grid_size):
            for grid_x in range(self.grid_size):
                for b in range(self.num_boxes):
                    # Get box parameters
                    box_offset = b * 5
                    confidence = target[grid_y, grid_x, box_offset].item()
                    
                    # Only draw boxes with objects
                    if confidence > 0:
                        # Get box coordinates relative to grid cell
                        x_cell = target[grid_y, grid_x, box_offset + 1].item()
                        y_cell = target[grid_y, grid_x, box_offset + 2].item()
                        width_cell = target[grid_y, grid_x, box_offset + 3].item()
                        height_cell = target[grid_y, grid_x, box_offset + 4].item()
                        
                        # Convert to absolute coordinates in the 448x448 image
                        x_center = (grid_x + x_cell) * cell_size_w
                        y_center = (grid_y + y_cell) * cell_size_h
                        width = width_cell * 448
                        height = height_cell * 448
                        
                        # Convert to top-left corner for rectangle
                        x_min = x_center - width / 2
                        y_min = y_center - height / 2
                        
                        # Find class with highest probability
                        class_idx = torch.argmax(target[grid_y, grid_x, 5 * self.num_boxes:]).item()
                        class_name = VOC_CLASSES[class_idx]
                        
                        # Draw rectangle
                        rect = patches.Rectangle(
                            (x_min, y_min), width, height, 
                            linewidth=2, edgecolor='r', facecolor='none'
                        )
                        ax.add_patch(rect)
                        
                        # Add class label
                        plt.text(
                            x_min, y_min - 5, 
                            f"{class_name} (box {b})", 
                            color='white', fontsize=10, 
                            bbox=dict(facecolor='red', alpha=0.5)
                        )
        
        # Draw grid
        for i in range(self.grid_size + 1):
            ax.axhline(y=i * cell_size_h, color='gray', linestyle='-', linewidth=0.5)
            ax.axvline(x=i * cell_size_w, color='gray', linestyle='-', linewidth=0.5)
        
        plt.title(f"YOLO Grid: {self.grid_size}×{self.grid_size}, Boxes per cell: {self.num_boxes}")
        fig.savefig(f"sample_{idx}.png")
        plt.show()
        
        return fig, ax

# Define a custom collate function that handles the raw image data
def custom_collate(batch):
    imgs, targets, extras = zip(*batch)
    imgs = torch.utils.data.dataloader.default_collate(imgs)
    targets = torch.utils.data.dataloader.default_collate(targets)
    return imgs, targets, list(extras)

def test_yolo_dataset():
    # Initialize dataset
    dataset = YOLOVOCDataset(
        root='./VOCdata',
        year='2007', 
        image_set='trainval',
        grid_size=7,
        num_boxes=2,
        num_classes=20
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Create dataloader with custom collate function
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=custom_collate)
    
    # Get a sample
    img_tensor, target, _ = next(iter(dataloader))
    
    print(f"Image shape: {img_tensor.shape}")
    print(f"Target shape: {target.shape}")
    
    # Visualize a few samples
    for i in range(3):
        random_idx = np.random.randint(0, len(dataset))
        print(f"Visualizing sample {random_idx}")
        dataset.visualize_sample(random_idx)


if __name__ == "__main__":
    test_yolo_dataset()
