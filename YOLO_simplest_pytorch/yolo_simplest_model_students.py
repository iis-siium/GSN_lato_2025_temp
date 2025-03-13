# SIMPLEST YOLO MODEL
# Using pytorch lightning

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision.ops as ops
from torchinfo import summary

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.1)
        
    def forward(self, x):
        return self.leaky_relu(self.bn(self.conv(x)))

class YOLOv1(nn.Module):
    def __init__(self, grid_size=7, num_boxes=2, num_classes=20):
        """
        Original YOLO v1 architecture
        
        Args:
            grid_size: Size of the grid (S in the paper)
            num_boxes: Number of bounding boxes per grid cell (B in the paper)
            num_classes: Number of classes (C in the paper)
        """
        super(YOLOv1, self).__init__()
        self.grid_size = grid_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        
        # Feature extraction inspired by original YOLO (simplified version of Darknet)
        self.features = nn.Sequential(

        )
        
        # Fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = 
        self.dropout = nn.Dropout(0.5)
        self.leaky_relu = nn.LeakyReLU(0.1)
        
        # Output layer (like in the article)
        self.fc2 =
        
    def forward(self, x):
        """Forward pass through the network
        
        Args:
            x: Input tensor with shape (batch_size, 3, 448, 448)
            
        Returns:
            Tensor with shape (batch_size, S, S, B*5+C)
        """
        x = self.features(x)
        x = self.flatten(x)
        x = self.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        # Reshape to [batch, S, S, B*5+C]
        batch_size = x.size(0)
        x = x.view(batch_size, self.grid_size, self.grid_size, (5 * self.num_boxes + self.num_classes))
        return x

class YOLOLoss(nn.Module):
    """
    YOLO Loss Function
    
    Implements the loss function from the original YOLO paper:
    1. Localization loss for bounding box coordinates
    2. Confidence loss for objectness
    3. Classification loss for class probabilities
    4. No-object confidence loss with lower weighting
    """
    def __init__(self, grid_size=7, num_boxes=2, num_classes=20, 
                 lambda_coord=5.0, lambda_noobj=0.5):
        super(YOLOLoss, self).__init__()
        self.grid_size = grid_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        self.lambda_coord = lambda_coord  # Weight for coordinate predictions
        self.lambda_noobj = lambda_noobj  # Weight for no-object confidence
        
    def forward(self, predictions, targets):
        """
        Calculate YOLO loss
        
        Args:
            predictions: Tensor of shape (batch_size, S, S, B*5+C)
            targets: Tensor of same shape as predictions
            
        Returns:
            Total loss (float), and dictionary of individual loss components
        """
        batch_size = predictions.size(0)
        
        # Reshape predictions to separate the box predictions
        pred_boxes = predictions[..., :self.num_boxes*5].contiguous()
        
        # Initialize loss components
        box_loss = torch.tensor(0.0, device=predictions.device)
        obj_loss = torch.tensor(0.0, device=predictions.device)
        noobj_loss = torch.tensor(0.0, device=predictions.device)
        class_loss = torch.tensor(0.0, device=predictions.device)
        
        # Iterate through each cell in the grid
        for b in range(batch_size):
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    # Class probability loss

                    
                    # Check if cell has an object
                    target_box_present = False

                    
                    if target_box_present:
                        # Class probability loss (only for cells with objects)

                        )
                    
                    # Process each box prediction for this cell
                    for k in range(self.num_boxes):
                        # Offset for this box in the tensor
                        box_offset = k * 5
                        
                        # Predicted box values
                        pred_conf = predictions[b, i, j, box_offset]
                        pred_x = 
                        pred_y = 
                        pred_w = 
                        pred_h = 
                        
                        # Target box values
                        target_conf = targets[b, i, j, box_offset]
                        target_x = 
                        target_y = 
                        target_w = 
                        target_h =
                        
                        # Check if this target box has an object
                        has_obj = (target_conf > 0)
                        
                        if has_obj:
                            # Box coordinate loss (only for cells with objects)
                            # Use square root of width and height as mentioned in the paper
                            box_loss += 
                            # multiple lines...

                            
                            # Object confidence loss
                            obj_loss 
                            
                        else:
                            # No-object confidence loss (with lower weight)
                            noobj_loss += 
        
        # Combine all loss components
        total_loss = box_loss + obj_loss + noobj_loss + class_loss
        
        # Normalize by batch size
        total_loss = total_loss / batch_size
        
        # Return total loss and components for logging
        loss_components = {
            'box_loss': box_loss / batch_size,
            'obj_loss': obj_loss / batch_size,
            'noobj_loss': noobj_loss / batch_size,
            'class_loss': class_loss / batch_size
        }
        
        return total_loss, loss_components

class YOLOv1Model(pl.LightningModule):
    def __init__(self, grid_size=7, num_boxes=2, num_classes=20, 
                 lambda_coord=5.0, lambda_noobj=0.5, learning_rate=1e-3):
        """
        YOLO model with PyTorch Lightning integration
        
        Args:
            grid_size: Size of the grid (S in the paper)
            num_boxes: Number of bounding boxes per grid cell (B in the paper)
            num_classes: Number of classes (C in the paper)
            lambda_coord: Weight for coordinate predictions
            lambda_noobj: Weight for no-object confidence
            learning_rate: Initial learning rate
        """
        super(YOLOv1Model, self).__init__()
        
        # Save hyperparameters
        self.save_hyperparameters()
        
        # Create model
        self.model = YOLOv1(grid_size, num_boxes, num_classes)
        
        # Create loss function
        self.loss_fn = YOLOLoss(grid_size, num_boxes, num_classes, lambda_coord, lambda_noobj)
        
    def forward(self, x):
        """Forward pass through the network"""
        return self.model(x)
    
    def configure_optimizers(self):
        """Configure the optimizer"""
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.learning_rate,
            momentum=0.9,
            weight_decay=5e-4
        )
        
        # Learning rate scheduler (similar to original paper)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, 
            milestones=[75, 105], 
            gamma=0.1
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }
    
    def training_step(self, batch, batch_idx):
        """Training step"""
        imgs, targets, _ = batch
        predictions = self(imgs)
        loss, loss_components = self.loss_fn(predictions, targets)
        
        # Log loss components
        self.log('train_loss', loss, prog_bar=True)
        for key, value in loss_components.items():
            self.log(f'train_{key}', value)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        imgs, targets, _ = batch
        predictions = self(imgs)
        loss, loss_components = self.loss_fn(predictions, targets)
        
        # Log loss components
        self.log('val_loss', loss, prog_bar=True)
        for key, value in loss_components.items():
            self.log(f'val_{key}', value)
        
        return loss
    
    def predict_boxes(self, x, confidence_threshold=0.5, nms_threshold=0.5):
        """
        Get predicted boxes from the model output
        
        Args:
            x: Input image tensor (batch_size, 3, H, W)
            confidence_threshold: Minimum confidence to keep a box
            nms_threshold: IoU threshold for non-maximum suppression
            
        Returns:
            List of lists containing boxes, scores, and class_ids for each image
        """
        # Get model predictions
        predictions = self(x)
        batch_size = predictions.size(0)
        
        all_boxes = []
        all_scores = []
        all_class_ids = []
        
        # Process each image in the batch
        for b in range(batch_size):
            boxes = []
            scores = []
            class_ids = []
            
            # Iterate through each cell
            for i in range(self.hparams.grid_size):
                for j in range(self.hparams.grid_size):
                    # Get class probabilities for this cell
                    class_probs = predictions[b, i, j, self.hparams.num_boxes*5:]
                    
                    # Find the class with highest probability
                    max_class_prob, max_class_idx = torch.max(class_probs, dim=0)
                    
                    # Check each bounding box in this cell
                    for k in range(self.hparams.num_boxes):
                        # Box offset
                        box_offset = k * 5
                        
                        # Get confidence
                        confidence = predictions[b, i, j, box_offset]
                        
                        # Calculate class confidence
                        class_confidence = confidence * max_class_prob
                        
                        # Filter boxes with low confidence
                        if class_confidence < confidence_threshold:
                            continue
                        
                        # Get box parameters
                        x_cell = predictions[b, i, j, box_offset + 1]
                        y_cell = predictions[b, i, j, box_offset + 2]
                        w_cell = predictions[b, i, j, box_offset + 3]
                        h_cell = predictions[b, i, j, box_offset + 4]
                        
                        # Convert to absolute coordinates (0-1)
                        x_center = (j + x_cell) / self.hparams.grid_size
                        y_center = (i + y_cell) / self.hparams.grid_size
                        w = w_cell
                        h = h_cell
                        
                        # Convert to corners format [x1, y1, x2, y2]
                        x1 = max(0, x_center - w/2)
                        y1 = max(0, y_center - h/2)
                        x2 = min(1, x_center + w/2)
                        y2 = min(1, y_center + h/2)
                        
                        # Add box, score, and class id to lists
                        boxes.append([x1, y1, x2, y2])
                        scores.append(class_confidence)
                        class_ids.append(max_class_idx)
            
            # Convert to tensors
            if boxes:
                boxes = torch.tensor(boxes, device=predictions.device)
                scores = torch.tensor(scores, device=predictions.device)
                class_ids = torch.tensor(class_ids, device=predictions.device)
                
                # Apply non-maximum suppression
                # Group boxes by class for class-wise NMS
                unique_classes = torch.unique(class_ids)
                nms_boxes = []
                nms_scores = []
                nms_class_ids = []
                
                for cls in unique_classes:
                    cls_mask = (class_ids == cls)
                    cls_boxes = boxes[cls_mask]
                    cls_scores = scores[cls_mask]
                    
                    # Apply NMS
                    keep_indices = ops.nms(cls_boxes, cls_scores, nms_threshold)
                    
                    # Add kept boxes to final results
                    nms_boxes.append(cls_boxes[keep_indices])
                    nms_scores.append(cls_scores[keep_indices])
                    nms_class_ids.append(torch.full((len(keep_indices),), cls, device=predictions.device))
                
                # Combine results
                if nms_boxes:
                    boxes = torch.cat(nms_boxes)
                    scores = torch.cat(nms_scores)
                    class_ids = torch.cat(nms_class_ids)
            else:
                boxes = torch.tensor([], device=predictions.device)
                scores = torch.tensor([], device=predictions.device)
                class_ids = torch.tensor([], device=predictions.device)
            
            # Add to batch results
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_class_ids.append(class_ids)
        
        return all_boxes, all_scores, all_class_ids
    

if __name__ == '__main__':
    # Test the YOLOv1 model
    model = YOLOv1Model(grid_size=7, num_boxes=2, num_classes=20)
    print(model)
    summary(model, (1, 3, 448, 448))
    
    # Test the YOLO loss function
    loss_fn = YOLOLoss(grid_size=7, num_boxes=2, num_classes=20)
    predictions = torch.randn(4, 7, 7, 30)
    targets = torch.randn(4, 7, 7, 30)
    loss, loss_components = loss_fn(predictions, targets)
    print('Total loss:', loss.item())
    print('Loss components:', loss_components)

    