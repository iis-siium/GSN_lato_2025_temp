# SIMPLIFIED YOLO TRAINING LOOP

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

# Import these from your actual module files
from yolo_simplest_dataset import YOLOVOCDataset, custom_collate, VOC_CLASSES
from yolo_simplest_model import YOLOv1Model

class VOCTrainer:
    def __init__(self, config):
        self.config = config
        self.setup_data()
        self.setup_model()
    
    def setup_data(self):
        """Set up datasets and dataloaders"""
        # Create train dataset
        self.train_dataset = YOLOVOCDataset(
            root=self.config.data_dir,
            year='2007',
            image_set='train',
            grid_size=self.config.grid_size,
            num_boxes=self.config.num_boxes,
            num_classes=self.config.num_classes
        )
        
        # Create validation dataset
        self.val_dataset = YOLOVOCDataset(
            root=self.config.data_dir,
            year='2007',
            image_set='val',
            grid_size=self.config.grid_size,
            num_boxes=self.config.num_boxes,
            num_classes=self.config.num_classes
        )
        
        # Create dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            collate_fn=custom_collate,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=custom_collate,
            pin_memory=True
        )
        
        print(f"Train dataset size: {len(self.train_dataset)}")
        print(f"Validation dataset size: {len(self.val_dataset)}")
    
    def setup_model(self):
        """Set up YOLO model"""
        self.model = YOLOv1Model(
            grid_size=self.config.grid_size,
            num_boxes=self.config.num_boxes,
            num_classes=self.config.num_classes,
            lambda_coord=self.config.lambda_coord,
            lambda_noobj=self.config.lambda_noobj,
            learning_rate=self.config.learning_rate
        )
    
    def train(self):
        """Train the model"""
        # Create callbacks
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            filename='yolo-{epoch:02d}-{val_loss:.2f}',
            save_top_k=3,
            mode='min'
        )
        
        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            patience=10,
            verbose=True,
            mode='min'
        )
        
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        
        # Create logger
        logger = TensorBoardLogger("logs", name="yolo_voc")
        
        # Create trainer
        trainer = pl.Trainer(
            max_epochs=self.config.epochs,
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=self.config.gpus if torch.cuda.is_available() else None,
            callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
            logger=logger,
            precision=self.config.precision,
            gradient_clip_val=self.config.gradient_clip_val  # Added back gradient clipping
        )
        
        # Train model
        trainer.fit(
            self.model,
            train_dataloaders=self.train_loader,
            val_dataloaders=self.val_loader
        )
        
        return trainer.checkpoint_callback.best_model_path


def main():
    # Define configuration
    class Config:
        data_dir = './VOCdata'  # Path to data directory
        grid_size = 7  # Grid size (S)
        num_boxes = 2  # Number of boxes per cell (B)
        num_classes = 20  # Number of classes (C)
        lambda_coord = 5.0  # Weight for coordinate predictions
        lambda_noobj = 0.5  # Weight for no-object confidence
        batch_size = 12
        learning_rate = 1e-3
        epochs = 50
        gpus = 1 if torch.cuda.is_available() else 0  # Number of GPUs
        num_workers = 4  # Number of workers for data loading
        precision = 32  # Precision for training
        gradient_clip_val = 1.0  # Gradient clipping value - added back
    
    config = Config()

    # Regular training with train/val split
    trainer = VOCTrainer(config)
    best_model_path = trainer.train()
    print(f"Best model saved at: {best_model_path}")


if __name__ == "__main__":
    main()