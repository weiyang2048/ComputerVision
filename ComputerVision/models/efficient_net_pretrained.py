import timm
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from typing import Optional, Union, List


class EfficientNetModel:
    """
    A configurable EfficientNet model wrapper that supports different model sizes (b0-b7)
    and automatic output dimension determination.

    Attributes:
        model_size (str): Size of EfficientNet model ('b0' to 'b7')
        num_classes (int): Number of output classes
        pretrained (bool): Whether to use pretrained weights
        learning_rate (float): Learning rate for optimization
        device (torch.device): Device to run the model on
    """

    def __init__(
        self,
        model_size: str = "b0",
        num_classes: Optional[int] = None,
        pretrained: bool = True,
        learning_rate: float = 0.001,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the EfficientNet model.

        Args:
            model_size (str): Size of EfficientNet model ('b0' to 'b7')
            num_classes (int, optional): Number of output classes. If None, will be determined from train_loader
            pretrained (bool): Whether to use pretrained weights
            learning_rate (float): Learning rate for optimization
            device (torch.device, optional): Device to run the model on. If None, will use CUDA if available
        """
        self.model_size = model_size.lower()
        if not self.model_size in [f"b{i}" for i in range(8)]:
            raise ValueError(f"model_size must be one of {[f'b{i}' for i in range(8)]}")

        self.num_classes = num_classes
        self.pretrained = pretrained
        self.learning_rate = learning_rate
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Initialize model
        self.model = self._create_model()

    def _create_model(self) -> nn.Module:
        """Create and configure the EfficientNet model."""
        model = timm.create_model(
            f"efficientnet_{self.model_size}", pretrained=self.pretrained
        )

        # If num_classes is not specified, we'll determine it later from the train_loader
        if self.num_classes is not None:
            model.classifier = nn.Linear(model.classifier.in_features, self.num_classes)

        return model.to(self.device)

    def determine_num_classes(self, train_loader: DataLoader) -> None:
        """Determine number of classes from the training data."""
        if self.num_classes is None:
            # Get unique labels from the first batch
            _, labels = next(iter(train_loader))
            self.num_classes = len(torch.unique(labels))
            # Update model's classifier
            self.model.classifier = nn.Linear(
                self.model.classifier.in_features, self.num_classes
            )
            self.model = self.model.to(self.device)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        save_path: Optional[str] = None,
    ) -> List[dict]:
        """
        Train the model.

        Args:
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
            epochs (int): Number of epochs to train
            save_path (str, optional): Path to save the best model

        Returns:
            List[dict]: List of dictionaries containing training history
        """
        # Determine number of classes if not specified
        self.determine_num_classes(train_loader)

        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        history = []
        best_val_acc = 0.0

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            train_loss = running_loss / len(train_loader)
            train_accuracy = 100.0 * correct / total

            # Validation phase
            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

            val_loss /= len(val_loader)
            val_accuracy = 100.0 * correct / total

            # Save best model if specified
            if save_path and val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                torch.save(self.model.state_dict(), save_path)

            # Record history
            history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                }
            )

            print(
                f"Epoch {epoch+1}/{epochs}, "
                f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
                f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%"
            )
            
        # Set model to evaluation mode
        self.model.eval()
        return history

    def predict(self, image: torch.Tensor) -> torch.Tensor:
        """
        Make predictions on a single image.

        Args:
            image (torch.Tensor): Input image tensor

        Returns:
            torch.Tensor: Predicted class probabilities
        """
        with torch.no_grad():
            image = image.to(self.device)
            outputs = self.model(image)
            return torch.softmax(outputs, dim=1)
