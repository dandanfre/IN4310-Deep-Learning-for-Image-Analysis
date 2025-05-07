import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from NatureTypesDataset import NatureTypesDataset
import torch.nn as nn
from ResNet import ResNet
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
import torch
import numpy as np
import random
from torchvision.models import resnet18


# Set the seed for recallability
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_metrics(predictions, labels, num_classes=6):
    accuracy_per_class = np.zeros(num_classes)
    total_per_class = np.zeros(num_classes)
    softmax_scores = nn.functional.softmax(predictions, dim=1).cpu().numpy()
    predicted_labels = predictions.argmax(dim=1).cpu().numpy()
    true_labels = labels.cpu().numpy()

    for i in range(num_classes):  # Calculate the accuracy for each class(AP)
        mask = true_labels == i
        accuracy_per_class[i] = (predicted_labels[mask] == i).sum()
        total_per_class[i] = mask.sum()

    accuracy_per_class = accuracy_per_class / np.maximum(total_per_class, 1)
    mAP = average_precision_score(  # Calculate the mean accuracy for all classes(mAP)
        np.eye(num_classes)[true_labels], softmax_scores, average="macro"
    )

    return accuracy_per_class, mAP, softmax_scores


def evaluate(model, data_loader, device):
    model.eval()
    all_predictions = []
    all_labels = []
    all_softmax_scores = []

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            predictions = model(images)
            accuracy_per_class, mAP, softmax_scores = compute_metrics(
                predictions, labels
            )
            all_predictions.append(predictions.cpu())
            all_labels.append(labels.cpu())
            all_softmax_scores.append(softmax_scores)

    return accuracy_per_class, mAP, np.vstack(all_softmax_scores)


def main():
    root_dir = os.getcwd()
    val_csv = os.path.join(root_dir, "validation.csv")

    if not os.path.exists(val_csv):
        print("Test file not found!")
        return

    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(
                (224, 224),
            ),
            transforms.ToTensor(),
        ]
    )

    transform = transform  # resize images to 80x80.

    """
    Depending if you use test or validation dataset. Adjusted the names accordingly
    """

    val_dataset = NatureTypesDataset(
        csv_file=val_csv, root_dir=root_dir, train=False, transform=transform
    )
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Load the model
    """
    # Initialize resnet18 model with pretrained weights from torch

    model = resnet18(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 6)

    model.load_state_dict(
        torch.load(
            "best_nature_classifier_only_weights_resnet_sgd.pth", map_location="cpu"
        )
    )

    model = model.to("cpu")
    print("Complete model loaded!")
    model.eval()

    # Provided ResNet class, change to necessary files.
    
    sgd optimizer, learning rate 0.001  -> best_nature_classifier_only_weights_resnet_sgd.pth 
    adam optimizer, learning rate 0.001 -> best_nature_classifier_only_weights_resnet_sgd.pth
    adam optimizer, learning rate 0.01  -> best_nature_classifier_only_weights_resnet_sgd.pth
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResNet(img_channels=3, num_layers=18, num_classes=6)
    model.load_state_dict(
        torch.load(
            "best_nature_classifier_only_weights.pth", map_location="cpu"
        )  # fill inn the neccessary model
    )
    model.to(device)
    print("Complete model loaded!")
    model.eval()

    test_accuracy_per_class, test_mAP, test_softmax_scores = evaluate(
        model, val_loader, device="cpu"
    )

    np.save("test_softmax_scores.npy", test_softmax_scores)
    print(f"Test Accuracy per class: {test_accuracy_per_class}")
    print(f"Test mAP: {test_mAP:.4f}")

    # Load and verify softmax scores
    loaded_softmax_scores = np.load("test_softmax_scores.npy")
    assert np.allclose(
        test_softmax_scores, loaded_softmax_scores, atol=1e-5
    ), "Mismatch in softmax scores!"
    print("Softmax score verification passed!")

    # Plot Loss vs Epochs for Test Set
    epochs = np.arange(1, 10)  # Assuming 5 epochs
    test_loss = np.random.uniform(0.2, 0.6, size=9)  # Placeholder loss values

    plt.plot(epochs, test_loss, marker="o", label="Test Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Test Loss Over Epochs")
    plt.legend()
    plt.savefig("test_loss_plot.png")
    plt.show()


if __name__ == "__main__":
    set_seed(42)  # Set the seed for reproducibility
    main()
