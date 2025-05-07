import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet18
from NatureTypesDataset import NatureTypesDataset


# Setting seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


config = {
    "root_dir": os.getcwd(),
    "path_to_data": os.path.join(os.getcwd(), "train.csv"),
    "transform": transforms.Compose(
        [transforms.ToPILImage(), transforms.Resize((224, 224)), transforms.ToTensor()]
    ),
}


# Hook function to capture feature maps
def get_activation(name):
    def hook(model, input, output):
        feature_maps[name] = output.detach()

    return hook


# Hook function to compute percentage of non-positive values
def compute_non_positive_percentage(name):
    def hook(model, input, output):
        # Calculate percentage of non-positive values (≤ 0)
        total_elements = output.numel()
        non_positive = (output <= 0).sum().item()
        percentage = (non_positive / total_elements) * 100

        # Update running statistics
        if name in non_positive_stats:
            non_positive_stats[name]["count"] += 1
            non_positive_stats[name]["sum"] += percentage
            non_positive_stats[name]["avg"] = (
                non_positive_stats[name]["sum"] / non_positive_stats[name]["count"]
            )
        else:
            non_positive_stats[name] = {
                "count": 1,
                "sum": percentage,
                "avg": percentage,
            }

    return hook


def visualize_feature_maps(feature_maps, output_dir, image_idx=0, max_channels=7):
    """
    Visualize feature maps from different layers
    Args:
        feature_maps: Dictionary of feature maps from different layers
        output_dir: Directory to save the visualizations
        image_idx: Index of the image (used for filename only)
        max_channels: Maximum number of channels to visualize per layer
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Number of layers to visualize
    num_layers = len(feature_maps)

    # Create a figure with subplots for each layer
    fig, axes = plt.subplots(num_layers, max_channels, figsize=(15, 3 * num_layers))

    # If only one layer, make sure axes is 2D
    if num_layers == 1:
        axes = np.expand_dims(axes, axis=0)

    # For each layer
    for i, (layer_name, feature_map) in enumerate(feature_maps.items()):
        # The feature map is already for a single image (batch size 1)
        # So we just take the first item in the batch dimension
        feature_map = feature_map[
            0
        ]  # Fixed: Always use index 0 because batch size is 1

        # Determine number of channels to visualize (up to max_channels)
        num_channels = min(feature_map.shape[0], max_channels)

        # Plot each channel
        for j in range(max_channels):
            if j < num_channels:
                # Normalize the feature map for better visualization
                fmap = feature_map[j].cpu().numpy()
                axes[i, j].imshow(fmap, cmap="viridis")
                axes[i, j].set_title(f"{layer_name[:3]}_ch{j}")

            # Remove axis ticks
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"feature_maps_image_{image_idx}.png"))
    plt.close(fig)


def plot_10_fmaps():

    pass


if __name__ == "__main__":
    # Set default seed=42 for reproducibility
    set_seed()
    print("Setting seed")

    # Setup dataset and dataloader
    t_dataset = NatureTypesDataset(
        csv_file=config["path_to_data"],
        root_dir=config["root_dir"],
        transform=config["transform"],
    )
    t_dataloader = DataLoader(t_dataset, batch_size=1, shuffle=True)

    # Load pretrained ResNet18 model
    model = resnet18(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 6)  # 6 classes in the dataset

    # Load trained weights
    model.load_state_dict(
        torch.load(
            "best_nature_classifier_only_weights_resnet18.pth", map_location="cpu"
        )
    )
    model = model.to("cpu")
    model.eval()

    # Part c: Visualize feature maps for 10 images  - inside feature_maps_visualization folder
    print("Part c: Visualizing feature maps for 10 images")

    # Select layers to analyze
    layer_names = ["layer1", "layer2", "layer3", "layer4"]

    # Create directory for feature map visualizations
    visualization_dir = os.path.join(os.getcwd(), "feature_maps_visualization")
    os.makedirs(visualization_dir, exist_ok=True)

    # Process 10 images one by one
    image_count = 0
    for images, labels in t_dataloader:
        # Dictionary to store feature maps
        feature_maps = {}

        # Register hooks to capture feature maps
        hooks = []
        for name in layer_names:
            layer = getattr(model, name)
            hooks.append(layer.register_forward_hook(get_activation(name)))

        # Forward pass with a single image
        with torch.no_grad():
            _ = model(images)  # images already has batch size 1

        # Visualize feature maps
        visualize_feature_maps(feature_maps, visualization_dir, image_count)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        image_count += 1
        if image_count >= 10:
            break

    print(f"Feature map visualizations saved to {visualization_dir}")

    # Part e & f: Compute percentage of non-positive values for 5 selected feature maps
    print("Part e & f: Computing percentage of non-positive values")

    # Dictionary to store statistics
    non_positive_stats = {}

    # Select 5 specific modules to analyze
    selected_modules = [
        "layer1.0.conv1",  # Early layer
        "layer2.0.conv1",  # Middle layer
        "layer3.0.conv1",  # Later middle layer
        "layer4.0.conv1",  # Late layer
        "layer4.1.conv2",  # Final conv layer
    ]

    # Register hooks for these modules
    hooks = []
    for name, module in model.named_modules():
        if name in selected_modules:
            hooks.append(
                module.register_forward_hook(compute_non_positive_percentage(name))
            )

    # Process 200 images to compute statistics
    print("Processing 200 images to compute non-positive percentage statistics...")
    image_count = 0

    # Reset the dataloader to process from the beginning
    t_dataloader = DataLoader(t_dataset, batch_size=1, shuffle=True)

    # Loop through batches until we process 200 images
    for batch_idx, (images, _) in enumerate(
        t_dataloader
    ):  # last tuple set is not needed, labels are not used
        with torch.no_grad():
            # Forward pass with single image (already batch size 1)
            _ = model(images)

            image_count += 1
            if image_count % 20 == 0:
                print(f"Processed {image_count}/200 images")

            if image_count >= 200:
                break

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Print and save non-positive percentage statistics
    print("\nPercentage of non-positive values in feature maps:")
    for module_name, stats in non_positive_stats.items():
        print(f"{module_name}: {stats['avg']:.2f}%")

    # Save statistics to a file
    with open("non_positive_stats.txt", "w") as f:
        f.write("Module Name | Avg. % of Non-Positive Values\n")
        f.write("-" * 50 + "\n")
        for module_name, stats in non_positive_stats.items():
            f.write(f"{module_name} | {stats['avg']:.2f}%\n")

    print("Non-positive statistics saved to non_positive_stats.txt")
