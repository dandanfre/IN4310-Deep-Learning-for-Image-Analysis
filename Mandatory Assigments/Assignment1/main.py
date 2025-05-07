import data_handling as dh
from sklearn.model_selection import train_test_split
import data_handling as dh
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
from NatureTypesDataset import NatureTypesDataset


"""
The main.py was primarily used to test dataclass and dataloader to check if they worked properly
"""

if __name__ == "__main__":
    path, labels = dh.create_labels_paths_lists()
    # label_counts = Counter(labels)
    # print("Label distribution:", label_counts)
    X_train, X_test, X_val = dh.create_datasets(path, labels)
    # csv_paths = dh.create_csv_files(X_train, X_val, X_test)

    root_dir = os.getcwd()

    training_data = NatureTypesDataset(csv_file="train.csv", root_dir=root_dir)
    test_data = NatureTypesDataset(csv_file="test.csv", root_dir=root_dir)
    val_data = NatureTypesDataset(csv_file="validation.csv", root_dir=root_dir)

    """
    print(len(val_data))

    fig = plt.figure()

    for i, (image, label) in enumerate(val_data):
        print(i, image.shape, label)

        ax = plt.subplot(1, 4, i + 1)
        plt.tight_layout()
        ax.set_title(
            f"Sample {i} - {val_data.inv_class_dict[label]}"  # map numeric value to class label
        )  # map class label to number
        ax.imshow(image)
        ax.axis("off")

        if i == 3:
            plt.show()
            break

        """

    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

    # Display image and label.
    train_features, train_labels = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    img = train_features[0].squeeze()
    label = train_labels[0]
    plt.imshow(img, cmap="gray")
    plt.show()
    print(f"Label: {label}")
