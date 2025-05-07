import os
from collections import Counter
import pandas as pd
from sklearn.model_selection import train_test_split
import csv


def create_labels_paths_lists():
    # 1. Create lists containing image paths and labels
    # 2. Create dictionary to map from class labels to class numbers, e.g. glacier -> 0
    # 2. Root directory
    # try:
    # 3. Iterate through each class folder by name
    # 4. Create a path to the class folder
    # 5. Check-condition for robustness: Skip if not a directory
    # 6. Iterate through each image in the class folder
    # 7. Append the relative path
    # 8. Append the class ID
    # except OSError as e:
    # print(e)
    # print("Error: Could not find  directory")

    data_path = os.path.join(
        os.getcwd(), "data"
    )  # Path to data directory = root + data
    img_paths = []  # List to store image paths
    labels = []  # List to store labels
    class_dict = {
        "buildings": 0,
        "forest": 1,
        "glacier": 2,
        "mountain": 3,
        "sea": 4,
        "street": 5,
    }  # mapping

    try:
        for class_folder in os.listdir(data_path):
            #  print(class_dict[str(class_folder)])
            class_folder_path = os.path.join(data_path, class_folder)
            # print(class_folder_path)
            if os.path.isdir(class_folder_path):  # continue to look for images
                for img in os.listdir(class_folder_path):
                    if (
                        img.endswith(".jpg")
                        or img.endswith(".jpeg")
                        or img.endswith(".png")
                    ):
                        img_paths.append(os.path.join(class_folder_path, img))
                        # str_class_folder = str(class_folder).strip().lower()
                        labels.append(class_dict[str(class_folder)])

    except OSError as e:
        print(e)
        print("Error: Could not find  directory")
    return img_paths, labels


def create_datasets(list_path, list_labels):
    # 1.convert to dataframe, path and labels
    # 2.First split training+val and test
    # 3.. Second split: train and validation
    # create csv files: train, test, val
    # return path to csv files

    # Convert to DataFrame class for easier handling
    df = pd.DataFrame({"path": list_path, "label": list_labels})
    # X=df
    # First split: split into training+val and testing
    temp_df, test_df = train_test_split(
        df, test_size=3000 / len(df), stratify=df["label"], random_state=42
    )

    # Second split: split training+val into training and validation
    train_df, val_df = train_test_split(
        temp_df,
        test_size=2000 / len(temp_df),
        stratify=temp_df["label"],
        random_state=42,
    )

    return train_df, test_df, val_df


def create_csv_files(train_df, test_df, val_df):
    # Create csv files for training, validation and testing sets
    if not (
        os.path.exists("train.csv")
        and os.path.exists("validation.csv")
        and os.path.exists("test.csv")
    ):
        train_df.to_csv("train.csv", index=False)
        val_df.to_csv("validation.csv", index=False)
        test_df.to_csv("test.csv", index=False)
        print("csv-files were created")

    print("The necessary csv-files exist ✅")
    # Check for overlapping files - Part b)
    assert set(train_df["path"]).isdisjoint(set(val_df["path"]))
    assert set(train_df["path"]).isdisjoint(set(test_df["path"]))
    assert set(val_df["path"]).isdisjoint(set(test_df["path"]))
    print("Dataset splits are disjoint ✅")

    root = os.getcwd()

    return [
        os.path.join(root, "train.csv"),
        os.path.join(root, "test.csv"),
        os.path.join(root, "validation.csv"),
    ]  # return path to cvs files
