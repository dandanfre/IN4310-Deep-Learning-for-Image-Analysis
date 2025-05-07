import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from NatureTypesDataset import NatureTypesDataset
import torch.nn as nn
from ResNet import ResNet
import os
import matplotlib.pyplot as plt
from torchvision.models import resnet18
from torch.utils.data import DataLoader
import numpy as np

# Resize all images to 80x80 as this decreases the training time, convert image to tensor, normalize image
transform = transforms.Compose(
    [
        transforms.ToPILImage(),  # Convert numpy image to PIL image
        transforms.Resize((80, 80)),  # resize all images to 80x80 for quicker training
        transforms.ToTensor(),
    ]
)


"""
The code below uses a similar structure to the coded provided in exercise 2 task.
"""


def loss_function(prediction, target):  # Returns softmax cross entropy loss
    loss_function = nn.CrossEntropyLoss()
    return loss_function(prediction, target)


def run_epoch(model, data_loader, optimizer, epoch, config, train=True):
    """
    Args:
        model        (obj): The neural network model - ResNet-18
        epoch        (int): The current epoch - 5
        data_loader  (obj): A pytorch data loader "torch.utils.data.DataLoader"
        optimizer    (obj): A pytorch optimizer "torch.optim" - ADAM
        train        (bool): Whether to use train (update) the model/weights or not.
        config       (dict): Configuration parameters: easy to modify and maintain

    Intermediate:
        totalLoss: (float): The accumulated loss from all batches.
                            Hint: Should be a numpy scalar and not a pytorch scalar

    Returns:
        loss_avg         (float): The average loss of the dataset
        accuracy         (float): The average accuracy of the dataset

    """

    if train == True:
        model.train()
    else:
        model.eval()

    total_loss = 0
    correct = 0

    for batch_idx, data_batch in enumerate(data_loader):
        if config["use_cuda"] == True and torch.cuda.is_available():
            images = data_batch[0].to("cuda")  # send data to GPU
            labels = data_batch[1].to("cuda")  # send data to GPU
        else:
            images = data_batch[0]  # use CPU
            labels = data_batch[1]  # use CPU

        if not train:
            with torch.no_grad():
                prediction = model(images)
                loss = loss_function(prediction, labels)
                total_loss += loss.detach().cpu().numpy()

        elif train:
            prediction = model(images)
            loss = loss_function(prediction, labels)
            total_loss += loss.detach().cpu().numpy()

            optimizer.zero_grad()  # Set gradients to zero
            loss.backward()  # Performs backpropagation, and 3) updating the parameters.')
            optimizer.step()  # Do a gradient descent step by 1

        # Update the number of correct classifications
        predicted_label = prediction.max(1, keepdim=True)[1][:, 0]
        correct += predicted_label.eq(labels).cpu().sum().numpy()

        # Print statistics
        # batchSize = len(labels)
        if batch_idx % config["log_interval"] == 0:
            print(
                f"Epoch={epoch} | {(batch_idx+1)/len(data_loader)*100:.2f}% | acc= {correct/len(data_loader.dataset):.3f} | loss = {loss:.3f}"
            )

        loss_avg = total_loss / len(data_loader)
        accuracy = correct / len(data_loader.dataset)

    return loss_avg, accuracy


# Main function
def main():
    # find root_dir (1) join root path with the train and validation sets containing paths to the images and labels in both training and validation sts(2)
    root_dir = os.getcwd()
    train_csv = os.path.join(root_dir, "train.csv")
    val_csv = os.path.join(root_dir, "validation.csv")

    if os.path.exists(train_csv) and os.path.exists(val_csv):
        print("Files found!")
    else:
        print("Files not found! Check paths.")

    # Customized dataclass for easier dataloading using pytorch-framewor
    train_dataset = NatureTypesDataset(
        csv_file=train_csv, root_dir=root_dir, train=True, transform=transform
    )
    val_dataset = NatureTypesDataset(
        csv_file=val_csv, root_dir=root_dir, train=False, transform=transform
    )

    config = {
        "batch_size": 32,
        "use_cuda": True,  # True=GPU, False=CPU
        "epochs": 10,  # PC do not have GPU, so runs on CPU most likely
        "num_workers": 4,  # Number of CPU cores to use
        "pin_memory": True,  # Pin memory for faster data transfer to GPU
        "log_interval": 10,  # How often to display batch loss during training
        "learningRate": 0.001,  # change to 0.001 and 0.1  for variation training
    }

    # dataloader
    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

    # For troubleshooting correct use of the data loader
    # for X, y in val_loader:                #switch with train_loader
    #   print(f"Shape of X [N,C,H,W]: {X.shape}{X.size}")
    #  print(f"Shape of y: {y.shape}{y.dtype}{y.size}")
    # break

    """
    To save time, but makes the code more unreadable is that user has to manually choose to use the provided ResNet class or torch's implementation.
    This can make the code at sometimes more unreadable and more prone to error.
    Commen out  line 150-153 to use the ResNet-18 model provided by torch
    ---->remove this to run resnet-18(torch)
    # Resnet provided from exercise 5
    model = ResNet(
        img_channels=3, num_layers=18, num_classes=6
    )  # 6 classes = buildings, forest, glacier, mountain, sea, street

    # Initialize resnet18 model with pretrained weights - still needs to be trained on the dataset
    ResNet-18 model with pretrained weights
    """
    model = resnet18(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 6)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config["use_cuda"] == True and torch.cuda.is_available():
        model.to("cuda")
    else:
        model.to("cpu")

    """For experimenting with different optimizers.
    Comment out the Adam optimizer to use the SGD or vice versa. This could been better implemented"""
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learningRate"])
    # optimizer = torch.optim.SGD(
    #   model.parameters(), lr=config["learningRate"], momentum=0.9
    # )

    train_loss = np.zeros(shape=config["epochs"])
    train_acc = np.zeros(shape=config["epochs"])
    val_loss = np.zeros(shape=config["epochs"])
    val_acc = np.zeros(shape=config["epochs"])

    for epoch in range(config["epochs"]):
        train_loss[epoch], train_acc[epoch] = run_epoch(
            model, train_loader, optimizer, epoch, train=True, config=config
        )

        val_loss[epoch], val_acc[epoch] = run_epoch(
            model, val_loader, optimizer, epoch, train=False, config=config
        )

        # Save model only when valiation accuracy is highest
        if epoch == 0 or val_acc[epoch] > val_acc[:epoch].max():
            torch.save(
                model.state_dict(),
                "best_nature_classifier_only_weights_resnet_sgd.pth",
            )
            print(
                f"Epoch {epoch} - Model saved with highest accuracy: {val_acc[epoch]}"
            )

    # After the loop, save the final model with architecture if needed - saves the complete model
    torch.save(model, "final_nature_classifier_full_model_resne_sgd.pth")

    """Used to plot the loss and accuracy graph for training and validation"""
    # Plot the loss and the accuracy in training and validation
    plt.figure(figsize=(18, 16), dpi=80, facecolor="w", edgecolor="k")

    # Plot loss
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(range(config["epochs"]), train_loss, "b", label="train loss")
    ax1.plot(range(config["epochs"]), val_loss, "r", label="validation loss")
    ax1.grid()
    ax1.set_ylabel("Loss", fontsize=18)
    ax1.set_xlabel("Epochs", fontsize=18)
    ax1.legend(loc="upper right", fontsize=16)

    # Plot accuracy
    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(range(config["epochs"]), train_acc, "b", label="train accuracy")
    ax2.plot(range(config["epochs"]), val_acc, "r", label="validation accuracy")
    ax2.grid()
    ax2.set_ylabel("Accuracy", fontsize=18)
    ax2.set_xlabel("Epochs", fontsize=18)
    val_acc_max = np.max(val_acc)
    val_acc_max_ind = np.argmax(val_acc)
    ax2.axvline(
        x=val_acc_max_ind,
        color="g",
        linestyle="--",
        label="Highest validation accuracy",
    )
    ax2.set_title(
        f"Highest validation accuracy = {val_acc_max * 100:.1f}%", fontsize=16
    )
    ax2.legend(loc="lower right", fontsize=16)

    # Save the figure
    plt.tight_layout()  # Adjust spacing
    plt.savefig("training_results_resNet18.png")  # Save the plot as an image
    plt.show()  # Display the plot


if __name__ == "__main__":
    main()  # used to generate and save the trained model
