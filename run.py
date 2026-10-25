
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


from utilities import ConvNet, training, plot # import class and functions from utilities.py


def positive_int(value):
    parsed_value = int(value)
    if parsed_value <= 0:
        raise argparse.ArgumentTypeError("Must be positive")
    return parsed_value


def positive_float(value):
    parsed_value = float(value)
    if not (0 < parsed_value < float('inf')):
        raise argparse.ArgumentTypeError("Must be positive and finite")
    return parsed_value


def main():
    # Command Line Arguments
    parser = argparse.ArgumentParser(description="MNIST CNN Training")

    parser.add_argument('-e', '--num_epochs', type=positive_int,
                        default=10, help="Number of epochs")

    parser.add_argument('-b', '--batch_size', type=positive_int,
                        default=1000, help="Batch size")

    parser.add_argument('-l', '--learning_rate', type=positive_float,
                        default=0.001, help="Learning rate")

    args = parser.parse_args()

    # Device Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load MNIST Dataset
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True,
        transform=transforms.ToTensor(), download=True)

    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False,
        transform=transforms.ToTensor(), download=True)

    # Data Loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize CNN
    model = ConvNet().to(device)

    # Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Train Model
    model, loss_list, acc_train, acc_test = training(
        model, args.num_epochs, train_loader, test_loader,
        criterion, optimizer, device)

    # Plot Results
    plot(loss_list, acc_train, acc_test)


if __name__ == '__main__':
    main()
