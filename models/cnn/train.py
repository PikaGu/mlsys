from net import Net
from torch.utils.data.dataloader import DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

model = Net()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.MNIST(root=".", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=".", train=False, download=True, transform=transform)
    
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)
    
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    
    for epoch in range(10):
        train_loss = 0.0
        train_acc = 0.0
        test_loss = 0.0
        test_acc = 0.0
        for i, data in enumerate(train_dataloader):
            images, labels = data
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_acc += (predicted == labels).sum().item()
        
        avg_train_loss = train_loss / len(train_dataloader)
        avg_train_acc = train_acc / len(train_dataloader.dataset)        

        with torch.no_grad():
            for data in test_dataloader:
                images, labels = data
                outputs = model(images)
                loss = loss_fn(outputs, labels)

                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                test_acc += (predicted == labels).sum().item()

            avg_test_loss = test_loss / len(test_dataloader)
            avg_test_acc = test_acc / len(test_dataloader.dataset)

        train_losses.append(avg_train_loss)
        train_accs.append(avg_train_acc)
        test_losses.append(avg_test_loss)
        test_accs.append(avg_test_acc)
        print(f'Epoch [{epoch + 1}/10], '
            f'Train Loss: {avg_train_loss:.4f}, Train Accuracy: {avg_train_acc:.4f}, '
            f'Test Loss: {avg_test_loss:.4f}, Test Accuracy: {avg_test_acc:.4f}')
    
    plt.plot(train_losses, label='train loss')
    plt.plot(test_losses, label='test loss')
    plt.plot(train_accs, label='train acc')
    plt.plot(test_accs, label='test acc')
    plt.title('Model training curves')
    plt.xlabel('Epoch')
    plt.ylabel('Performance')
    plt.legend()
    plt.show()
