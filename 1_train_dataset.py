import argparse
import torchvision

def train(opt):
    epochs = opt.epochs
    batch_size = opt.batch_size
    name = opt.name
    # Train dataset
    transforms = torchvision.transforms.ToTensor()
    train_dataset = torchvision.datasets.CIFAR10('./data', train=True, download=True, 
                                                    transform=transforms)
    classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')                                                
    pass  # debug breakpoint

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200, help='target epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='batch size')
    parser.add_argument('--name', default='ohhan', help='name for the run')

    opt = parser.parse_args()

    train(opt)
