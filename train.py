import argparse

def train(opt):
    epochs = opt.epochs
    batch_size = opt.batch_size
    name = opt.name
    print(epochs)
    print(batch_size)
    print(name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200, help='target epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='batch size')
    parser.add_argument('--name', default='ohhan', help='name for the run')

    opt = parser.parse_args()

    train(opt)
