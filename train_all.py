import argparse

from train import train


def main():
    parser = argparse.ArgumentParser(
        prog='CNN Zoo Trainer',
        description='Updates and Trains All CNNs for CIFAR10 and CIFAR100'
    )
    parser.add_argument('iterations', type=int)
    args = parser.parse_args()
    iterations = args.iterations
    model_dataset_pairs = [
        # Workload A
        ("vgg11", "cifar10"),
        ("resnet18", "cifar10"),
        ("vgg16", "cifar10"),
        ("resnet50", "cifar100"),
        ("mobilenetv2", "cifar100"),
        # Workload B
        ("vgg13", "cifar10"),
        ("resnet34", "cifar10"),
        ("vgg19", "cifar10"),
        ("mobilenetv1", "cifar100"),
        ("resnet50", "cifar100")
    ]

    for model_name, dataset_name in model_dataset_pairs:
        train(model_name, dataset_name, iterations, None)


if __name__ == '__main__':
    main()
