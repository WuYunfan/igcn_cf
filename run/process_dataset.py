from dataset import get_dataset
from utils import set_seed


def process_dataset(name):
    dataset_config = {'name': name + 'Dataset', 'path': 'data/' + name,
                      'device': 'cpu', 'split_ratio': [0.7, 0.1, 0.2], 'min_inter': 10}
    dataset = get_dataset(dataset_config)
    dataset.output_dataset('data/' + name + '/time')
    for i in range(5):
        set_seed(2021 + 2 ** i)
        dataset.shuffle = True
        dataset.generate_data()
        dataset.output_dataset('data/' + name + '/' + str(i))


def main():
    process_dataset('Amazon')


if __name__ == '__main__':
    main()
