import random

from dataset_io import DatasetIO
from hyperparameters import Hyperparameters


class DatasetGenerator(object):
    @staticmethod
    def generate_dataset(n):
        return [(DatasetGenerator.generate_dataset_entry()) for _ in range(n)]

    @staticmethod
    def generate_dataset_entry():
        k = 10000
        a = random.randint(-k, k)
        b = random.randint(-k, k)
        x = [a, b]
        y = a + b
        return x, y


if __name__ == "__main__":
    ds = DatasetGenerator.generate_dataset(1000000)
    DatasetIO.write_dataset_to_csv_file(ds, Hyperparameters.DATASET_PATH)
