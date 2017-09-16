from itertools import groupby

from dataset_io import DatasetIO
from hyperparameters import Hyperparameters


class DatasetPreparer(object):
    @staticmethod
    def prepare_dataset(ds):
        return DatasetPreparer.deduplicate(ds)

    @staticmethod
    def deduplicate(a):
        return [k for k, v in groupby(sorted(a))]


if __name__ == "__main__":
    ds = DatasetIO.read_dataset_from_csv_file(Hyperparameters.DATASET_PATH)
    pds = DatasetPreparer.prepare_dataset(ds)
    DatasetIO.write_dataset_to_csv_file(
        pds, Hyperparameters.PREPARED_DATASET_PATH)
