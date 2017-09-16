import csv


class DatasetIO(object):
    @staticmethod
    def write_dataset_to_csv_file(ds, ds_path):
        with open(ds_path, "w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            for x, y in ds:
                writer.writerow(x + [y])

    @staticmethod
    def read_dataset_from_csv_file(csv_file_path):
        with open(csv_file_path, newline="") as csv_file:
            reader = csv.reader(csv_file)
            ds = []
            for row in reader:
                ds.append((row[:2], row[2]))
            return ds
