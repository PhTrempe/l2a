import os

from keras import backend as b
from keras.callbacks import Callback, ModelCheckpoint
from keras.models import load_model

from dataset_generator import DatasetGenerator
from dataset_io import DatasetIO
from dataset_preparer import DatasetPreparer
from hyperparameters import Hyperparameters
from model_builder import ModelBuilder


class Trainer(object):
    @staticmethod
    def train():
        if not os.path.exists(Hyperparameters.PREPARED_DATASET_PATH):
            if not os.path.exists(Hyperparameters.DATASET_PATH):
                Trainer.generate_dataset()
            Trainer.prepare_dataset()

        pds = DatasetIO.read_dataset_from_csv_file(
            Hyperparameters.PREPARED_DATASET_PATH)

        x_observations, y_observations = [list(x) for x in zip(*pds)]

        model = Trainer.load_model()
        model.summary()

        nl = Trainer.PrintNewLineCallback()
        cp = ModelCheckpoint(
            filepath=Hyperparameters.MODEL_PATH,
            monitor="val_loss",
            verbose=1,
            save_best_only=True
        )
        cbs = [nl, cp]

        model.fit(
            x_observations, y_observations,
            batch_size=Hyperparameters.BATCH_SIZE,
            epochs=Hyperparameters.NUM_EPOCHS,
            validation_split=Hyperparameters.VALIDATION_SPLIT,
            shuffle=True,
            verbose=1,
            callbacks=cbs
        )

        b.clear_session()

    @staticmethod
    def generate_dataset():
        ds = DatasetGenerator.generate_dataset(1000000)
        DatasetIO.write_dataset_to_csv_file(
            ds, Hyperparameters.DATASET_PATH)

    @staticmethod
    def prepare_dataset():
        ds = DatasetIO.read_dataset_from_csv_file(
            Hyperparameters.DATASET_PATH)
        pds = DatasetPreparer.prepare_dataset(ds)
        DatasetIO.write_dataset_to_csv_file(
            pds, Hyperparameters.PREPARED_DATASET_PATH)

    @staticmethod
    def load_model():
        if not os.path.exists(Hyperparameters.MODEL_PATH):
            model = ModelBuilder.build_model(
                Hyperparameters.INPUT_SHAPE, Hyperparameters.OUTPUT_SHAPE)
        else:
            model = load_model(
                Hyperparameters.MODEL_PATH,
                custom_objects=ModelBuilder.get_model_custom_objects()
            )
        return model

    class PrintNewLineCallback(Callback):
        def on_epoch_end(self, epoch, logs=None):
            print()


if __name__ == "__main__":
    Trainer.train()
