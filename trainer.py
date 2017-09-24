import os
import warnings

from keras import backend as b
from keras.callbacks import Callback, ModelCheckpoint, TensorBoard
from keras.models import load_model
import tensorflow as tf

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

        cp = ModelCheckpoint(
            filepath=Hyperparameters.MODEL_PATH,
            monitor="val_loss",
            verbose=1,
            save_best_only=True
        )
        tb = TensorBoard(
            log_dir="./logs",
            histogram_freq=1,
            batch_size=Hyperparameters.BATCH_SIZE,
            write_graph=True
        )
        esv = Trainer.EarlyStoppingByValueCallback(
            monitor="val_acc",
            condition=">=",
            value=1.0,
            verbose=1
        )
        nl = Trainer.PrintNewLineCallback()
        cbs = [cp, tb, esv, nl]

        with tf.name_scope("training"):
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

    class EarlyStoppingByValueCallback(Callback):
        def __init__(self, monitor, condition, value=0.00001, verbose=0):
            super(Callback, self).__init__()
            self.monitor = monitor
            self.condition = condition
            self.value = value
            self.verbose = verbose
            if self.condition not in {"<", "<=", ">=", ">"}:
                raise ValueError("condition must be '<', '<=', '>=', or '>'")

        def on_epoch_end(self, epoch, logs=None):
            if logs is None:
                logs = {}
            monitored_val = logs.get(self.monitor)
            if monitored_val is None:
                warnings.warn(
                    "Watching {} for early stopping, but variable unavailable."
                        .format(self.monitor), RuntimeWarning)
            if self.condition == "<" and monitored_val < self.value \
                    or self.condition == "<=" and monitored_val <= self.value \
                    or self.condition == ">=" and monitored_val >= self.value \
                    or self.condition == ">" and monitored_val > self.value:
                self.model.stop_training = True
                if self.verbose > 0:
                    print("Epoch {}: early stopping since condition {} {} {} "
                          "has been met".format(epoch, self.monitor,
                                                self.condition, self.value))


if __name__ == "__main__":
    Trainer.train()
