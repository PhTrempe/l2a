import tensorflow as tf
from keras import backend as b
from keras.layers import Dense
from keras.models import Sequential

from hyperparameters import Hyperparameters


class ModelBuilder(object):
    @staticmethod
    def build_model(input_shape, output_shape):
        with tf.name_scope("model"):
            model = Sequential()
            model.add(Dense(8, name="fcl1", input_shape=input_shape))
            model.add(Dense(*output_shape, name="out"))

        with tf.name_scope("optimization"):
            model.compile(
                loss="mse",
                optimizer="adam",
                metrics=["mae", "acc", ModelBuilder.accuracy]
            )

        return model

    @staticmethod
    def get_model_custom_objects():
        return {
            "acc1": ModelBuilder.accuracy
        }

    @staticmethod
    def accuracy(y_observations, y_predictions):
        with tf.name_scope("accuracy"):
            return b.mean(b.abs(b.round(
                y_predictions) - y_observations) <= 0.01)


if __name__ == "__main__":
    model = ModelBuilder.build_model(
        Hyperparameters.INPUT_SHAPE, Hyperparameters.OUTPUT_SHAPE)
    model.summary()
    print(model.get_weights())
