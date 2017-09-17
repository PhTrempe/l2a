import numpy

from keras import backend as b

from trainer import Trainer


class Predictor(object):
    @staticmethod
    def predict():
        model = Trainer.load_model()

        x_observations = numpy.array([
            [2, 5],
            [1, 1],
            [-3, 2],
            [42, 3],
            [7, -7]
        ])

        y_predictions = numpy.round(
            model.predict_on_batch(x_observations)).astype("int")

        for x_t, y_h in zip(x_observations, y_predictions):
            print("{} ==> {}".format(x_t, y_h))

        b.clear_session()


if __name__ == "__main__":
    Predictor.predict()
