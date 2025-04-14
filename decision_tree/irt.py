import numpy as np
import tensorflow as tf

from tqdm import tqdm
from multiprocessing import Process, Queue, Value


def loss(y_true, y_pred):
    loss = tf.keras.losses.MeanSquaredError()
    return loss(y_true, y_pred)

class irt(object):
    def irt_four(self, thi, delj, aj):
        sig_thi, sig_delj = tf.math.sigmoid(thi), tf.math.sigmoid(delj)
        param_aj = tf.convert_to_tensor(aj)

        term1 = sig_delj / (1 - sig_delj)
        term2 = (1 - sig_thi) / sig_thi
        mult = tf.tensordot(term1, term2, axes=1)

        est = 1 / (1 + mult ** (param_aj))
        return est

    def fit_four(self, *args):
        (queue, X, n_respondents, n_items, epochs, lr, random_seed) = args

        np.random.seed(random_seed)
        tf.random.set_seed(
            random_seed
        )

        X = tf.constant(
            X
        )

        thi = tf.Variable(
            np.random.beta(1, 1, size=(1, n_respondents)),
            trainable=True, dtype=tf.float32
        )

        delj = tf.Variable(
            np.random.beta(1, 1, size=(n_items, 1)),
            trainable=True, dtype=tf.float32
        )


        aj = tf.Variable(
            # np.ones((n_items, 1)),
            np.random.normal(1, 0.5, size=(n_items, 1)),
            trainable=True, dtype=tf.float32
        )

        t = 0
        self.current_loss = 0
        for _ in tqdm(range(epochs)):

            variables = [
                thi,
                delj,
                aj
            ]

            with tf.GradientTape() as g:
                g.watch(variables)
                pred = self.irt_four(thi=thi, delj=delj, aj=aj)
                old_loss = self.current_loss
                self.current_loss = loss(X, pred)

            gradients = g.gradient(self.current_loss, variables)

            _thi, _delj, _aj = gradients
            aj.assign_sub(tf.math.scalar_mul(lr, _aj))

            thi.assign_sub(tf.math.scalar_mul(lr, _thi))
            delj.assign_sub(tf.math.scalar_mul(lr, _delj))

            t += 1

        abilities = tf.math.sigmoid(thi).numpy().flatten()
        difficulties = tf.math.sigmoid(delj).numpy().flatten()
        discriminations = tf.convert_to_tensor(aj).numpy().flatten()

        parameters = (abilities, difficulties, discriminations)

        queue.put(parameters)


class Beta3(irt):
    def __init__(
        self, learning_rate=1,
        epochs=10000, n_respondents=20,
        n_items=100,
        n_workers=-1,
        random_seed=1
    ):
        self.lr = learning_rate
        self.epochs = epochs
        self.n_respondents = n_respondents
        self.n_items = n_items
        self.n_seed = random_seed
        self.n_workers = n_workers
        self._params = {
            'learning_rate': learning_rate,
            'epochs': epochs,
            'n_respondents': n_respondents,
            'n_items': n_items
        }

    def fit(self, X):
        self.pij = X
        queue = Queue()

        args = (
            queue, X,
            self.n_respondents,
            self.n_items,
            self.epochs,
            self.lr,
            self.n_seed,
        )

        p = Process(target=super().fit_four, args=list(args))
        p.start()
        abi, dif, dis = queue.get()
        p.join()

        self.abilities = abi
        self.difficulties = dif
        self.discriminations = dis

        return self
