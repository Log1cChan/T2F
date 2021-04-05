import keras
import tensorflow.compat.v1 as tf
from keras.layers import Dense, Reshape
from keras.models import Sequential
import keras.backend as K

class textTodlatentsModel:
    def __init__(self, sess = None):
        self.sess = tf.get_default_session() if sess is None else sess
        K.set_session(self.sess)

    def build(self):
        model = Sequential()
        model.add(Dense(2304))
        print("Dense NO.1")
        model.add(Dense(4608))
        print("Dense NO.2")
        model.add(Dense(9216))
        print("Dense NO.3")
        model.add(Reshape([18,512]))
        print("Reshape over")
        model.compile(optimizer=keras.optimizers.Adam(lr=0.001), loss=keras.losses.mean_squared_error)
        print("compile over")
        return model
