
from metaflow import FlowSpec, step, Parameter, JSONType, IncludeFile, card
from taxi_modules import init, MODELS, MODEL_LIBRARIES
import json


class NNFlow(FlowSpec):
    """
    train a NN
    """
    @card
    @step
    def start(self):
        """
        Load the data
        """
        from tensorflow import keras

        # the data, split between train and test sets
        (self.x_train, self.y_train), (self.x_test, self.y_test) = keras.datasets.mnist.load_data()
        self.next(self.wrangle)
        
    @step
    def wrangle(self):
        """
        massage data
        """
        import numpy as np
        from tensorflow import keras
        # Model / data parameters
        self.num_classes = 10
        self.input_shape = (28, 28, 1)

        # Scale images to the [0, 1] range
        self.x_train = self.x_train.astype("float32") / 255
        self.x_test = self.x_test.astype("float32") / 255
        # Make sure images have shape (28, 28, 1)
        self.x_train = np.expand_dims(self.x_train, -1)
        self.x_test = np.expand_dims(self.x_test, -1)

        # convert class vectors to binary class matrices
        self.y_train = keras.utils.to_categorical(self.y_train, self.num_classes)
        self.y_test = keras.utils.to_categorical(self.y_test, self.num_classes)
        
        self.next(self.build_model)


    @step
    def build_model(self):
        """
        build NN model
        """
        import tempfile
        import numpy as np
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers

        model = keras.Sequential(
            [
                keras.Input(shape=self.input_shape),
                layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Flatten(),
                layers.Dropout(0.5),
                layers.Dense(self.num_classes, activation="softmax"),
            ]
        )
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        with tempfile.NamedTemporaryFile() as f:
            tf.keras.models.save_model(model, f.name, save_format='h5')
            self.model = f.read()
        self.next(self.train)

        
        
    @step
    def train(self):
        """
        Train the model
        """
        import tempfile
        import tensorflow as tf
        self.batch_size = 128
        self.epochs = 15
        
        with tempfile.NamedTemporaryFile() as f:
            f.write(self.model)
            f.flush()
            model =  tf.keras.models.load_model(f.name)
        model.fit(self.x_train, self.y_train, batch_size=self.batch_size, epochs=self.epochs, validation_split=0.1)
        
        self.next(self.end)
        
        
    @step
    def end(self):
        """
        End of flow!
        """
        print("NNFlow is all done.")


if __name__ == "__main__":
    NNFlow()
