from metaflow import FlowSpec, step, Parameter, JSONType, IncludeFile, card, current
from taxi_modules import init, MODELS, MODEL_LIBRARIES
import json


def make_image_grid(data):
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import ImageGrid

    samples = data[np.random.choice(data.shape[0], 9, replace=False)]
    
    fig = plt.figure(figsize=(4., 4.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
        nrows_ncols=(3, 3),  # creates 2x2 grid of axes
        axes_pad=0.1,  # pad between axes in inch.
    )

    for ax, im in zip(grid, samples):
        ax.imshow(im, cmap="gray")
        ax.set_xticks([])
        ax.set_yticks([])

    return fig


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
        (self.x_train, self.y_train), (self.x_test, self.y_test) = ____
        self.next(self.wrangle)
        
    @step
    def wrangle(self):
        """
        massage data
        """
        import numpy as np
        from tensorflow import keras
        # Model / data parameters
        self.num_classes = ____
        self.input_shape = ____

        # Scale images to the [0, 1] range
        self.x_train = ____
        self.x_test = ____
        # Make sure images have shape (28, 28, 1)
        self.x_train = ____
        self.x_test = ____

        # convert class vectors to binary class matrices
        self.y_train = ____
        self.y_test = ____

        current.card.append(Image.from_matplotlib(make_image_grid(self.x_train)))
        
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

        model = ____
        
        model.____(____)
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