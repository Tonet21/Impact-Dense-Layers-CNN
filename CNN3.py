import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class CNNModel2:
    def __init__(self, input_shape=(64, 64, 3), num_classes=10, learning_rate=0.001):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.model = self.build_model()
    
    def build_model(self):
        model = models.Sequential()
        model.add(layers.Conv2D(128, (3, 3), activation='relu', input_shape=self.input_shape, padding="same", name="conv1"))
        model.add(layers.MaxPooling2D((2, 2), name="pool1"))
        model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=self.input_shape, padding="same", name="conv2"))
        model.add(layers.MaxPooling2D((2, 2), name="pool2"))
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape, padding="same", name="conv3"))
        model.add(layers.MaxPooling2D((2, 2), name="pool3"))
        model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=self.input_shape, padding="same", name="conv4"))
        model.add(layers.MaxPooling2D((2, 2), name="pool4"))
        model.add(layers.Conv2D(8, (3, 3), activation='relu', input_shape=self.input_shape, padding="same", name="conv5"))
        model.add(layers.GlobalAveragePooling2D(name="global_average_pooling"))
        model.add(layers.Dense(8, activation='relu', name="dense1"))
        model.add(layers.Dense(8, activation='relu', name="dense2"))
        model.add(layers.Dense(8, activation='relu', name="dense3"))
        model.add(layers.Dense(8, activation='relu', name="dense4"))
        model.add(layers.Dense(8, activation='relu', name="dense5"))
        model.add(layers.Dense(8, activation='relu', name="dense6"))
        model.add(layers.Dense(8, activation='relu', name="dense7"))
        model.add(layers.Dense(8, activation='relu', name="dense8"))
        model.add(layers.Dense(8, activation='relu', name="dense9"))
        model.add(layers.Dense(8, activation='relu', name="dense10"))
        model.add(layers.Dense(8, activation='relu', name="dense11"))
        model.add(layers.Dense(8, activation='relu', name="dense12"))
        model.add(layers.Dense(self.num_classes, activation='softmax', name="output"))

        optimizer = optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def summary(self):
        self.model.summary()