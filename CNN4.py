import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class CNNModelWithSkipConnections2:
    def __init__(self, input_shape=(64, 64, 3), num_classes=10, learning_rate=0.001):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.model = self.build_model()

    def build_model(self):
        input_layer = layers.Input(shape=self.input_shape, name="input")

                # Block 1
        conv1 = layers.Conv2D(128, (3, 3), activation='relu', padding="same", name="conv1")(input_layer)
        pool1 = layers.MaxPooling2D((2, 2), name="pool1")(conv1)

        # Block 2 with skip connection from Block 1
        conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding="same", name="conv2")(pool1)
        skip_connection2 = layers.Conv2D(64, (1, 1), padding="same", name="skip_connection2")(pool1)
        skip_connection2 = layers.Add()([skip_connection2, conv2])
        pool2 = layers.MaxPooling2D((2, 2), name="pool2")(skip_connection2)

        # Block 3 with skip connection from Block 2
        conv3 = layers.Conv2D(32, (3, 3), activation='relu', padding="same", name="conv3")(pool2)
        skip_connection3 = layers.Conv2D(32, (1, 1), padding="same", name="skip_connection3")(pool2)
        skip_connection3 = layers.Add()([skip_connection3, conv3])
        pool3 = layers.MaxPooling2D((2, 2), name="pool3")(skip_connection3)

        # Block 4 with skip connection from Block 3
        conv4 = layers.Conv2D(16, (3, 3), activation='relu', padding="same", name="conv4")(pool3)
        skip_connection4 = layers.Conv2D(16, (1, 1), padding="same", name="skip_connection4")(pool3)
        skip_connection4 = layers.Add()([skip_connection4, conv4])
        pool4 = layers.MaxPooling2D((2, 2), name="pool4")(skip_connection4)

        # Block 5 with skip connection from Block 4
        conv5 = layers.Conv2D(8, (3, 3), activation='relu', padding="same", name="conv5")(pool4)
        skip_connection5 = layers.Conv2D(8, (1, 1), padding="same", name="skip_connection5")(pool4)
        skip_connection5 = layers.Add()([skip_connection5, conv5])

        # Global average pooling
        global_avg_pooling = layers.GlobalAveragePooling2D(name="global_average_pooling")(skip_connection5)

        # Dense layers
        dense1 = layers.Dense(8, activation='relu', name="dense1")(global_avg_pooling)
        dense2 = layers.Dense(8, activation='relu', name="dense2")(dense1)
        dense3 = layers.Dense(8, activation='relu', name="dense3")(dense2)
        dense4 = layers.Dense(8, activation='relu', name="dense4")(dense3)
        dense5 = layers.Dense(8, activation='relu', name="dense5")(dense4)
        dense6 = layers.Dense(8, activation='relu', name="dense6")(dense5)
        dense7 = layers.Dense(8, activation='relu', name="dense7")(dense6)
        dense8 = layers.Dense(8, activation='relu', name="dense8")(dense7)
        dense9 = layers.Dense(8, activation='relu', name="dense9")(dense8)
        dense10 = layers.Dense(8, activation='relu', name="dense10")(dense9)
        dense11 = layers.Dense(8, activation='relu', name="dense11")(dense10)
        dense12 = layers.Dense(8, activation='relu', name="dense12")(dense11)

        # Output layer
        output = layers.Dense(self.num_classes, activation='softmax', name="output")(dense12)

        model = models.Model(inputs=input_layer, outputs=output, name="CNNModelWithSkipConnections")

        optimizer = optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model


    def summary(self):
        self.model.summary()
