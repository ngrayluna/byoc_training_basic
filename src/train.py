#!/bin/usr/python3
# CNN MNIST classification example.

import tensorflow as tf 
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras import Model

# Placeholder for CLI mode
config = {
    "epochs" : 5
}

class MyModel(Model):
    """docstring for MyModel"""
    def __init__(self, optimizer, loss_object, train_loss, train_metric, test_loss, test_metric):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(units=128, activation='relu')
        self.d2 = Dense(units=10, activation='softmax')

        self.optimizer = optimizer
        self.loss_fn = loss_object
        self.train_loss = train_loss
        self.train_metric = train_metric
        self.test_loss = test_loss
        self.test_metric = test_metric

    def model(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)

    @tf.function
    def train_step(self, images, labels):
        # By default, the resources held by a GradientTape are released as 
        # soon as GradientTape.gradient() method is called. 
        with tf.GradientTape() as tape:
            # Compute prediction
            predictions = self.model(images)
            # Compute loss
            loss = self.loss_fn(labels, predictions)

        # Compute gradient
        # Take the gradient of the loss w.r.t. the trainable_variables
        gradient = tape.gradient(loss, self.trainable_variables)

        # Apply gradient/step
        self.optimizer.apply_gradients(zip(gradient, self.trainable_variables))

        # Keep track of loss and loss metric
        self.train_loss(loss)
        self.train_metric(labels, predictions)

    @tf.function
    def test_step(self, images, labels):
        test_predictions = self.model(images)
        t_loss = self.loss_fn(labels, test_predictions)
        
        self.test_loss(t_loss)
        self.test_metric(labels, test_predictions)

    def fit(self, train, test, epochs):
        for epoch in range(epochs):
            for image,label in train:
                self.train_step(image, label)
            for image,label in test:
                self.test_step(image, label)

            template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
            print(template.format(epoch+1,
                                  self.train_loss.result(),
                                  self.train_metric.result()*100,
                                  self.test_loss.result(),
                                  self.test_metric.result()*100))

            # Reset the metrics for the next epoch
            self.train_loss.reset_states()
            self.train_metric.reset_states()
            self.test_loss.reset_states()
            self.test_metric.reset_states()

def init_hyper_parameters():
    # Make Loss object
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

    # Select the optimizer
    optimizer = tf.keras.optimizers.Adam()

    # Specify metrics for training
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    # Specify metrics for testing
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
    return loss_object, optimizer, train_loss, train_metric, test_loss, test_metric 

def download_data():
    mnist = tf.keras.datasets.mnist

    # Load data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Normalizes
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Add a channels dimension
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]
    return shuffle_data(x_train, x_test, y_train, y_test)

def shuffle_data(x_train, x_test, y_train, y_test):
    # Use tf.data to batch and shuffle the dataset:
    train_ds = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).shuffle(10000).batch(32)

    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)
    return train_ds, test_ds


def main():
    # Initialize hyperparameters and metrics
    loss_object, optimizer, train_loss, train_metric, test_loss, test_metric = init_hyper_parameters()

    # Create an instance of the model
    model = MyModel(loss_object = loss_object,
                    optimizer = optimizer,
                    train_loss = train_loss,
                    train_metric = train_metric,
                    test_loss = test_loss,
                    test_metric = test_metric)
    
    # Load data in 
    train_ds, test_ds = download_data()

    EPOCHS = config["epochs"]

    model.fit(train = train_ds,
              test = test_ds,
              epochs = EPOCHS)

if __name__ == '__main__':
    main()