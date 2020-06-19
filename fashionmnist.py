import tensorflow as tf
import matplotlib.pyplot as plt

print(tf.__version__)

mnist = tf.keras.datasets.fashion_mnist

(training_images, training_labels), \
(test_images, test_labels) = mnist.load_data()

plt.imshow(training_images[0])
plt.show()
print(training_labels[0])

training_images = training_images / 255.0
test_images = test_images / 255.0

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

model.compile(optimizer=tf.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


class MyCallBack(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('loss') < 0.4:
            print("\n Reached! Callback is applied")
            self.model.stop_training = True


mycallback = MyCallBack()
model.fit(training_images, training_labels, epochs=5, callbacks=[mycallback])
