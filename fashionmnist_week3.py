import tensorflow as tf
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), \
(test_images, test_labels) = mnist.load_data()

training_images = training_images.reshape(60000,28,28,1)
training_images = training_images / 255.0

test_images = test_images.reshape(10000,28,28,1)
test_images=test_images / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=64,
                           kernel_size=(3,3),
                           activation='relu',
                           input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

model.fit(training_images, training_labels,
          epochs=5, batch_size=64)

f, axarr = plt.subplots(2, 4)
layer_outputs = [layer.output for layer in model.layers]
activation_model = tf.keras.models.Model(
    inputs=model.input, outputs=layer_outputs)
for x in range(0,4):
    f1 = activation_model.predict(
        test_images[0].reshape(1,28,28,1))[x]
    axarr[0,x].imshow(f1[0,:,:,1], cmap='inferno')
    axarr[0,x].grid(False)
    f2 = activation_model.predict(
        test_images[10].reshape(1, 28, 28, 1))[x]
    axarr[1, x].imshow(f2[0, :, :, 1], cmap='inferno')
    axarr[1, x].grid(False)
plt.show()