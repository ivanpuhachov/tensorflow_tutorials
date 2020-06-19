# wget --no-check-certificate https://storage.googleapis.com/laurencemoroney-blog.appspot.com/horse-or-human.zip
# wget --no-check-certificate     https://storage.googleapis.com/laurencemoroney-blog.appspot.com/validation-horse-or-human.zip
import numpy as np
import random
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array, load_img

train_horse_dir = os.path.join('horse-or-human/horses')
train_human_dir = os.path.join('horse-or-human/humans')

validation_horse_dir = os.path.join('validation-horse-or-human/horses')
validation_human_dir = os.path.join('validation-horse-or-human/humans')

train_horse_names = os.listdir(train_horse_dir)
print(train_horse_names[:10])
train_human_names = os.listdir(train_human_dir)
print(train_human_names[:10])

# Parameters for our graph; we'll output images in a 4x4 configuration
nrows = 4
ncols = 4

# Index for iterating over images
pic_index = 0

# Set up matplotlib fig, and size it to fit 4x4 pics
fig = plt.gcf()
fig.set_size_inches(ncols * 4, nrows * 4)

pic_index += 8
next_horse_pix = [os.path.join(train_horse_dir, fname)
                  for fname in train_horse_names[pic_index - 8:pic_index]]
next_human_pix = [os.path.join(train_human_dir, fname)
                  for fname in train_human_names[pic_index - 8:pic_index]]

for i, img_path in enumerate(next_horse_pix + next_human_pix):
    # Set up subplot; subplot indices start at 1
    sp = plt.subplot(nrows, ncols, i + 1)
    sp.axis('Off')  # Don't show axes (or gridlines)

    img = mpimg.imread(img_path)
    plt.imshow(img)

plt.show()

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3),
                           activation='relu',
                           input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.summary()

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    'horse-or-human/',
    target_size=(300, 300),
    batch_size=128,
    class_mode='binary'
)

validation_datagen = ImageDataGenerator(rescale=1.0/255)
validation_generator = validation_datagen.flow_from_directory(
    'validation-horse-or-human/',
    target_size=(300, 300),
    batch_size=32,
    class_mode='binary'
)

history = model.fit(
    train_generator,
    steps_per_epoch=8,
    epochs=3,
    verbose=1,
    validation_data=validation_generator,
    validation_steps=8
)

# successive_outputs = [
#     layer.output for layer in model.layers[1:]]
#
# visualization_model = tf.keras.models.Model(
#     inputs=model.input,
#     outputs=successive_outputs
# )
#
# # Let's prepare a random input image from the training set.
# horse_img_files = [os.path.join(train_horse_dir, f) for f in train_horse_names]
# human_img_files = [os.path.join(train_human_dir, f) for f in train_human_names]
# img_path = random.choice(horse_img_files + human_img_files)
#
# img = load_img(img_path, target_size=(300, 300))
# x = img_to_array(img)
# x = x.reshape((1,) + x.shape)
# x = x / 255.0
#
# successive_feature_maps = visualization_model.predict(x)
#
# layer_names = [layer.name for layer in model.layers[1:]]

# Now let's display our representations
# for layer_name, feature_map in zip(layer_names, successive_feature_maps):
#     # если это свертки - рисуй
#     if len(feature_map.shape) == 4:
#         n_features = feature_map.shape[-1]
#         size = feature_map.shape[1]
#         display_grid = np.zeros((size, size * n_features))
#         for i in range(n_features):
#             x = feature_map[0, :, :, i]
#             x -= x.mean()
#             # x /= x.std()
#             x *= 64
#             x += 128
#             x = np.clip(x, 0, 255).astype('uint8')
#             display_grid[:, i * size: (i + 1) * size] = x
#         scale = 20. / n_features
#
#         plt.figure(figsize=(scale * n_features, scale))
#         plt.title(layer_name)
#         plt.grid(False)
#         plt.axis('off')
#         plt.imshow(display_grid, aspect='auto', cmap='viridis')
#         plt.show()