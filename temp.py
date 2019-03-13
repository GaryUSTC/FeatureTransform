import numpy as np
import math
import cv2
import tensorflow as tf
# from tensorflow import keras as keras
from tensorflow.keras import layers
import config_file as conf
from sklearn.model_selection import train_test_split

sess = conf.initialize_GPU()

num_classes = 15
batch_size = 32


def transform_model_build(inputs, num_class=8631):
    # inputs = layers.Input(shape=(224, 224, 3,), name='input')

    # Block 1
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv1')(inputs)
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv1')(x)
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv1')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv2')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv1')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv2')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv1')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv2')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # classification module
    x = layers.Flatten(name='flatten')(x)

    x = layers.Dense(4096, activation='relu', name='fc1')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(4096, activation='relu', name='fc2')(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(num_class, activation='softmax', name='predictions')(x)

    # model = tf.keras.models.Model(inputs=inputs, outputs=outputs, name='vgg16')

    # opt = tf.keras.optimizers.Adam()
    #
    # model.compile(optimizer=opt,
    #               loss="categorical_crossentropy",
    #               metrics=['acc'])
    return outputs


def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image_decoded = tf.image.convert_image_dtype(image_decoded, tf.float32)
    image_resized = tf.image.resize_images(image_decoded, [224, 224])
    label = tf.one_hot(label, num_classes)
    return image_resized, label


def dataset_generator(path_set, label_set, is_training=True):
    # A vector of filenames
    filenames = tf.constant(path_set)
    # `labels[i]` is the label for the image in `filenames[i].
    labels = tf.constant(label_set)

    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    if is_training:
        dataset = dataset.shuffle(buffer_size=20000)
        dataset = dataset.repeat()
    dataset = dataset.map(_parse_function)
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.prefetch(1)
    return dataset


def train_mnist():
    def cnn_layers():
        inputs = layers.Input(shape=(28, 28, 1,), name='input')

        x = layers.Conv2D(32, (3, 3),
                          activation='relu', padding='valid')(inputs)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Flatten()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        predictions = layers.Dense(10,
                                   activation='softmax',
                                   name='x_train_out')(x)
        model = tf.keras.models.Model(inputs=inputs, outputs=predictions, name='vgg16')

        opt = tf.keras.optimizers.Adam()

        model.compile(optimizer=opt,
                      loss="categorical_crossentropy",
                      metrics=['acc'])
        return model

    batch_size = 128
    buffer_size = 10000
    steps_per_epoch = int(np.ceil(60000 / float(batch_size)))  # = 469
    epochs = 5
    num_classes = 10

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype(np.float32) / 255
    x_train = np.expand_dims(x_train, -1)
    y_train = tf.one_hot(y_train, num_classes)

    # Create the dataset and its associated one-shot iterator.
    dataset_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    dataset_train = dataset_train.repeat()
    dataset_train = dataset_train.shuffle(buffer_size)
    dataset_train = dataset_train.batch(batch_size)
    # iterator = dataset.make_one_shot_iterator()

    dataset_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    # dataset_test = dataset_test.repeat()
    # dataset_test = dataset_test.shuffle(buffer_size)
    dataset_test = dataset_test.batch(batch_size)

    model = cnn_layers()
    model.summary()

    filepath = "./models/saved-model-{epoch:02d}-{val_acc:.2f}.hdf5"
    callbacks = [
        # Interrupt training if `val_loss` stops improving for over 2 epochs
        tf.keras.callbacks.EarlyStopping(patience=2, monitor='loss'),
        tf.keras.callbacks.TensorBoard(log_dir='./logs/'),
        tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, mode='max')
    ]

    model.fit(dataset_train.make_one_shot_iterator(), steps_per_epoch=steps_per_epoch, epochs=100,
              validation_data=dataset_test.make_one_shot_iterator(), validation_steps=3,
              callbacks=callbacks)


def train():
    train_set = np.load('./transformation_train_set.npz')
    path_set = train_set['path']
    label_set = train_set['label']

    # first 2000 classes to form a subset
    end_index = np.where(label_set == num_classes)[0][0]
    print("Total training images :  ", end_index)
    path_set = path_set[:end_index]
    label_set = label_set[:end_index]
    steps_per_epoch = int(end_index / batch_size)

    x_train, x_test, y_train, y_test = train_test_split(path_set, label_set, shuffle=True, test_size=0.05,
                                                        random_state=520)

    train_dataset = dataset_generator(x_train, y_train, is_training=True)
    val_dataset = dataset_generator(x_test, y_test, is_training=False)

    train_iter = train_dataset.make_one_shot_iterator()
    inputs, targets = train_iter.get_next()

    # build model
    # model = transform_model_build(num_class=num_classes)
    model_input = layers.Input(tensor=inputs)
    mode_output = transform_model_build(model_input, num_class=num_classes)
    train_model = tf.keras.models.Model(inputs=model_input, outputs=mode_output)
    train_model.compile(optimizer=tf.keras.optimizers.Adam(),
                        loss="categorical_crossentropy",
                        metrics=['acc'])

    train_model.summary()

    # print('data loaded')

    filepath = "./models/saved-model-{epoch:02d}-{val_acc:.2f}.hdf5"
    callbacks = [
        # Interrupt training if `val_loss` stops improving for over 2 epochs
        # tf.keras.callbacks.EarlyStopping(patience=2, monitor='loss'),
        # tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, mode='max'),
        tf.keras.callbacks.TensorBoard(log_dir='./logs/class_10/')
    ]

    train_model.fit(train_dataset.make_one_shot_iterator(), steps_per_epoch=steps_per_epoch, epochs=100,
                    # validation_data=val_dataset, validation_steps=3,
                    callbacks=callbacks)

    # model.save('./models/model_source_v2.h5')
    print("ok")


if __name__ == '__main__':
    a = tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape=None, alpha=1.0, depth_multiplier=1,
                                                       include_top=True,
                                                       weights='imagenet', input_tensor=None, pooling=None,
                                                       classes=1000)

    train()
