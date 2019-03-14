import numpy as np
import math
# import cv2
import tensorflow as tf
import os
from sklearn.svm import SVC
import sys
import argparse

layers = tf.keras.layers


def conv_bn(inputs, filters, kernel, stride=1, padding='valid'):
    x = layers.Conv2D(filters=filters, kernel_size=kernel, strides=stride, activation='relu', padding=padding)(inputs)
    x = layers.BatchNormalization()(x)
    return x


def deconv_bn(inputs, filters, kernel, stride=1, padding='valid'):
    x = layers.Conv2DTranspose(filters=filters, kernel_size=kernel, strides=stride, activation='relu', padding=padding)(
        inputs)
    x = layers.BatchNormalization()(x)
    return x


def reconstruction_network(deconv_setting, input_shape=(512,)):
    inputs = layers.Input(shape=input_shape)
    x = inputs
    x = layers.Reshape(target_shape=(1, 1, 512))(x)
    for f, k, s, p in deconv_setting:
        # filter, kernel, stride, padding
        p = 'same' if p else 'valid'
        x = deconv_bn(x, f, k, s, p)

    model = tf.keras.Model(inputs, x)
    return model


def representation_network(invert_res_setting, target_dim=512, input_shape=(64, 64, 3)):
    def _inverted_res_block(inputs, expansion, stride, filters):
        in_channels = inputs.shape.as_list()[-1]
        pointwise_conv_filters = filters
        pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
        x = inputs

        x = layers.Conv2D(expansion * in_channels,
                          kernel_size=1,
                          padding='same',
                          use_bias=False,
                          activation=None)(x)
        x = layers.BatchNormalization(epsilon=1e-3,
                                      momentum=0.999)(x)
        x = layers.ReLU(6.)(x)

        # Depthwise
        if stride == 2:
            x = layers.ZeroPadding2D()(x)
        x = layers.DepthwiseConv2D(kernel_size=3,
                                   strides=stride,
                                   activation=None,
                                   use_bias=False,
                                   padding='same' if stride == 1 else 'valid')(x)
        x = layers.BatchNormalization(epsilon=1e-3, momentum=0.999)(x)

        x = layers.ReLU(6.)(x)

        # Project
        x = layers.Conv2D(pointwise_filters,
                          kernel_size=1,
                          padding='same',
                          use_bias=False,
                          activation=None)(x)
        x = layers.BatchNormalization(epsilon=1e-3, momentum=0.999)(x)

        if in_channels == pointwise_filters and stride == 1:
            return layers.Add()([inputs, x])
        return x

    def _make_divisible(v, divisor, min_value=None):
        '''
            This function is taken from the original tf repo.
            It ensures that all layers have a channel number that is divisible by 8
            It can be seen here:
            https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py

        '''

        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v

    inputs = layers.Input(shape=input_shape)
    x = inputs
    # in_channels = inputs.shape.as_list()[-1]

    # Expand
    x = layers.Conv2D(16,
                      kernel_size=1,
                      padding='same',
                      use_bias=False,
                      activation=None
                      )(x)
    x = layers.BatchNormalization(epsilon=1e-3, momentum=0.999)(x)
    x = layers.ReLU(6.)(x)

    for t, c, n, s in invert_res_setting:
        for i in range(n):
            if i == 0:
                x = _inverted_res_block(inputs=x, expansion=tz, stride=s, filters=c)
            else:
                x = _inverted_res_block(inputs=x, expansion=t, stride=1, filters=c)
    x = conv_bn(x, kernel=1, filters=1280)
    x = layers.AveragePooling2D(8)(x)
    x = layers.Dense(units=target_dim, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.Reshape(target_shape=(512,))(x)

    model = tf.keras.Model(inputs, x)
    return model


def r3_cat_model(target_dim=512):
    deconv_setting = [
        # f, k, s, p
        [128, 4, 2, 0],
        [64, 4, 2, 1],
        [32, 4, 2, 1],
        [8, 4, 2, 1],
        [3, 4, 2, 1]
    ]
    t = 6  # expansion_rate
    inverted_res_setting = [
        # t, c, n, s
        [t, 32, 3, 2],
        [t, 64, 4, 2],
        [t, 96, 3, 1],
        [t, 160, 3, 2],
        [t, 320, 1, 1],
    ]

    feature_inputs = layers.Input(shape=(512,))
    img_inputs = layers.Input(shape=(64, 64, 3,))

    # reconstruction
    reconstruction_model = reconstruction_network(deconv_setting)
    img_restored = reconstruction_model(feature_inputs)

    # concatenate
    z = layers.concatenate([img_inputs, img_restored])

    # representation
    representation_model = representation_network(inverted_res_setting, input_shape=(64, 64, 6), target_dim=target_dim)
    outputs = representation_model(z)

    model = tf.keras.Model(inputs=[feature_inputs, img_inputs], outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss="mse",
                  metrics=[tf.keras.metrics.mse, tf.keras.metrics.kld])
    return model


def dataset_api():
    def _parse_function(src_fea, img_path, tar_fea):
        image_string = tf.read_file(img_path)
        image_decoded = tf.image.decode_jpeg(image_string, channels=3)
        image_decoded = tf.image.convert_image_dtype(image_decoded, tf.float32)
        image_resized = tf.image.resize_images(image_decoded, [64, 64])
        return src_fea, image_resized, tar_fea

    train_embed_v1 = np.load('/data/disk4/gsk/VGGFace2/transformation_embedding_v1.npy')
    print('source feature data loaded')
    train_embed_v2 = np.load('/data/disk4/gsk/VGGFace2/transformation_embedding_v2.npy')
    print('target feature data loaded')
    train_set = np.load('./transformation_train_set.npz')
    train_label_set = train_set['label']

    dataset = tf.data.Dataset.from_tensor_slices((train_embed_v1, train_label_set, train_embed_v2))
    dataset = dataset.shuffle(buffer_size=20000)
    dataset = dataset.repeat()
    dataset = dataset.map(_parse_function)
    dataset = dataset.batch(batch_size=32)
    dataset = dataset.prefetch(1)

    return dataset


def train(transform_model_path, tensorboard_path):
    model = r3_cat_model()
    model.summary()

    # model_path = transform_model_path + "r2an-model-{epoch:02d}-{val_acc:.2f}.hdf5"
    callbacks = [
        # Interrupt training if `val_loss` stops improving for over 2 epochs
        tf.keras.callbacks.EarlyStopping(patience=2),
        # tf.keras.callbacks.ModelCheckpoint(model_path, monitor='loss', verbose=1, save_best_only=False, mode='min'),
        tf.keras.callbacks.TensorBoard(log_dir=tensorboard_path)
    ]

    dataset = dataset_api()

    model.fit(dataset,
              steps_per_epoch=98000,
              epochs=10,
              callbacks=callbacks,
              )
    model.save(transform_model_path)


def r3_model(target_dim=512):
    deconv_setting = [
        # f, k, s, p
        [128, 4, 2, 0],
        [64, 4, 2, 1],
        [32, 4, 2, 1],
        [8, 4, 2, 1],
        [3, 4, 2, 1]
    ]
    t = 6  # expansion_rate
    inverted_res_setting = [
        # t, c, n, s
        [t, 32, 3, 2],
        [t, 64, 4, 2],
        [t, 96, 3, 1],
        [t, 160, 3, 2],
        [t, 320, 1, 1],
    ]
    # x = inputs
    # x = tf.placeholder(tf.float32, shape=(None, 1, 1, 512))
    inputs = layers.Input(shape=(512,))
    x = layers.Reshape(target_shape=(1, 1, 512))(inputs)
    y = reconstruction_network(deconv_setting)(x)
    z = representation_network(inverted_res_setting, target_dim=target_dim)(y)
    outputs = layers.Reshape(target_shape=(512,))(z)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss="mse",
                  metrics=[tf.keras.metrics.mse, tf.keras.metrics.kld])
    return model


def transform_model_training(transform_model_path, tensorboard_path):
    model = r3_model()
    model.summary()

    # model_path = transform_model_path + "r2an-model-{epoch:02d}-{val_acc:.2f}.hdf5"
    callbacks = [
        # Interrupt training if `val_loss` stops improving for over 2 epochs
        tf.keras.callbacks.EarlyStopping(patience=2),
        # tf.keras.callbacks.ModelCheckpoint(model_path, monitor='loss', verbose=1, save_best_only=False, mode='min'),
        tf.keras.callbacks.TensorBoard(log_dir=tensorboard_path)
    ]

    train_embed_v1 = np.load('/data/disk4/gsk/VGGFace2/transformation_embedding_v1.npy')
    print('source feature data loaded')
    train_embed_v2 = np.load('/data/disk4/gsk/VGGFace2/transformation_embedding_v2.npy')
    print('target feature data loaded')

    model.fit(train_embed_v1, train_embed_v2, batch_size=128, epochs=10, callbacks=callbacks,
              validation_split=0.1)
    model.save(transform_model_path)


def get_embedding(transform_model_path, train_emb_path, test_emb_path):
    model = r3_cat_model()
    model.load_weights(transform_model_path)
    model.summary()

    test_set = np.load('./test_set.npz')
    # test_path_set = test_set['path']
    test_label_set = test_set['label']
    train_set = np.load('./train_set.npz')
    # train_path_set = train_set['path']
    train_label_set = train_set['label']

    train_embed = np.load('./train_embedding_v1.npy')
    test_embed = np.load('./test_embedding_v1.npy')

    num_img = len(test_label_set)
    batch_size = 128
    num_batch_per_epoch = int(math.ceil(1.0 * num_img / batch_size))
    emb_array = np.zeros((num_img, 512))
    for i in range(num_batch_per_epoch):
        start_index = i * batch_size
        end_index = min((i + 1) * batch_size, num_img)
        emb_v1 = test_embed[start_index:end_index]
        emb_array[start_index:end_index, :] = model.predict(emb_v1)
        print(end_index)
        # feed_dict = {images_placeholder: images}
        # emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)
    np.save(test_emb_path, emb_array)

    num_img = len(train_label_set)
    batch_size = 128
    num_batch_per_epoch = int(math.ceil(1.0 * num_img / batch_size))
    emb_array = np.zeros((num_img, 512))
    for i in range(num_batch_per_epoch):
        start_index = i * batch_size
        end_index = min((i + 1) * batch_size, num_img)
        emb_v1 = train_embed[start_index:end_index]
        emb_array[start_index:end_index, :] = model.predict(emb_v1)
        print(end_index)
        # feed_dict = {images_placeholder: images}
        # emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)
    np.save(train_emb_path, emb_array)


def svc(train_emb_path, test_emb_path, svc_model_path):
    import joblib
    print('Training classifier')
    train_embed = np.load(train_emb_path)
    train_set = np.load('./train_set.npz')
    train_label = train_set['label']
    model = SVC(kernel='linear', probability=True)
    model.fit(train_embed, train_label)
    joblib.dump(model, svc_model_path)

    print('Testing classifier')
    test_embed = np.load(test_emb_path)
    test_set = np.load('./test_set.npz')
    test_label = test_set['label']

    predictions = model.predict_proba(test_embed)
    best_class_indices = np.argmax(predictions, axis=1)
    best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
    for i in range(len(best_class_indices)):
        print('%4d: %.3f' % (i, best_class_probabilities[i]))
    accuracy = np.mean(np.equal(best_class_indices, test_label))
    print('Accuracy: %.3f' % accuracy)


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.Session(config=config)

    version_type = args.version_name
    transform_model_path = './transform_weights_' + version_type + '.h5'
    train_emb_path = './train_embedding_' + version_type + '.npy'
    test_emb_path = './test_embedding_' + version_type + '.npy'
    svc_model_path = './svc_' + version_type + '.joblib'
    tensorboard_path = './logs/' + version_type + '/'

    train(transform_model_path, tensorboard_path)
    print("model trained")
    get_embedding(transform_model_path, train_emb_path, test_emb_path)
    print("embedding calculated")
    svc(train_emb_path, test_emb_path, svc_model_path)
    print(version_type)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('version_name', type=str, help='version name')

    parser.add_argument('--gpu_id', type=str,
                        help='gpu id', default='2')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
