import keras_vggface
import tensorflow as tf
import numpy as np
import config_file as conf
import math
import cv2
import model_v2_build
import keras
from sklearn.svm import SVC


def initial_model_v2():
    base_model = model_v2_build.Vggface2_ResNet50(mode='train')
    base_model.load_weights('./weights.h5')
    model = keras.Model(inputs=base_model.input, outputs=base_model.get_layer('dim_proj').output)
    return model


def initial_model_v1():
    model = keras_vggface.VGGFace(include_top=False, input_shape=(224, 224, 3), pooling='avg')
    return model


def get_embedding():
    '''
        use this function to extract features from model v1 or v2
    :return:
    '''
    with tf.Graph().as_default():
        sess = conf.initialize_GPU()

        model = initial_model_v2()
        model.summary()
        images_placeholder = tf.placeholder(tf.float32, shape=(None, 224, 224, 3), name='image_placeholder')
        embeddings = model(images_placeholder)
        embedding_size = embeddings.get_shape()[1]

        # test_set = np.load('./test_set.npz')
        # test_path_set = test_set['path']
        # test_label_set = test_set['label']
        train_set = np.load('./transformation_train_set.npz')
        train_label_set = train_set['label']

        num_img = len(train_label_set)
        batch_size = 16
        num_batch_per_epoch = int(math.ceil(1.0 * num_img / batch_size))
        emb_array = np.zeros((num_img, embedding_size))
        for i in range(num_batch_per_epoch):
            start_index = i * batch_size
            end_index = min((i + 1) * batch_size, num_img)
            paths_batch = train_path_set[start_index:end_index]
            images = load_batch_data(paths_batch)
            emb_array[start_index:end_index, :] = model.predict(images)
            print('v2:  ' + str(end_index))
            # feed_dict = {images_placeholder: images}
            # emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)
        np.save('/data/disk4/gsk/VGGFace2/transformation_embedding_v2', emb_array)

        print('Training classifier')
        # model = SVC(kernel='linear', probability=True)
        # model.fit(emb_array, test_label_set)
        # model.save('./weight.h5')

        # print('Testing classifier')
        # predictions = model.predict_proba(emb_array)
        # best_class_indices = np.argmax(predictions, axis=1)
        # best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]


def svc_v1():
    print('v1')
    import joblib
    print('Training classifier')
    train_embed = np.load('./train_embedding_v1.npy')
    train_set = np.load('./train_set.npz')
    train_label = train_set['label']
    model = SVC(kernel='linear', probability=True)
    model.fit(train_embed, train_label)
    joblib.dump(model, 'svc_v1.joblib')

    print('Testing classifier')
    test_embed = np.load('./test_embedding_v1.npy')
    test_set = np.load('./test_set.npz')
    test_label = test_set['label']

    predictions = model.predict_proba(test_embed)
    best_class_indices = np.argmax(predictions, axis=1)
    best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
    for i in range(len(best_class_indices)):
        print('%4d: %.3f' % (i, best_class_probabilities[i]))
    accuracy = np.mean(np.equal(best_class_indices, labels))
    print('Accuracy: %.3f' % accuracy)


def svc_v2():
    print('v2')
    import joblib
    print('Training classifier')
    train_embed = np.load('./train_embedding_v2.npy')
    train_set = np.load('./train_set.npz')
    train_label = train_set['label']
    model = SVC(kernel='linear', probability=True)
    model.fit(train_embed, train_label)
    joblib.dump(model, 'svc_v2.joblib')

    print('Testing classifier')
    test_embed = np.load('./test_embedding_v2.npy')
    test_set = np.load('./test_set.npz')
    test_label = test_set['label']

    predictions = model.predict_proba(test_embed)
    best_class_indices = np.argmax(predictions, axis=1)
    best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
    for i in range(len(best_class_indices)):
        print('%4d: %.3f' % (i, best_class_probabilities[i]))
    accuracy = np.mean(np.equal(best_class_indices, labels))
    print('Accuracy: %.3f' % accuracy)


def load_classifier_and_evaluate_v1():
    import joblib
    model = joblib.load('./svc_v1.joblib')

    print('Testing classifier')
    test_embed = np.load('./test_embedding_v1.npy')
    test_set = np.load('./test_set.npz')
    test_label = test_set['label']

    predictions = model.predict_proba(test_embed)
    best_class_indices = np.argmax(predictions, axis=1)
    best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
    for i in range(len(best_class_indices)):
        print('%4d: %.3f' % (i, best_class_probabilities[i]))
    accuracy = np.mean(np.equal(best_class_indices, test_label))
    print('Accuracy: %.3f' % accuracy)


def load_classifier_and_evaluate_v2():
    import joblib
    model = joblib.load('./svc_v2.joblib')

    print('Testing classifier')
    test_embed = np.load('./test_embedding_v2.npy')
    test_set = np.load('./test_set.npz')
    test_label = test_set['label']

    predictions = model.predict_proba(test_embed)
    best_class_indices = np.argmax(predictions, axis=1)
    best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
    for i in range(len(best_class_indices)):
        print('%4d: %.3f' % (i, best_class_probabilities[i]))
    accuracy = np.mean(np.equal(best_class_indices, test_label))
    print('Accuracy: %.3f' % accuracy)


def load_data(path, shape=(224, 224, 3)):
    mean = (91.4953, 103.8827, 131.0912)  # vggface2
    # mean = (93.5940, 104.7624, 129.1863)  # vggface
    short_size = 256.0
    img = cv2.imread(path)
    im = np.asarray(img, 'uint8')
    im_size = shape
    im_shape = im.shape
    short_axis = 1
    if im_shape != (256, 256, 3):
        if im_shape[0] < im_shape[1]:
            short_axis = 0
            ratio = float(short_size) / im_shape[0]
        else:
            short_axis = 1
            ratio = float(short_size) / im_shape[1]
        im = cv2.resize(im,
                        (int(max(256.0, round(im_shape[1] * ratio))),
                         int(max(256.0, round(im_shape[0] * ratio)))),
                        interpolation=cv2.INTER_LINEAR)
    # im.shape : w x h x 3
    if short_axis == 0:
        st = 16
        margin = max(0, int((im.shape[1] - im_size[1]) / 2))
        temp = im[st:st + im_size[0], margin:margin + im_size[1], :]
    else:
        st = 16
        margin = max(0, int((im.shape[0] - im_size[0]) / 2))
        temp = im[margin: margin + im_size[0], st:st + im_size[1], :]
    return temp - mean


def load_batch_data(image_paths):
    nrof_samples = len(image_paths)
    images = np.zeros((nrof_samples, 224, 224, 3))
    for i in range(nrof_samples):
        img = load_data(image_paths[i])
        images[i, :, :, :] = img
    return images


get_embedding()