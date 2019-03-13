import keras_vggface
import tensorflow as tf
import numpy as np
import config_file as conf
import math
import cv2
import model_v2_build
import keras
from keras import backend as K
from sklearn.svm import SVC

version_type = 'v10'
transform_model_path = './transform_weights_' + version_type + '.h5'
train_emb_path = './train_embedding_' + version_type + '.npy'
test_emb_path = './test_embedding_' + version_type + '.npy'
svc_model_path = './svc_' + version_type + '.joblib'

sess = conf.initialize_GPU()


def transform_model_build():
    inputs = keras.layers.Input(shape=(512,), name='base_input')
    h1 = keras.layers.Dense(1024, activation='relu', name='hidden1')(inputs)
    h2 = keras.layers.Dense(1024, activation='relu', name='hidden2')(h1)
    emb = keras.layers.Dense(512, activation='relu', name='embedding')(h2)
    model = keras.models.Model(inputs=inputs, outputs=emb)
    opt = keras.optimizers.Adam()
    model.compile(optimizer=opt,
                  loss='mean_squared_error')
    return model


def transform_model_build_vae():
    def my_loss(input, emb):
        def loss(y_true, y_pred):
            return K.mean(K.square(input - y_pred), axis=-1) + K.mean(K.square(y_true - emb), axis=-1)

        return loss

    inputs = keras.layers.Input(shape=(512,), name='base_input')
    h1 = keras.layers.Dense(1024, activation='relu', name='hidden1')(inputs)
    h2 = keras.layers.Dense(1024, activation='relu', name='hidden2')(h1)
    emb = keras.layers.Dense(512, activation='relu', name='embedding')(h2)
    rh2 = keras.layers.Dense(1024, activation='relu', name='reverse_hidden2')(emb)
    rh1 = keras.layers.Dense(1024, activation='relu', name='reverse_hidden1')(rh2)
    x = keras.layers.Dense(512, activation='relu', name='reconstruction')(rh1)
    model = keras.models.Model(inputs=inputs, outputs=x)
    opt = keras.optimizers.Adam()
    model.compile(optimizer=opt, loss=my_loss(inputs, emb))
    return model


def transform_model_training():
    train_embed_v1 = np.load('/data/disk4/gsk/VGGFace2/transformation_embedding_v1.npy')

    train_embed_v2 = np.load('/data/disk4/gsk/VGGFace2/transformation_embedding_v2.npy')

    model = transform_model_build_vae()
    model.summary()
    model.fit(train_embed_v1, train_embed_v2, batch_size=128, epochs=50)
    model.save(transform_model_path)


def get_embedding():
    base_model = transform_model_build_vae()
    base_model.load_weights(transform_model_path)
    # base_model = keras.models.load_model(transform_model_path)
    model = keras.Model(inputs=base_model.input, outputs=base_model.get_layer('embedding').output)
    model.summary()

    test_set = np.load('./test_set.npz')
    test_path_set = test_set['path']
    test_label_set = test_set['label']
    train_set = np.load('./train_set.npz')
    train_path_set = train_set['path']
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


def svc():
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
    print(version_type)


def load_classifier_and_evaluate():
    import joblib
    model = joblib.load(svc_model_path)

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
    print(version_type)


if __name__ == '__main__':
    transform_model_training()
    # print('training complete')
    # get_embedding()
    # print('feature extracted')
    # svc()
    # load_classifier_and_evaluate()
