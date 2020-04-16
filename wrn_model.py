from keras.models import Model
import h5py
import random
import keras
import numpy as np
from keras.layers import Input, Add, Activation, Dropout, Flatten, Dense
from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras import backend as K
import tensorflow as tf
from keras import optimizers
from keras.callbacks import Callback
import keras.layers as KL
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.utils import plot_model


def initial_conv(input):
    x = Convolution2D(16, (3, 3), padding='same', kernel_initializer='he_normal',
                      use_bias=False)(input)

    channel_axis = 1 if K.image_data_format() == "channels_first" else -1

    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
    x = Activation('relu')(x)
    return x


def expand_conv(init, base, k, strides=(1, 1)):
    x = Convolution2D(base * k, (3, 3), padding='same', strides=strides, kernel_initializer='he_normal',
                      use_bias=False)(init)

    channel_axis = 1 if K.image_data_format() == "channels_first" else -1

    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
    x = Activation('relu')(x)

    x = Convolution2D(base * k, (3, 3), padding='same', kernel_initializer='he_normal',
                      use_bias=False)(x)

    skip = Convolution2D(base * k, (1, 1), padding='same', strides=strides, kernel_initializer='he_normal',
                      use_bias=False)(init)

    m = Add()([x, skip])

    return m


def conv1_block(input, k=1, dropout=0.0):
    init = input

    channel_axis = 1 if K.image_data_format() == "channels_first" else -1

    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(input)
    x = Activation('relu')(x)
    x = Convolution2D(16 * k, (3, 3), padding='same', kernel_initializer='he_normal',
                      use_bias=False)(x)

    if dropout > 0.0: x = Dropout(dropout)(x)

    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
    x = Activation('relu')(x)
    x = Convolution2D(16 * k, (3, 3), padding='same', kernel_initializer='he_normal',
                      use_bias=False)(x)

    m = Add()([init, x])
    return m

def conv2_block(input, k=1, dropout=0.0):
    init = input

    channel_axis = 1 if K.image_dim_ordering() == "th" else -1

    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(input)
    x = Activation('relu')(x)
    x = Convolution2D(32 * k, (3, 3), padding='same', kernel_initializer='he_normal',
                      use_bias=False)(x)

    if dropout > 0.0: x = Dropout(dropout)(x)

    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
    x = Activation('relu')(x)
    x = Convolution2D(32 * k, (3, 3), padding='same', kernel_initializer='he_normal',
                      use_bias=False)(x)

    m = Add()([init, x])
    return m

def conv3_block(input, k=1, dropout=0.0):
    init = input

    channel_axis = 1 if K.image_dim_ordering() == "th" else -1

    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(input)
    x = Activation('relu')(x)
    x = Convolution2D(64 * k, (3, 3), padding='same', kernel_initializer='he_normal',
                      use_bias=False)(x)

    if dropout > 0.0: x = Dropout(dropout)(x)

    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
    x = Activation('relu')(x)
    x = Convolution2D(64 * k, (3, 3), padding='same', kernel_initializer='he_normal',
                      use_bias=False)(x)

    m = Add()([init, x])
    return m

class WbceLoss(KL.Layer):
    def __init__(self, **kwargs):
        super(WbceLoss, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        """
        # inputs：Input tensor, or list/tuple of input tensors.
        如上，父类KL.Layer的call方法明确要求inputs为一个tensor，或者包含多个tensor的列表/元组
        所以这里不能直接接受多个入参，需要把多个入参封装成列表/元组的形式然后在函数中自行解包，否则会报错。
        """
        # 解包入参
        sup_pred, ori_pred, aug_pred, y_true, step = inputs
        y_true = y_true[:64, :]
        sup_pred = sup_pred[:64, :]
        # data = K.argmax(y_true, axis=-1)
        # 复杂的损失函数
        # sup_loss的形状 = 【batch_size,1】, y_true的形状 = 【batch_size,10】, sup_pred的形状 = 【batch_size,10】
        sup_loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=sup_pred)
        sup_loss, avg_sup_loss = anneal_sup_loss(sup_pred, y_true, sup_loss, step)
        aug_loss = kl_distance(ori_pred, aug_pred)
        aug_loss = get_unsup_loss(aug_loss, ori_pred)
        avg_unsup_loss = K.mean(aug_loss)
        total_loss = avg_sup_loss + avg_unsup_loss
        # 重点：把自定义的loss添加进层使其生效，同时加入metric方便在KERAS的进度条上实时追踪
        self.add_loss(total_loss, inputs=True)
        return total_loss


def create_wide_residual_network(input_dim, nb_classes=10, N=2, k=1, dropout=0.0, verbose=1):
    """
    Creates a Wide Residual Network with specified parameters
    :param input: Input Keras object
    :param nb_classes: Number of output classes
    :param N: Depth of the network. Compute N = (n - 4) / 6.
              Example : For a depth of 16, n = 16, N = (16 - 4) / 6 = 2
              Example2: For a depth of 28, n = 28, N = (28 - 4) / 6 = 4
              Example3: For a depth of 40, n = 40, N = (40 - 4) / 6 = 6
    :param k: Width of the network.
    :param dropout: Adds dropout if value is greater than 0.0
    :param verbose: Debug info to describe created WRN
    :return:
    """
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1

    ip_ori = Input(shape=input_dim)

    x = initial_conv(ip_ori)
    nb_conv = 4

    x = expand_conv(x, 16, k)

    for i in range(N - 1):
        x = conv1_block(x, k, dropout)
        nb_conv += 2

    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
    x = Activation('relu')(x)

    x = expand_conv(x, 32, k, strides=(2, 2))

    for i in range(N - 1):
        x = conv2_block(x, k, dropout)
        nb_conv += 2

    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
    x = Activation('relu')(x)

    x = expand_conv(x, 64, k, strides=(2, 2))

    for i in range(N - 1):
        x = conv3_block(x, k, dropout)
        nb_conv += 2

    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
    x = Activation('relu')(x)

    x = AveragePooling2D((8, 8))(x)
    x = Flatten()(x)
    x = Dense(nb_classes, activation='softmax')(x)

    vision_model = Model(ip_ori, x)
    input_a = Input(shape=input_dim)
    input_b = Input(shape=input_dim)
    input_c = Input(shape=input_dim)
    input_d = Input(shape=(None, ))
    step = Input(shape=(None, ))
    output_a = vision_model(input_a)
    output_b = vision_model(input_b)
    output_c = vision_model(input_c)

    loss = WbceLoss()([output_a, output_b, output_c, input_d, step])
    model_train = Model(inputs=[input_a, input_b, input_c, input_d, step], output=loss)
    model_predict = Model(inputs=input_a, output=output_a)

    if verbose: print("Wide Residual Network-%d-%d created." % (nb_conv, k))
    return model_train, model_predict


class callback4e(Callback):
    def __init__(self):
        self.global_step = 22000
        self.init_lr = 0.03
        self.warmup_lr = 0.0
        self.train_steps = 40000
        self.learning_rate = 0.03
        self.filepath = "D:\\tensorflow\\savemodel\\model-step_{}_loss_{}.h5"

    def get_step(self):
        return self.global_step

    def decay_lr(self):
        lrate = tf.clip_by_value(tf.cast(self.global_step, tf.float32) / self.train_steps, 0, 1)
        decay_lr = self.learning_rate * tf.cos(lrate * (7. / 8) * np.pi / 2)
        learning_rate = tf.where(self.global_step < 0, self.warmup_lr, decay_lr)
        return learning_rate

    def on_train_begin(self, logs=None):
        logs = logs or {}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if self.global_step % 10 == 0:
            K.set_value(self.model.optimizer.lr, K.get_value(self.decay_lr()))
        if self.global_step % 500 == 0:
            loss = logs.get('loss')
            filepath = self.filepath.format(self.global_step, loss)
            self.model.save_weights(filepath, overwrite=True)
        self.global_step += 1





def get_tsa_threshold(schedule, global_step, num_train_steps, start, end):
    step_ratio = global_step / num_train_steps
    if schedule == "linear_schedule":
        coeff = step_ratio
    elif schedule == "exp_schedule":
        scale = 5
    # [exp(-5), exp(0)] = [1e-2, 1], coeff = [1-2.7]
        coeff = K.exp((step_ratio - 1) * scale)
    elif schedule == "log_schedule":
        scale = 5
        # [1 - exp(0), 1 - exp(-5)] = [0, 0.99]
        coeff = 1 - K.exp((-step_ratio) * scale)
    return coeff * (end - start) + start


def anneal_sup_loss(sup_logits, sup_labels, sup_loss, step):
    step = np.amax(step)
    tsa_start = 1. / 10
    train_steps = 40000
    tsa = "log_schedule"
    #eff = [0.1 - 0.99]
    eff_train_prob_threshold = K.cast(get_tsa_threshold(tsa, step, train_steps, tsa_start, end=1), 'float32')
    # one_hot_labels = K.one_hot(sup_labels, 10)
    # sup_probs = K.softmax(sup_logits, axis=-1)
    correct_label_probs = K.sum(sup_labels * sup_logits, axis=-1)
    larger_than_threshold = K.greater(correct_label_probs, eff_train_prob_threshold)
    loss_mask = 1 - K.cast(larger_than_threshold, 'float32')
    loss_mask = K.stop_gradient(loss_mask)
    sup_loss = sup_loss * loss_mask
    avg_sup_loss = (K.sum(sup_loss) / K.maximum(K.sum(loss_mask), 1))
    # metric_dict["sup/sup_trained_ratio"] = K.mean(loss_mask)
    # metric_dict["sup/eff_train_prob_threshold"] = eff_train_prob_threshold
    return sup_loss, avg_sup_loss

def get_unsup_loss(aug_loss, ori_pred):
    largest_prob = tf.reduce_max(ori_pred, axis=-1)
    loss_mask = tf.cast(tf.greater(largest_prob, 0.8), tf.float32)
    loss_mask = tf.stop_gradient(loss_mask)
    aug_loss = aug_loss * loss_mask
    return aug_loss


# def decay_weights(cost, weight_decay_rate, model):
#   """Calculates the loss for l2 weight decay and adds it to `cost`."""
#   costs = []
#   for var in model.trainable_weights:
#     costs.append(K.sum(var**2)/2)
#   cost += tf.multiply(weight_decay_rate, tf.add_n(costs))
#   return cost


#定义新的loss函数
# def new_loss(y_true, y_pred):
#     data = K.argmax(y_true, axis=-1)
#     sup_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=data, logits=y_pred)
#     sup_loss, avg_sup_loss = anneal_sup_loss(y_pred, data, sup_loss, global_step)
#     total_loss = avg_sup_loss + K.get_value(avg_unsup_loss)
#     total_loss = decay_weights(total_loss, weight_decay_rate, wrn_28_2)
#     return total_loss


#计算无监督数据中的扩充数据和原始数据之间的KL距离
def kl_distance(p, q):
    p = p/0.4 + 1e-8
    q = q + 1e-8
    log_p = K.log(p)
    log_q = K.log(q)
    kl = K.sum(p * (log_p - log_q), -1)
    return kl


def merge_input_shape(sup, num):
    new_sup = sup
    for i in range(num - 1):
        new_sup = np.concatenate((new_sup, sup))
    return new_sup



if __name__ == "__main__":
    from keras.utils import np_utils
    from keras.layers import Input
    from keras.models import Model
    from keras.datasets import cifar10


    unsup_radio = 7
    sup_input_dim = (32, 32, 3)
    num_classes = 10
    sup_batch = 64
    unsup_batch = sup_batch * unsup_radio
    epoch = 100
    global_step = 22000
    weight_decay_rate = 5e-4
    train_step_all = 20000
    sup_size = 4000
    # log_dir = "D:\\tensorflow\\log"
    # metric_dict = []
    # global_step = K.variable(0, dtype='int64', name="global_step")

    #数据集分为三部分，原始数据，扩增数据，扩增数据对应原始数据（无标签）
    (_, _), (x_test, y_test) = cifar10.load_data()
    # y_train = np_utils.to_categorical(y_train, num_classes)
    # y_test = np_utils.to_categorical(y_test, num_classes)

    #前者为训练模型，后者为评估模型，共享模型的权重
    wrn_28_2, wrn_28_2_predict = create_wide_residual_network(sup_input_dim, nb_classes=10, N=4, k=2, dropout=0.2)
    wrn_28_2.load_weights('D:\\tensorflow\\savemodel\\model-step_22000_loss_2.9345059394836426.h5', by_name=True)

    wrn_28_2.summary()
    # plot_model(wrn_28_2, to_file='model.png')
    opt = optimizers.SGD(lr=0.03, momentum=0.9, nesterov=True)

    wrn_28_2.compile(loss=lambda y_true, y_pred: y_pred, optimizer=opt)
    wrn_28_2_predict.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
    # checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='min', period=1)
    # tbCallBack = TensorBoard(log_dir=log_dir, update_freq=4480000)
    history = callback4e()

    #训练过程
    for j in range(10, epoch):
        aug_images = np.load("D:\\tensorflow\\cifar_data\\augment_data_{}.npy".format(j))
        ori_images = np.load("D:\\tensorflow\\cifar_data\\x_train.npy")
        x_train = np.load("D:\\tensorflow\\cifar_data\\x_train.npy")
        y_train = np.load("D:\\tensorflow\\cifar_data\\y_train.npy")
        y_train = np_utils.to_categorical(y_train, num_classes)
        sup_start = 0
        unsup_start = 0

        #打乱无监督数据的顺序,相同的seed打乱顺序相同
        # seed = random.randint(0, 50)
        # np.random.seed(seed)
        # np.random.shuffle(ori_images)
        # np.random.seed(seed)
        # np.random.shuffle(aug_images)

        for i in range(200):
            global_step += 1
            X = x_train[sup_start:sup_start + sup_batch, :]
            X = merge_input_shape(X, unsup_radio)
            Y = y_train[sup_start:sup_start + sup_batch, :]
            Y = merge_input_shape(Y, unsup_radio)
            sup_start += sup_batch
            if sup_size - sup_start < sup_batch:
                sup_start = sup_batch + sup_start - sup_size
            aug = aug_images[unsup_start:unsup_start + unsup_batch, :]
            ori = ori_images[unsup_start:unsup_start + unsup_batch, :]
            unsup_start += unsup_batch
            if sup_size - unsup_start < unsup_batch:
                unsup_start = unsup_batch + unsup_start - sup_size
            temp = np.linspace(global_step, global_step, sup_batch).reshape([sup_batch, 1])
            temp = np.asarray(temp, dtype=np.float64)
            temp = merge_input_shape(temp, unsup_radio)

            #向loss添加l2权重衰退
            # costs = []
            # for var in wrn_28_2.trainable_weights:
            #     costs.append(K.sum(var ** 2) / 2)
            # l2_loss = float(K.get_value(tf.multiply(weight_decay_rate, tf.add_n(costs))))
            # temp2 = np.linspace(l2_loss, l2_loss, sup_batch).reshape([sup_batch, 1])
            # temp2 = np.asarray(temp2, dtype=np.float64)
            # temp2 = merge_input_shape(temp2, unsup_radio)


            #进行模型训练
            wrn_28_2.fit([X, ori, aug, Y, temp], Y, verbose=2, shuffle=True, callbacks=[history])


    #评估过程
    # sup_input_dim = (32, 32, 3)
    # (_, _), (x_test, y_test) = cifar10.load_data()
    # wrn_28_2, wrn_28_2_predict = create_wide_residual_network(sup_input_dim, nb_classes=10, N=4, k=2, dropout=0.2)
    # wrn_28_2.load_weights('D:\\tensorflow\\savemodel\\model-step_20500_loss_2.5716272933142528.h5', by_name=True)
    # y_pred = wrn_28_2_predict.predict(x_test)
    # y_pred = np.argmax(y_pred, axis=1)
    # acc = np.sum(y_pred == y_test.flatten())/10000
    # print(acc)
