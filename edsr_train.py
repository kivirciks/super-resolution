import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Add, Conv2D, Input, Lambda
from tensorflow.keras.models import Model

# Непосредственно получение датасета
train = DIV2K(scale=4, downgrade='bicubic', subset='train')
train_ds = train.dataset(batch_size=16, random_transform=True)

# Усреднение RGB каналов
DIV2K_RGB_MEAN = np.array([0.4488, 0.4371, 0.4040]) * 255

# Задание модели EDSR
def edsr(scale, num_filters=64, num_res_blocks=8, res_block_scaling=None):
    x_in = Input(shape=(None, None, 3))
    x = Lambda(normalize)(x_in)

    # Задание слоев сверточной нейронной сети
    x = b = Conv2D(num_filters, 3, padding='same')(x)
    for i in range(num_res_blocks):
        b = res_block(b, num_filters, res_block_scaling)
    b = Conv2D(num_filters, 3, padding='same')(b)
    x = Add()([x, b])

    x = upsample(x, scale, num_filters)
    x = Conv2D(3, 3, padding='same')(x)

    x = Lambda(denormalize)(x)
    return Model(x_in, x, name="edsr")

# Задание выходных слоев EDSR
def res_block(x_in, filters, scaling):
    x = Conv2D(filters, 3, padding='same', activation='relu')(x_in)
    x = Conv2D(filters, 3, padding='same')(x)
    if scaling:
        x = Lambda(lambda t: t * scaling)(x)
    x = Add()([x_in, x])
    return x

# Субпиксельная свертка
def upsample(x, scale, num_filters):
    def upsample_1(x, factor, **kwargs):
        x = Conv2D(num_filters * (factor ** 2), 3, padding='same', **kwargs)(x)
        return Lambda(pixel_shuffle(scale=factor))(x)
    if scale == 2:
        x = upsample_1(x, 2, name='conv2d_1_scale_2')
    elif scale == 3:
        x = upsample_1(x, 3, name='conv2d_1_scale_3')
    elif scale == 4:
        x = upsample_1(x, 2, name='conv2d_1_scale_2')
        x = upsample_1(x, 2, name='conv2d_2_scale_2')
    return x

# Перемешивание пикселей
def pixel_shuffle(scale):
    return lambda x: tf.nn.depth_to_space(x, scale)

# Нормализация
def normalize(x):
    return (x - DIV2K_RGB_MEAN) / 127.5

# Денормализация
def denormalize(x):
    return x * 127.5 + DIV2K_RGB_MEAN

import os
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay

# Создание директории для сохранения весов
weights_dir = 'weights/EDSR'
os.makedirs(weights_dir, exist_ok=True)

# EDSR baseline
model_edsr = edsr(scale=4, num_res_blocks=16)

#  Оптимизатор Адама с планировщиком, который вдвое снижает скорость обучения после 200 000 шагов
optim_edsr = Adam(learning_rate=PiecewiseConstantDecay(boundaries=[200000], values=[1e-4, 5e-5]))

# Компиляция и обучение модели для 300 000 шагов
model_edsr.compile(optimizer=optim_edsr, loss='mean_absolute_error')
model_edsr.fit(train_ds, epochs=300, steps_per_epoch=1000)

# Сохранение весов
model_edsr.save_weights(os.path.join(weights_dir, 'weights-edsr-16-x4.h5'))
