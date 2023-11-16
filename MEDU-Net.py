from keras.models import Model
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.layers import BatchNormalization, Activation, add


## 构建U-Net网络模型
def conv2d_same(x, filters, num_row, num_col, padding='same', strides=(1, 1), activation='relu', name=None):
    x = Conv2D(filters, (num_row, num_col), strides=strides, padding=padding, use_bias=False)(x)
    x = BatchNormalization(axis=3, scale=False)(x)

    if (activation == None):
        return x

    x = Activation(activation, name=name)(x)

    return x


def googlenet1(inputs):
    goog1_conv1 = conv2d_same(inputs, 16, 1, 1)
    goog1_conv3 = conv2d_same(goog1_conv1, 16, 3, 3)
    goog1_conv3 = conv2d_same(goog1_conv3, 16, 1, 1)
    goog1_conv5 = conv2d_same(goog1_conv1, 16, 5, 5)
    goog1_conv5 = conv2d_same(goog1_conv5, 16, 1, 1)
    # goog1_p1 = MaxPooling2D(pool_size=(1, 1))(inputs)
    # goog1_p1 = conv2d_bn(goog1_p1, 16, 1, 1)
    x00 = concatenate([goog1_conv5, goog1_conv3, goog1_conv1])
    pool1 = MaxPooling2D(pool_size=(2, 2))(x00)
    # (256,256)

    goog2_conv1 = conv2d_same(pool1, 16 * 2, 1, 1)
    goog2_conv3 = conv2d_same(goog2_conv1, 16 * 2, 3, 3)
    goog2_conv3 = conv2d_same(goog2_conv3, 16 * 2, 1, 1)
    goog2_conv5 = conv2d_same(goog2_conv1, 16 * 2, 5, 5)
    goog2_conv5 = conv2d_same(goog2_conv5, 16 * 2, 1, 1)
    # goog2_p1 = MaxPooling2D(pool_size=(1, 1))(pool1)
    # goog2_p1 = conv2d_bn(goog2_p1, 16 * 2, 1, 1)
    x10 = concatenate([goog2_conv5, goog2_conv3, goog2_conv1])
    x01 = Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same')(x10)
    x01 = concatenate([x01, x00])
    pool2 = MaxPooling2D(pool_size=(2, 2))(x10)
    # 128,128

    goog3_conv1 = conv2d_same(pool2, 16 * 4, 1, 1)
    goog3_conv3 = conv2d_same(goog3_conv1, 16 * 4, 3, 3)
    goog3_conv3 = conv2d_same(goog3_conv3, 16 * 4, 1, 1)
    goog3_conv5 = conv2d_same(goog3_conv1, 16 * 4, 5, 5)
    goog3_conv5 = conv2d_same(goog3_conv5, 16 * 4, 1, 1)
    # goog3_p1 = MaxPooling2D(pool_size=(1, 1))(pool2)
    # goog3_p1 = conv2d_bn(goog3_p1, 16 * 4, 1, 1)
    x20 = concatenate([goog3_conv5, goog3_conv3, goog3_conv1])
    x11 = Conv2DTranspose(16 * 2, (3, 3), strides=(2, 2), padding='same')(x20)
    x11 = concatenate([x11, x10])
    x02 = Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same')(x11)
    x02 = concatenate([x01, x02])
    pool3 = MaxPooling2D(pool_size=(2, 2))(x20)
    # 64,64

    goog4_conv1 = conv2d_same(pool3, 16 * 8, 1, 1)
    goog4_conv3 = conv2d_same(goog4_conv1, 16 * 8, 3, 3)
    goog4_conv3 = conv2d_same(goog4_conv3, 16 * 8, 1, 1)
    goog4_conv5 = conv2d_same(goog4_conv1, 16 * 8, 5, 5)
    goog4_conv5 = conv2d_same(goog4_conv5, 16 * 8, 1, 1)
    # goog4_p1 = MaxPooling2D(pool_size=(1, 1))(pool3)
    # goog4_p1 = conv2d_bn(goog4_p1, 16 * 8, 1, 1)
    x30 = concatenate([goog4_conv5, goog4_conv3, goog4_conv1])
    x21 = Conv2DTranspose(16 * 4, (3, 3), strides=(2, 2), padding='same')(x30)
    x21 = concatenate([x21, x20])
    x12 = Conv2DTranspose(16 * 2, (3, 3), strides=(2, 2), padding='same')(x21)
    x12 = concatenate([x12, x11])
    x03 = Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same')(x12)
    x03 = concatenate([x03, x02])
    pool4 = MaxPooling2D(pool_size=(2, 2))(x30)
   #32,32

    goog5_conv1 = conv2d_same(pool4, 16 * 16, 1, 1)
    goog5_conv3 = conv2d_same(goog5_conv1, 16 * 16, 3, 3)
    goog5_conv3 = conv2d_same(goog5_conv3, 16 * 16, 1, 1)
    goog5_conv5 = conv2d_same(goog5_conv1, 16 * 16, 5, 5)
    goog5_conv5 = conv2d_same(goog5_conv5, 16 * 16, 1, 1)
    # goog5_p1 = MaxPooling2D(pool_size=(1, 1))(pool4)
    # goog5_p1 = conv2d_bn(goog5_p1, 16 * 16, 1, 1)
    x40 = concatenate([goog5_conv5, goog5_conv3, goog5_conv1])
    #32,32

    x311 = Conv2DTranspose(16 * 8, (3, 3), strides=(1, 1), padding='same')(x40)
    x312 = Conv2DTranspose(16 * 8, (3, 3), strides=(1, 1), padding='same')(x40)
    x31 = concatenate([x311, x312])
    x31 = Conv2DTranspose(16 * 8, (3, 3), strides=(2, 2), padding='same')(x31)
    x31 = concatenate([x31, x30], axis=3)

    x221 = Conv2DTranspose(16 * 4, (3, 3), strides=(1, 1), padding='same')(x31)
    x222 = Conv2DTranspose(16 * 4, (3, 3), strides=(1, 1), padding='same')(x31)
    x22 = concatenate([x221, x222])
    x22 = Conv2DTranspose(16 * 4, (3, 3), strides=(2, 2), padding='same')(x22)
    x22 = concatenate([x22, x21], axis=3)

    x131 = Conv2DTranspose(16 * 2, (3, 3), strides=(1, 1), padding='same')(x22)
    x132 = Conv2DTranspose(16 * 2, (3, 3), strides=(1, 1), padding='same')(x22)
    x13 = concatenate([x131, x132])
    x13 = Conv2DTranspose(16 * 2, (3, 3), strides=(2, 2), padding='same')(x13)
    x13 = concatenate([x13, x12], axis=3)

    x041 = Conv2DTranspose(16, (3, 3), strides=(1, 1), padding='same')(x13)
    x042 = Conv2DTranspose(16, (3, 3), strides=(1, 1), padding='same')(x13)
    x04 = concatenate([x041, x042])
    x04 = Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same')(x04)
    x04 = concatenate([x04, x03], axis=3)

    conv10 = conv2d_same(x04, 1, 1, 1, activation='sigmoid')

    model = Model(inputs=[inputs], outputs=[conv10])

    return model