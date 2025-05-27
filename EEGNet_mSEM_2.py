from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Activation, AveragePooling2D
from tensorflow.keras.layers import Conv2D, SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization, Flatten
from tensorflow.keras.layers import Concatenate, Input, Permute
from tensorflow.keras.regularizers import L2
from tensorflow.keras.constraints import max_norm


def ATCNet_(n_classes, in_chans=22, in_samples=1125, n_windows=5, attention='mha',
            eegn_F1=16, eegn_D=2, eegn_kernelSize=64, eegn_poolSize=7, eegn_dropout=0.3,
            tcn_depth=2, tcn_kernelSize=4, tcn_filters=32, tcn_dropout=0.3,
            tcn_activation='elu', fuse='average'):
    input1 = Input(shape=(1, in_chans, in_samples))
    input2 = Permute((3, 2, 1))(input1)
    regRate = .25

    eegnet = EEGNet(input_layer=input2, F1=eegn_F1, kernLength=eegn_kernelSize, D=eegn_D, Chans=in_chans,
                    dropout=eegn_dropout)
    eegnet = Flatten()(eegnet)
    dense = Dense(n_classes, kernel_constraint=max_norm(regRate))(eegnet)
    softmax = Activation('softmax', name='softmax')(dense)

    return Model(inputs=input1, outputs=softmax)


def EEGNet(input_layer, F1=8, kernLength=64, D=2, Chans=22, dropout=0.25):
    weightDecay = 0.009
    maxNorm = 0.6
    F2 = F1 * D
    block1 = Conv2D(F1, (kernLength, 1), kernel_regularizer=L2(weightDecay),
                    kernel_constraint=max_norm(maxNorm, axis=[0, 1, 2]),
                    padding='same', data_format='channels_last', use_bias=False)(input_layer)
    block1 = BatchNormalization(axis=-1)(block1)

    for i in [0, 3, 9, 15, 19, 21]:
        if i == 0:
            input_cen = block1[:, :, i:i + 1, :]
        else:
            va = block1[:, :, i:i + 1, :]
            input_cen = Concatenate(axis=2)([input_cen, va])
    input_fcl = block1[:, :, 1:3, :]
    input_cl = block1[:, :, 6:9, :]

    for i in [13, 14, 18]:
        if i == 13:
            input_cpl = block1[:, :, i:i + 1, :]
        else:
            va = block1[:, :, i:i + 1, :]
            input_cpl = Concatenate(axis=2)([input_cpl, va])
    input_fcr = block1[:, :, 4:6, :]
    input_cp = block1[:, :, 10:13, :]
    for i in [16, 17, 20]:
        if i == 16:
            input_cpr = block1[:, :, i:i + 1, :]
        else:
            va = block1[:, :, i:i + 1, :]
            input_cpr = Concatenate(axis=2)([input_cpr, va])

    block_fc = DepthwiseConv2D((1, 6), depth_multiplier=1, data_format='channels_last',
                               depthwise_constraint=max_norm(maxNorm, axis=[0, 1, 2]), use_bias=True)(input_cen)
    block_fc = BatchNormalization(axis=-1)(block_fc)
    block_c = DepthwiseConv2D((1, 2), depth_multiplier=1, data_format='channels_last',
                              depthwise_constraint=max_norm(maxNorm, axis=[0, 1, 2]), use_bias=True)(input_fcl)
    block_c = BatchNormalization(axis=-1)(block_c)
    block_cp = DepthwiseConv2D((1, 3), depth_multiplier=1, data_format='channels_last',
                               depthwise_constraint=max_norm(maxNorm, axis=[0, 1, 2]), use_bias=True)(input_cl)
    block_cp = BatchNormalization(axis=-1)(block_cp)
    block_p = DepthwiseConv2D((1, 3), depth_multiplier=1, data_format='channels_last',
                              depthwise_constraint=max_norm(maxNorm, axis=[0, 1, 2]), use_bias=True)(input_cpl)
    block_p = BatchNormalization(axis=-1)(block_p)
    block_c1 = DepthwiseConv2D((1, 2), depth_multiplier=1, data_format='channels_last',
                               depthwise_constraint=max_norm(maxNorm, axis=[0, 1, 2]), use_bias=True)(input_fcr)
    block_c1 = BatchNormalization(axis=-1)(block_c1)
    block_p1 = DepthwiseConv2D((1, 3), depth_multiplier=1, data_format='channels_last',
                               depthwise_constraint=max_norm(maxNorm, axis=[0, 1, 2]), use_bias=True)(input_cpr)
    block_p1 = BatchNormalization(axis=-1)(block_p1)
    block_cp1 = DepthwiseConv2D((1, 3), depth_multiplier=1, data_format='channels_last',
                                depthwise_constraint=max_norm(maxNorm, axis=[0, 1, 2]), use_bias=True)(input_cp)
    block_cp1 = BatchNormalization(axis=-1)(block_cp1)

    block = [block_c, block_cp, block_p, block_c1, block_p1, block_cp1]

    for b in block:
        block_fc = Concatenate(axis=-2)([block_fc, b])
    block2 = DepthwiseConv2D((1, 7), depth_multiplier=D, data_format='channels_last',
                             depthwise_constraint=max_norm(maxNorm, axis=[0, 1, 2]), use_bias=True)(block_fc)

    block2 = BatchNormalization(axis=-1)(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((8, 1), data_format='channels_last')(block2)
    block2 = Dropout(dropout)(block2)
    block3 = SeparableConv2D(F2, (16, 1), kernel_regularizer=L2(weightDecay),
                             kernel_constraint=max_norm(maxNorm, axis=[0, 1, 2]), data_format='channels_last',
                             use_bias=False, padding='same')(block2)
    block3 = BatchNormalization(axis=-1)(block3)
    block3 = Activation('elu')(block3)
    block3 = AveragePooling2D((8, 1), data_format='channels_last')(block3)
    block3 = Dropout(dropout)(block3)
    return block3
