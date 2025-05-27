import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Activation, AveragePooling2D
from tensorflow.keras.layers import Conv1D, Conv2D, DepthwiseConv2D, BatchNormalization
from tensorflow.keras.layers import Add, Concatenate, Lambda, Input, Permute
from tensorflow.keras.regularizers import L2
from tensorflow.keras.constraints import max_norm
from attention_models import attention_block


# %% The proposed ATCNet model, 22/14 channel mSEM-3
def ATCNet_(n_classes, in_chans=22, in_samples=1125, n_windows=5, attention='mha',
            eegn_F1=16, eegn_D=2, eegn_kernelSize=64, eegn_poolSize=7, eegn_dropout=0.3,
            tcn_depth=2, tcn_kernelSize=4, tcn_filters=32, tcn_dropout=0.3,
            tcn_activation='elu', fuse='average'):
    input_1 = Input(shape=(1, in_chans, in_samples))  # TensorShape([None, 1, 22, 1125])
    input_2 = Permute((3, 2, 1))(input_1)

    dense_weightDecay = 0.5
    conv_weightDecay = 0.009
    conv_maxNorm = 0.6
    from_logits = False

    numFilters = eegn_F1
    F2 = numFilters * eegn_D

    block1 = Conv_block_(input_layer=input_2, F1=eegn_F1, D=eegn_D,
                         kernLength=eegn_kernelSize, poolSize=eegn_poolSize,
                         weightDecay=conv_weightDecay, maxNorm=conv_maxNorm,
                         in_chans=in_chans, dropout=eegn_dropout)

    # Sliding window
    sw_concat = []  # to store concatenated or averaged sliding window outputs
    for i in range(n_windows):
        st = i
        end = block1.shape[1] - n_windows + i + 1
        block2 = block1[:, st:end, :]

        # Attention_model
        if attention is not None:
            if (attention == 'se' or attention == 'cbam'):
                block2 = Permute((2, 1))(block2)  # shape=(None, 32, 16)
                block2 = attention_block(block2, attention)
                block2 = Permute((2, 1))(block2)  # shape=(None, 16, 32)
            else:
                block2 = attention_block(block2, attention)

        # Temporal convolutional network (TCN)
        block3 = TCN_block_(input_layer=block2, input_dimension=F2, depth=tcn_depth,
                            kernel_size=tcn_kernelSize, filters=tcn_filters,
                            weightDecay=conv_weightDecay, maxNorm=conv_maxNorm,
                            dropout=tcn_dropout, activation=tcn_activation)
        # Get feature maps of the last sequence
        block3 = Lambda(lambda x: x[:, -1, :])(block3)

        # Outputs of sliding window: Average_after_dense or concatenate_then_dense
        if (fuse == 'average'):
            sw_concat.append(Dense(n_classes, kernel_regularizer=L2(dense_weightDecay))(block3))
        elif (fuse == 'concat'):
            if i == 0:
                sw_concat = block3
            else:
                sw_concat = Concatenate()([sw_concat, block3])

    if (fuse == 'average'):
        if len(sw_concat) > 1:  # more than one window
            sw_concat = tf.keras.layers.Average()(sw_concat[:])
        else:  # one window (# windows = 1)
            sw_concat = sw_concat[0]
    elif (fuse == 'concat'):
        sw_concat = Dense(n_classes, kernel_regularizer=L2(dense_weightDecay))(sw_concat)

    if from_logits:  # No activation here because we are using from_logits=True
        out = Activation('linear', name='linear')(sw_concat)
    else:  # Using softmax activation
        out = Activation('softmax', name='softmax')(sw_concat)

    return Model(inputs=input_1, outputs=out)


def Conv_block_(input_layer, F1=4, kernLength=64, poolSize=8, D=2, in_chans=22,
                weightDecay=0.009, maxNorm=0.6, dropout=0.25):
    F2 = F1 * D
    block1 = Conv2D(F1, (kernLength, 1), padding='same', data_format='channels_last',
                    kernel_regularizer=L2(weightDecay), name='conv1',
                    kernel_constraint=max_norm(maxNorm, axis=[0, 1, 2]),
                    use_bias=False)(input_layer)
    block1 = BatchNormalization(axis=-1, name='bn1')(block1)  # bn_axis = -1 if data_format() == 'channels_last' else 1

    #     for i in [0,2,3,4]:
    #         if i == 0:
    #             input_cen = block1[:, :, i:i+1, :]
    #         else:
    #             va = block1[:, :, i:i+1, :]
    #             input_cen = Concatenate(axis=2)([input_cen, va])
    #     input_fcl = block1[:, :, 8:11, :]
    #     input_cl = block1[:, :, 14:17, :]

    #     for i in [1,6,7,13]:
    #         if i == 1:
    #             input_cpl = block1[:, :, i:i+1, :]
    #         else:
    #             va = block1[:, :, i:i+1, :]
    #             input_cpl = Concatenate(axis=2)([input_cpl, va])
    #     input_fcr = block1[:, :, 18:22, :]
    #     for i in [5,11,12,17]:
    #         if i == 5:
    #             input_cpr = block1[:, :, i:i+1, :]
    #         else:
    #             va = block1[:, :, i:i+1, :]
    #             input_cpr = Concatenate(axis=2)([input_cpr, va])

    #     block_fc = DepthwiseConv2D((1, 4), depth_multiplier=1, data_format='channels_last',
    #                                depthwise_constraint=max_norm(maxNorm, axis=[0, 1, 2]), use_bias=False)(input_cen)
    #     block_fc = BatchNormalization(axis=-1)(block_fc)
    #     block_c = DepthwiseConv2D((1, 3), depth_multiplier=1, data_format='channels_last',
    #                               depthwise_constraint=max_norm(maxNorm, axis=[0, 1, 2]), use_bias=False)(input_fcl)
    #     block_c = BatchNormalization(axis=-1)(block_c)
    #     block_cp = DepthwiseConv2D((1, 3), depth_multiplier=1, data_format='channels_last',
    #                                depthwise_constraint=max_norm(maxNorm, axis=[0, 1, 2]), use_bias=False)(input_cl)
    #     block_cp = BatchNormalization(axis=-1)(block_cp)
    #     block_p = DepthwiseConv2D((1, 4), depth_multiplier=1, data_format='channels_last',
    #                               depthwise_constraint=max_norm(maxNorm, axis=[0, 1, 2]), use_bias=False)(input_cpl)
    #     block_p = BatchNormalization(axis=-1)(block_p)
    #     block_c1 = DepthwiseConv2D((1, 4), depth_multiplier=1, data_format='channels_last',
    #                               depthwise_constraint=max_norm(maxNorm, axis=[0, 1, 2]), use_bias=False)(input_fcr)
    #     block_c1 = BatchNormalization(axis=-1)(block_c1)
    #     block_p1 = DepthwiseConv2D((1, 4), depth_multiplier=1, data_format='channels_last',
    #                               depthwise_constraint=max_norm(maxNorm, axis=[0, 1, 2]), use_bias=False)(input_cpr)
    #     block_p1 = BatchNormalization(axis=-1)(block_p1)

    for i in [0, 2]:
        if i == 0:
            input_cen = block1[:, :, i:i + 1, :]
        else:
            va = block1[:, :, i:i + 1, :]
            input_cen = Concatenate(axis=2)([input_cen, va])
    input_fcl = block1[:, :, 5:7, :]
    input_cl = block1[:, :, 9:10, :]

    for i in [1, 4, 8]:
        if i == 1:
            input_cpl = block1[:, :, i:i + 1, :]
        else:
            va = block1[:, :, i:i + 1, :]
            input_cpl = Concatenate(axis=2)([input_cpl, va])
    input_fcr = block1[:, :, 11:14, :]
    for i in [3, 7, 10]:
        if i == 3:
            input_cpr = block1[:, :, i:i + 1, :]
        else:
            va = block1[:, :, i:i + 1, :]
            input_cpr = Concatenate(axis=2)([input_cpr, va])

    block_fc = DepthwiseConv2D((1, 2), depth_multiplier=1, data_format='channels_last',
                               depthwise_constraint=max_norm(maxNorm, axis=[0, 1, 2]), use_bias=False)(input_cen)
    block_fc = BatchNormalization(axis=-1)(block_fc)
    block_c = DepthwiseConv2D((1, 2), depth_multiplier=1, data_format='channels_last',
                              depthwise_constraint=max_norm(maxNorm, axis=[0, 1, 2]), use_bias=False)(input_fcl)
    block_c = BatchNormalization(axis=-1)(block_c)
    block_cp = DepthwiseConv2D((1, 1), depth_multiplier=1, data_format='channels_last',
                               depthwise_constraint=max_norm(maxNorm, axis=[0, 1, 2]), use_bias=False)(input_cl)
    block_cp = BatchNormalization(axis=-1)(block_cp)
    block_p = DepthwiseConv2D((1, 3), depth_multiplier=1, data_format='channels_last',
                              depthwise_constraint=max_norm(maxNorm, axis=[0, 1, 2]), use_bias=False)(input_cpl)
    block_p = BatchNormalization(axis=-1)(block_p)
    block_c1 = DepthwiseConv2D((1, 3), depth_multiplier=1, data_format='channels_last',
                               depthwise_constraint=max_norm(maxNorm, axis=[0, 1, 2]), use_bias=False)(input_fcr)
    block_c1 = BatchNormalization(axis=-1)(block_c1)
    block_p1 = DepthwiseConv2D((1, 3), depth_multiplier=1, data_format='channels_last',
                               depthwise_constraint=max_norm(maxNorm, axis=[0, 1, 2]), use_bias=False)(input_cpr)
    block_p1 = BatchNormalization(axis=-1)(block_p1)

    block = [block_fc, block_c, block_cp, block_p, block_c1, block_p1]

    for b in block:
        # b = Reshape((b.shape[1], 1, b.shape[2]))(b)
        block1 = Concatenate(axis=-2)([block1, b])
    #     block1 = attention_block(block1, 'mha')
    block2 = DepthwiseConv2D((1, in_chans + 6), depth_multiplier=D, data_format='channels_last',
                             # depthwise_regularizer=L2(weightDecay),  # name='depthwise_conv2d',
                             depthwise_constraint=max_norm(maxNorm, axis=[0, 1, 2]), use_bias=False)(block1)

    block2 = BatchNormalization(axis=-1)(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((poolSize, 1), data_format='channels_last')(block2)
    block2 = Dropout(dropout)(block2)

    block3 = Conv2D(F2, (16, 1),
                    data_format='channels_last',
                    kernel_regularizer=L2(weightDecay),
                    kernel_constraint=max_norm(maxNorm, axis=[0, 1, 2]),
                    use_bias=False, padding='same')(block2)
    block3 = BatchNormalization(axis=-1)(block3)
    block3 = Activation('elu')(block3)

    block3 = AveragePooling2D((poolSize, 1), data_format='channels_last')(block3)
    block3 = Dropout(dropout)(block3)
    block3 = Lambda(lambda x: x[:, :, -1, :])(block3)
    return block3


def TCN_block_(input_layer, input_dimension, depth, kernel_size, filters, dropout,
               weightDecay=0.009, maxNorm=0.6, activation='relu'):
    block = Conv1D(filters, kernel_size=kernel_size, dilation_rate=1, activation='linear',
                   kernel_regularizer=L2(weightDecay),
                   kernel_constraint=max_norm(maxNorm, axis=[0, 1]),

                   padding='causal', kernel_initializer='he_uniform')(input_layer)
    block = BatchNormalization()(block)
    block = Activation(activation)(block)
    block = Dropout(dropout)(block)
    block = Conv1D(filters, kernel_size=kernel_size, dilation_rate=1, activation='linear',
                   kernel_regularizer=L2(weightDecay),
                   kernel_constraint=max_norm(maxNorm, axis=[0, 1]),

                   padding='causal', kernel_initializer='he_uniform')(block)
    block = BatchNormalization()(block)
    block = Activation(activation)(block)
    block = Dropout(dropout)(block)
    if (input_dimension != filters):
        conv = Conv1D(filters, kernel_size=1,
                      kernel_regularizer=L2(weightDecay),
                      kernel_constraint=max_norm(maxNorm, axis=[0, 1]),

                      padding='same')(input_layer)
        added = Add()([block, conv])
    else:
        added = Add()([block, input_layer])
    out = Activation(activation)(added)

    for i in range(depth - 1):
        block = Conv1D(filters, kernel_size=kernel_size, dilation_rate=2 ** (i + 1), activation='linear',
                       kernel_regularizer=L2(weightDecay),
                       kernel_constraint=max_norm(maxNorm, axis=[0, 1]),

                       padding='causal', kernel_initializer='he_uniform')(out)
        block = BatchNormalization()(block)
        block = Activation(activation)(block)
        block = Dropout(dropout)(block)
        block = Conv1D(filters, kernel_size=kernel_size, dilation_rate=2 ** (i + 1), activation='linear',
                       kernel_regularizer=L2(weightDecay),
                       kernel_constraint=max_norm(maxNorm, axis=[0, 1]),

                       padding='causal', kernel_initializer='he_uniform')(block)
        block = BatchNormalization()(block)
        block = Activation(activation)(block)
        block = Dropout(dropout)(block)
        added = Add()([block, out])
        out = Activation(activation)(added)

    return out
