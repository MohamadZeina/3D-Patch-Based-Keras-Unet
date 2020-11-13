import tensorflow as tf
import pandas as pd 
import numpy as np 
from functools import partial
import keras
import keras.backend as K
from keras import optimizers, regularizers
from keras.models import Sequential, Model, load_model
from keras.layers import (Activation, concatenate, Input, Dense, Dropout,
                          Conv3D, Conv3DTranspose, MaxPooling3D, Flatten,
                          BatchNormalization, UpSampling3D, SpatialDropout3D)
import mo_net_utils as mnu

def modular_unet(dims, modalities=1, classes=4, secondary_input=True, downsamples=2, 
                 convs_per_block=2, learning_rate=0.0001, learning_rate_decay=0.0, 
                 min_channels=8, channel_power=2, input_dropout=0.0, conv_dropout=0.0, 
                 noise_stddev=0.0, deconv_dropout=True, extra_out_layers=1, 
                 pool_dropout=0.0, bottleneck_layers=4, extra_input_conv=True, 
                 per_class_dice=True, training=None):
    
    """Asks for 'training' so that dropout can be kept on if I want to infer confidence"""

    if type(dims) == int:
        dims = [dims, dims, dims]

    #activation_layer = keras.layers.LeakyReLU(alpha=0.3)
    activation_layer = "relu"
    channels = [min_channels * channel_power ** (i) for i in range(downsamples + 1)]

    input_1 = Input(shape=(*dims, modalities))
    if noise_stddev:
        input_1_noise = keras.layers.GaussianNoise(noise_stddev)(input_1) 
    else:
        input_1_noise = input_1

    if secondary_input: 
        input_2 = Input(shape=(*dims, 3))
        concat = concatenate([input_1_noise, input_2], name="concat_inp")
    else:
        concat = input_1

    input_post_do = Dropout(input_dropout)(concat, training = training)
    if extra_input_conv:
        input_post_do = Conv3D(min_channels, 3, padding="same")(input_post_do)
        if type(activation_layer) == str:
            input_post_do = Activation(activation_layer)(input_post_do )
        else:
            input_post_do = activation_layer(input_post_do)

    skips = ["Uninitialised skip output" for i in range(downsamples + 1)]
    poolings = ["Uninitialised pooling output" for i in range(downsamples + 1)]
    upsamples = ["Uninitialised upsampling output" for i in range(downsamples + 1)]

    poolings[0] = input_post_do

    # Downsampling, aka encoder, aka convolutional block
    for i in range(downsamples):
        i += 1
        skips[i - 1], poolings[i] = convolutional_block(
            poolings[i - 1], channels[i - 1], convs_per_block,
            conv_dropout=conv_dropout, pool_dropout=pool_dropout,
            activation=activation_layer, unet=True, training=training)

    # Create bottleneck
    bottleneck_1 = bottleneck(
        poolings[-1], channels[-1], layers = bottleneck_layers,
        conv_dropout=conv_dropout, activation=activation_layer, training=training)
    upsamples[0] = bottleneck_1

    # Disable spatial dropout on the deconvolution block if deconv_dropout False
    if not deconv_dropout: conv_dropout = 0.0

    # Usampling, aka decoder, aka deconvolutional block
    for i in range(downsamples):
        i += 1
        upsamples[i] = deconvolution_block(
            upsamples[i - 1], skips[downsamples - i], channels[downsamples - i], convs_per_block,
            conv_dropout=conv_dropout, activation=activation_layer, training=training)

    upsample_output = upsamples[-1]

    for i in range(extra_out_layers):
        upsample_output = Conv3D(channels[0], 3, padding="same")(upsample_output)

        if type(activation_layer) == str:
            upsample_output = Activation(activation_layer)(upsample_output)
        else:
            upsample_output = activation_layer(upsample_output)

    output = Conv3D(classes, 1, padding="same", activation="softmax")(upsample_output)

    if secondary_input:
        model = Model(inputs=[input_1, input_2], outputs=output)
    else:
        model = Model(inputs=input_1, outputs=output)
    

    adam = keras.optimizers.Adam(lr=learning_rate, decay=learning_rate_decay)
    #SGD = keras.optimizers.SGD(lr=learning_rate, momentum=0.9)

    # Optimising the DICE loss instead:
    if per_class_dice:
        metrics = [dice_0, dice_1, dice_2, dice_3, overall_dice_coef]
    else:
        metrics = [overall_dice_coef]

    model.compile(#loss=combined_dice_cross_entropy_loss, 
                  loss=overall_dice_loss,
                  optimizer=adam,
                  metrics=metrics)

    # Print important options
    print("#" * 8 + " Initialised neural net with the following options:" + "#" * 8)
    print("Parameters: %.3fm" % (int(model.count_params()) / 1000000))
    print("Channels in successive layers: ", channels)
    print("Bottleneck has shape: ", bottleneck_1.shape)
    print("#" * 70)

    return model

