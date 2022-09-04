from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, BatchNormalization, Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose

def basic_unet(optimizer, loss, metrics, classes, act_fn='relu'):
    inputs = Input((None, None, 1))
    
    start_neurons = 32
    
    #Initialisation Convolution - Start of Block 1 - BatchNorm-Relu-Conv2D order based on https://arxiv.org/pdf/1603.05027v3.pdf - Change to leaky_relu based on https://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf
    conv1 = BatchNormalization()(inputs)
    conv1 = Conv2D(start_neurons * 1, (3, 3), padding='same', kernel_initializer='he_normal')(conv1)

    #Down Convolution Block 1    
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation(act_fn)(conv1)
    conv1 = Conv2D(start_neurons * 1, (3, 3), padding='same', kernel_initializer='he_normal')(conv1)
    
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation(act_fn)(conv1)
    conv1 = Conv2D(start_neurons * 1, (3, 3), padding='same', kernel_initializer='he_normal')(conv1)
    
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    drop1 = Dropout(0.25)(pool1)

    #Down Convolution Block 2
    conv2 = BatchNormalization()(drop1)
    conv2 = Activation(act_fn)(conv2)
    conv2 = Conv2D(start_neurons * 2, (3, 3), padding='same', kernel_initializer='he_normal')(conv2)
    
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation(act_fn)(conv2)
    conv2 = Conv2D(start_neurons * 2, (3, 3), padding='same', kernel_initializer='he_normal')(conv2)
    
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    drop2 = Dropout(0.5)(pool2)

    #Down Convolution Block 3    
    conv3 = BatchNormalization()(drop2)
    conv3 = Activation(act_fn)(conv3)
    conv3 = Conv2D(start_neurons * 4, (3, 3), padding='same', kernel_initializer='he_normal')(conv3)
    
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation(act_fn)(conv3)
    conv3 = Conv2D(start_neurons * 4, (3, 3), padding='same', kernel_initializer='he_normal')(conv3)
    
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    drop3 = Dropout(0.5)(pool3)

    #Down Convolution Block 4    
    conv4 = BatchNormalization()(drop3)
    conv4 = Activation(act_fn)(conv4)
    conv4 = Conv2D(start_neurons * 8, (3, 3), padding='same', kernel_initializer='he_normal')(conv4)
    
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation(act_fn)(conv4)
    conv4 = Conv2D(start_neurons * 8, (3, 3), padding='same', kernel_initializer='he_normal')(conv4)
    
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    drop4 = Dropout(0.5)(pool4)

    #Middle Layer    
    convm = Activation(act_fn)(drop4)
    convm = Conv2D(start_neurons * 16, (3, 3), padding='same', kernel_initializer='he_normal')(convm)
    convm = Activation(act_fn)(convm)
    convm = Conv2D(start_neurons * 16, (3, 3), padding='same', kernel_initializer='he_normal')(convm)

    #Up Convolution Block 4    
    deconv4 = Conv2DTranspose(start_neurons * 8, (2, 2), strides=(2, 2), padding='same', kernel_initializer='he_normal')(convm)
    upconv4 = concatenate([deconv4, conv4])#, axis=3)
    upconv4 = Activation(act_fn)(upconv4)
    upconv4 = Conv2D(start_neurons * 8, (3, 3), padding='same', kernel_initializer='he_normal')(upconv4)
    upconv4 = Activation(act_fn)(upconv4)
    upconv4 = Conv2D(start_neurons * 8, (3, 3), padding='same', kernel_initializer='he_normal')(upconv4)

    #Up Convolution Block 3  
    deconv3 = Conv2DTranspose(start_neurons * 4, (2, 2), strides=(2, 2), padding='same', kernel_initializer='he_normal')(upconv4)
    upconv3 = concatenate([deconv3, conv3])#, axis=3)
    upconv3 = Activation(act_fn)(upconv3)
    upconv3 = Conv2D(start_neurons * 4, (3, 3), padding='same', kernel_initializer='he_normal')(upconv3)
    upconv3 = Activation(act_fn)(upconv3)
    upconv3 = Conv2D(start_neurons * 4, (3, 3), padding='same', kernel_initializer='he_normal')(upconv3)

    #Up Convolution Block 2    
    deconv2 = Conv2DTranspose(start_neurons * 2, (2, 2), strides=(2, 2), padding='same', kernel_initializer='he_normal')(upconv3)
    upconv2 = concatenate([deconv2, conv2])#, axis=3)
    upconv2 = Activation(act_fn)(upconv2)
    upconv2 = Conv2D(start_neurons * 2, (3, 3), padding='same', kernel_initializer='he_normal')(upconv2)
    upconv2 = Activation(act_fn)(upconv2)
    upconv2 = Conv2D(start_neurons * 2, (3, 3), padding='same', kernel_initializer='he_normal')(upconv2)

    #Up Convolution Block 1    
    deconv1 = Conv2DTranspose(start_neurons * 1, (2, 2), strides=(2, 2), padding='same', kernel_initializer='he_normal')(upconv2)
    upconv1 = concatenate([deconv1, conv1])#, axis=3)
    upconv1 = Activation(act_fn)(upconv1)
    upconv1 = Conv2D(start_neurons * 1, (3, 3), padding='same', kernel_initializer='he_normal')(upconv1)
    upconv1 = Activation(act_fn)(upconv1)
    upconv1 = Conv2D(start_neurons * 1, (3, 3), padding='same', kernel_initializer='he_normal')(upconv1)

    output_layer = Conv2D(classes, (1, 1), padding='same', activation='softmax', kernel_initializer='he_normal')(upconv1)

    model = Model(inputs=[inputs], outputs=[output_layer])

    model.compile(optimizer, loss, metrics)

    return model