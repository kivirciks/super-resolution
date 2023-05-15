import tensorflow as tf
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.layers import concatenate, Input, Activation, Add, Conv2D, Lambda
from tensorflow.keras.models import Model
import numpy as np
from tensorflow.keras.layers import Input, Activation, Dense, Conv2D, BatchNormalization, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg19 import VGG19
import logging
import os

def get_logger(name, job_dir='.'):
    """ Returns logger that prints on stdout at INFO level and on file at DEBUG level. """
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        # stream handler ensures that logging events are passed to stdout
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch_formatter = logging.Formatter('%(message)s')
        ch.setFormatter(ch_formatter)
        logger.addHandler(ch)
        
        # file handler ensures that logging events are passed to log file
        if not os.path.exists(job_dir):
            os.makedirs(job_dir)
        
        fh = logging.FileHandler(filename=os.path.join(job_dir, 'log_file'))
        fh.setLevel(logging.DEBUG)
        fh_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(fh_formatter)
        logger.addHandler(fh)
    
    return logger

class Cut_VGG19:
    """
    Class object that fetches keras' VGG19 model trained on the imagenet dataset
    and declares <layers_to_extract> as output layers. Used as feature extractor
    for the perceptual loss function.

    Args:
        layers_to_extract: list of layers to be declared as output layers.
        patch_size: integer, defines the size of the input (patch_size x patch_size).

    Attributes:
        loss_model: multi-output vgg architecture with <layers_to_extract> as output layers.
    """
    
    def __init__(self, patch_size, layers_to_extract):
        self.patch_size = patch_size
        self.input_shape = (patch_size,) * 2 + (3,)
        self.layers_to_extract = layers_to_extract
        self.logger = get_logger(__name__)
        
        if len(self.layers_to_extract) > 0:
            self._cut_vgg()
        else:
            self.logger.error('Invalid VGG instantiation: extracted layer must be > 0')
            raise ValueError('Invalid VGG instantiation: extracted layer must be > 0')
    
    def _cut_vgg(self):
        """
        Loads pre-trained VGG, declares as output the intermediate
        layers selected by self.layers_to_extract.
        """
        
        vgg = VGG19(weights='imagenet', include_top=False, input_shape=self.input_shape)
        vgg.trainable = False
        outputs = [vgg.layers[i].output for i in self.layers_to_extract]
        self.model = Model([vgg.input], outputs)
        self.model._name = 'feature_extractor'
        self.name = 'vgg19'  # used in weights naming

class Discriminator:
    """
    Implementation of the discriminator network for the adversarial
    component of the perceptual loss.

    Args:
        patch_size: integer, determines input size as (patch_size, patch_size, 3).
        kernel_size: size of the kernel in the conv blocks.

    Attributes:
        model: Keras model.
        name: name used to identify what discriminator is used during GANs training.
        model._name: identifies this network as the discriminator network
            in the compound model built by the trainer class.
        block_param: dictionary, determines the number of filters and the strides for each
            conv block.

    """
    
    def __init__(self, patch_size, kernel_size=3):
        self.patch_size = patch_size
        self.kernel_size = kernel_size
        self.block_param = {}
        self.block_param['filters'] = (64, 128, 128, 256, 256, 512, 512)
        self.block_param['strides'] = (2, 1, 2, 1, 1, 1, 1)
        self.block_num = len(self.block_param['filters'])
        self.model = self._build_disciminator()
        optimizer = Adam(0.0002, 0.5)
        self.model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        self.model._name = 'discriminator'
        self.name = 'srgan-large'
    
    def _conv_block(self, input, filters, strides, batch_norm=True, count=None):
        """ Convolutional layer + Leaky ReLU + conditional BN. """
        
        x = Conv2D(
            filters,
            kernel_size=self.kernel_size,
            strides=strides,
            padding='same',
            name='Conv_{}'.format(count),
        )(input)
        x = LeakyReLU(alpha=0.2)(x)
        if batch_norm:
            x = BatchNormalization(momentum=0.8)(x)
        return x
    
    def _build_disciminator(self):
        """ Puts the discriminator's layers together. """
        
        HR = Input(shape=(self.patch_size, self.patch_size, 3))
        x = self._conv_block(HR, filters=64, strides=1, batch_norm=False, count=1)
        for i in range(self.block_num):
            x = self._conv_block(
                x,
                filters=self.block_param['filters'][i],
                strides=self.block_param['strides'][i],
                count=i + 2,
            )
        x = Dense(self.block_param['filters'][-1] * 2, name='Dense_1024')(x)
        x = LeakyReLU(alpha=0.2)(x)
        # x = Flatten()(x)
        x = Dense(1, name='Dense_last')(x)
        HR_v_SR = Activation('sigmoid')(x)
        
        discriminator = Model(inputs=HR, outputs=HR_v_SR)
        return discriminator

lr_train_patch_size = 40
layers_to_extract = [5, 9]
scale = 2
hr_train_patch_size = lr_train_patch_size * scale

rrdn  = RRDN(arch_params={'C':4, 'D':3, 'G':64, 'G0':64, 'T':10, 'x':scale}, patch_size=lr_train_patch_size)
f_ext = Cut_VGG19(patch_size=hr_train_patch_size, layers_to_extract=layers_to_extract)
discr = Discriminator(patch_size=hr_train_patch_size, kernel_size=3)

def process_array(image_array, expand=True):
    """ Process a 3-dimensional array into a scaled, 4 dimensional batch of size 1. """
    
    image_batch = image_array / 255.0
    if expand:
        image_batch = np.expand_dims(image_batch, axis=0)
    return image_batch

def process_output(output_tensor):
    """ Transforms the 4-dimensional output tensor into a suitable image format. """
    
    sr_img = output_tensor.clip(0, 1) * 255
    sr_img = np.uint8(sr_img)
    return sr_img


def pad_patch(image_patch, padding_size, channel_last=True):
    """ Pads image_patch with with padding_size edge values. """
    
    if channel_last:
        return np.pad(
            image_patch,
            ((padding_size, padding_size), (padding_size, padding_size), (0, 0)),
            'edge',
        )
    else:
        return np.pad(
            image_patch,
            ((0, 0), (padding_size, padding_size), (padding_size, padding_size)),
            'edge',
        )


def unpad_patches(image_patches, padding_size):
    return image_patches[:, padding_size:-padding_size, padding_size:-padding_size, :]


def split_image_into_overlapping_patches(image_array, patch_size, padding_size=2):
    """ Splits the image into partially overlapping patches.

    The patches overlap by padding_size pixels.

    Pads the image twice:
        - first to have a size multiple of the patch size,
        - then to have equal padding at the borders.

    Args:
        image_array: numpy array of the input image.
        patch_size: size of the patches from the original image (without padding).
        padding_size: size of the overlapping area.
    """
    
    xmax, ymax, _ = image_array.shape
    x_remainder = xmax % patch_size
    y_remainder = ymax % patch_size
    
    # modulo here is to avoid extending of patch_size instead of 0
    x_extend = (patch_size - x_remainder) % patch_size
    y_extend = (patch_size - y_remainder) % patch_size
    
    # make sure the image is divisible into regular patches
    extended_image = np.pad(image_array, ((0, x_extend), (0, y_extend), (0, 0)), 'edge')
    
    # add padding around the image to simplify computations
    padded_image = pad_patch(extended_image, padding_size, channel_last=True)
    
    xmax, ymax, _ = padded_image.shape
    patches = []
    
    x_lefts = range(padding_size, xmax - padding_size, patch_size)
    y_tops = range(padding_size, ymax - padding_size, patch_size)
    
    for x in x_lefts:
        for y in y_tops:
            x_left = x - padding_size
            y_top = y - padding_size
            x_right = x + patch_size + padding_size
            y_bottom = y + patch_size + padding_size
            patch = padded_image[x_left:x_right, y_top:y_bottom, :]
            patches.append(patch)
    
    return np.array(patches), padded_image.shape


def stich_together(patches, padded_image_shape, target_shape, padding_size=4):
    """ Reconstruct the image from overlapping patches.

    After scaling, shapes and padding should be scaled too.

    Args:
        patches: patches obtained with split_image_into_overlapping_patches
        padded_image_shape: shape of the padded image contructed in split_image_into_overlapping_patches
        target_shape: shape of the final image
        padding_size: size of the overlapping area.
    """
    
    xmax, ymax, _ = padded_image_shape
    patches = unpad_patches(patches, padding_size)
    patch_size = patches.shape[1]
    n_patches_per_row = ymax // patch_size
    
    complete_image = np.zeros((xmax, ymax, 3))
    
    row = -1
    col = 0
    for i in range(len(patches)):
        if i % n_patches_per_row == 0:
            row += 1
            col = 0
        complete_image[
        row * patch_size: (row + 1) * patch_size, col * patch_size: (col + 1) * patch_size, :
        ] = patches[i]
        col += 1
    return complete_image[0: target_shape[0], 0: target_shape[1], :]

class ImageModel:
    """ISR models parent class.

    Contains functions that are common across the super-scaling models.
    """
    
    def predict(self, input_image_array, by_patch_of_size=None, batch_size=10, padding_size=2):
        """
        Processes the image array into a suitable format
        and transforms the network output in a suitable image format.

        Args:
            input_image_array: input image array.
            by_patch_of_size: for large image inference. Splits the image into
                patches of the given size.
            padding_size: for large image inference. Padding between the patches.
                Increase the value if there is seamlines.
            batch_size: for large image inferce. Number of patches processed at a time.
                Keep low and increase by_patch_of_size instead.
        Returns:
            sr_img: image output.
        """
        
        if by_patch_of_size:
            lr_img = process_array(input_image_array, expand=False)
            patches, p_shape = split_image_into_overlapping_patches(
                lr_img, patch_size=by_patch_of_size, padding_size=padding_size
            )
            # return patches
            for i in range(0, len(patches), batch_size):
                batch = self.model.predict(patches[i: i + batch_size])
                if i == 0:
                    collect = batch
                else:
                    collect = np.append(collect, batch, axis=0)
            
            scale = self.scale
            padded_size_scaled = tuple(np.multiply(p_shape[0:2], scale)) + (3,)
            scaled_image_shape = tuple(np.multiply(input_image_array.shape[0:2], scale)) + (3,)
            sr_img = stich_together(
                collect,
                padded_image_shape=padded_size_scaled,
                target_shape=scaled_image_shape,
                padding_size=padding_size * scale,
            )
        
        else:
            lr_img = process_array(input_image_array)
            sr_img = self.model.predict(lr_img)[0]
        
        sr_img = process_output(sr_img)
        return sr_img

WEIGHTS_URLS = {
    'gans': {
        'arch_params': {'C': 4, 'D': 3, 'G': 32, 'G0': 32, 'x': 4, 'T': 10},
        'url': 'https://public-asai-dl-models.s3.eu-central-1.amazonaws.com/ISR/rrdn-C4-D3-G32-G032-T10-x4-GANS/rrdn-C4-D3-G32-G032-T10-x4_epoch299.hdf5',
        'name': 'rrdn-C4-D3-G32-G032-T10-x4_epoch299.hdf5',
    },
}

def make_model(arch_params, patch_size):
    """ Returns the model.
    Used to select the model.
    """
    return RRDN(arch_params, patch_size)

def get_network(weights):
    if weights in WEIGHTS_URLS.keys():
        arch_params = WEIGHTS_URLS[weights]['arch_params']
        url = WEIGHTS_URLS[weights]['url']
        name = WEIGHTS_URLS[weights]['name']
    else:
        raise ValueError('Available RRDN network weights: {}'.format(list(WEIGHTS_URLS.keys())))
    c_dim = 3
    kernel_size = 3
    return arch_params, c_dim, kernel_size, url, name

class RRDN(ImageModel):
    """Implementation of the Residual in Residual Dense Network for image super-scaling.

    The network is the one described in https://arxiv.org/abs/1809.00219 (Wang et al. 2018).

    Args:
        arch_params: dictionary, contains the network parameters C, D, G, G0, T, x.
        patch_size: integer or None, determines the input size. Only needed at
            training time, for prediction is set to None.
        beta: float <= 1, scaling parameter for the residual connections.
        c_dim: integer, number of channels of the input image.
        kernel_size: integer, common kernel size for convolutions.
        upscaling: string, 'ups' or 'shuffle', determines which implementation
            of the upscaling layer to use.
        init_val: extreme values for the RandomUniform initializer.
        weights: string, if not empty, download and load pre-trained weights.
            Overrides other parameters.

    Attributes:
        C: integer, number of conv layer inside each residual dense blocks (RDB).
        D: integer, number of RDBs inside each Residual in Residual Dense Block (RRDB).
        T: integer, number or RRDBs.
        G: integer, number of convolution output filters inside the RDBs.
        G0: integer, number of output filters of each RDB.
        x: integer, the scaling factor.
        model: Keras model of the RRDN.
        name: name used to identify what upscaling network is used during training.
        model._name: identifies this network as the generator network
            in the compound model built by the trainer class.
    """
    
    def __init__(
            self, arch_params={}, patch_size=None, beta=0.2, c_dim=3, kernel_size=3, init_val=0.05, weights=''
    ):
        if weights:
            arch_params, c_dim, kernel_size, url, fname = get_network(weights)
        
        self.params = arch_params
        self.beta = beta
        self.c_dim = c_dim
        self.C = self.params['C']
        self.D = self.params['D']
        self.G = self.params['G']
        self.G0 = self.params['G0']
        self.T = self.params['T']
        self.scale = self.params['x']
        self.initializer = RandomUniform(minval=-init_val, maxval=init_val, seed=None)
        self.kernel_size = kernel_size
        self.patch_size = patch_size
        self.model = self._build_rdn()
        self.model._name = 'generator'
        self.name = 'rrdn'
        if weights:
            weights_path = tf.keras.utils.get_file(fname=fname, origin=url)
            self.model.load_weights(weights_path)
    
    def _dense_block(self, input_layer, d, t):
        """
        Implementation of the (Residual) Dense Block as in the paper
        Residual Dense Network for Image Super-Resolution (Zhang et al. 2018).

        Residuals are incorporated in the RRDB.
        d is an integer only used for naming. (d-th block)
        """
        
        x = input_layer
        for c in range(1, self.C + 1):
            F_dc = Conv2D(
                self.G,
                kernel_size=self.kernel_size,
                padding='same',
                kernel_initializer=self.initializer,
                name='F_%d_%d_%d' % (t, d, c),
            )(x)
            F_dc = Activation('relu', name='F_%d_%d_%d_Relu' % (t, d, c))(F_dc)
            x = concatenate([x, F_dc], axis=3, name='RDB_Concat_%d_%d_%d' % (t, d, c))
        
        # DIFFERENCE: in RDN a kernel size of 1 instead of 3 is used here
        x = Conv2D(
            self.G0,
            kernel_size=3,
            padding='same',
            kernel_initializer=self.initializer,
            name='LFF_%d_%d' % (t, d),
        )(x)
        return x
    
    def _RRDB(self, input_layer, t):
        """Residual in Residual Dense Block.

        t is integer, for naming of RRDB.
        beta is scalar.
        """
        
        # SUGGESTION: MAKE BETA LEARNABLE
        x = input_layer
        for d in range(1, self.D + 1):
            LFF = self._dense_block(x, d, t)
            LFF_beta = MultiplyBeta(self.beta)(LFF)
            x = Add(name='LRL_%d_%d' % (t, d))([x, LFF_beta])
        x = MultiplyBeta(self.beta)(x)
        x = Add(name='RRDB_%d_out' % (t))([input_layer, x])
        return x
    
    def _pixel_shuffle(self, input_layer):
        """ PixelShuffle implementation of the upscaling part. """
        x = Conv2D(
            self.c_dim * self.scale ** 2,
            kernel_size=3,
            padding='same',
            kernel_initializer=self.initializer,
            name='PreShuffle',
        )(input_layer)

        return PixelShuffle(self.scale)(x)
    
    def _build_rdn(self):
        LR_input = Input(shape=(self.patch_size, self.patch_size, 3), name='LR_input')
        pre_blocks = Conv2D(
            self.G0,
            kernel_size=self.kernel_size,
            padding='same',
            kernel_initializer=self.initializer,
            name='Pre_blocks_conv',
        )(LR_input)
        # DIFFERENCE: in RDN an extra convolution is present here
        for t in range(1, self.T + 1):
            if t == 1:
                x = self._RRDB(pre_blocks, t)
            else:
                x = self._RRDB(x, t)
        # DIFFERENCE: in RDN a conv with kernel size of 1 after a concat operation is used here
        post_blocks = Conv2D(
            self.G0,
            kernel_size=3,
            padding='same',
            kernel_initializer=self.initializer,
            name='post_blocks_conv',
        )(x)
        # Global Residual Learning
        GRL = Add(name='GRL')([post_blocks, pre_blocks])
        # Upscaling
        PS = self._pixel_shuffle(GRL)
        # Compose SR image
        SR = Conv2D(
            self.c_dim,
            kernel_size=self.kernel_size,
            padding='same',
            kernel_initializer=self.initializer,
            name='SR',
        )(PS)
        return Model(inputs=LR_input, outputs=SR)

class PixelShuffle(tf.keras.layers.Layer):
    def __init__(self, scale, *args, **kwargs):
        super(PixelShuffle, self).__init__(*args, **kwargs)
        self.scale = scale

    def call(self, x):
        return tf.nn.depth_to_space(x, block_size=self.scale, data_format='NHWC')

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'scale': self.scale,
        })
        return config

class MultiplyBeta(tf.keras.layers.Layer):
    def __init__(self, beta, *args, **kwargs):
        super(MultiplyBeta, self).__init__(*args, **kwargs)
        self.beta = beta

    def call(self, x, **kwargs):
        return x * self.beta

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'beta': self.beta,
        })
        return config
