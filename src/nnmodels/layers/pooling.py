"""
Pooling layers.
"""

try:
    from keras import backend as K
    from keras.layers import Layer
    import numpy as np
except ImportError as e:
    exit("{}: {}".format(__file__, err))


class MaxPoolingWithArgmax2D(Layer):
    """
    TODO
    """

    def __init__(self, pool_size=(2, 2), strides=(2, 2), padding='same', **kwargs):
        super().__init__(**kwargs)
        self.padding = padding
        self.pool_size = pool_size
        self.strides = strides

    def call(self, inputs, **kwargs):
        """
        TODO

        :param inputs:
        :param kwargs:
        :return:
        """
        padding = self.padding
        pool_size = self.pool_size
        strides = self.strides
        if K.backend() == 'tensorflow':
            ksize = [1, pool_size[0], pool_size[1], 1]
            padding = padding.upper()
            strides = [1, strides[0], strides[1], 1]
            output, argmax = K.tf.nn.max_pool_with_argmax(
                inputs,
                ksize=ksize,
                strides=strides,
                padding=padding)
        else:
            errmsg = '{} backend is not supported for layer {}'.format(
                K.backend(), type(self).__name__)
            raise NotImplementedError(errmsg)
        argmax = K.cast(argmax, K.floatx())
        return [output, argmax]

    def compute_output_shape(self, input_shape):
        """
        TODO

        :param input_shape:
        :return:
        """
        ratio = (1, 2, 2, 1)
        output_shape = [
            dim // ratio[idx]
            if dim is not None else None
            for idx, dim in enumerate(input_shape)]
        output_shape = tuple(output_shape)
        return [output_shape, output_shape]

    def compute_mask(self, inputs, mask=None):
        """
        TODO

        :param inputs:
        :param mask:
        :return:
        """
        return 2 * [None]

    def get_config(self):
        """
        Returns the config of the layer.

        A layer config is a Python dictionary (serializable)
        containing the configuration of a layer.
        The same layer can be reinstantiated later
        (without its trained weights) from this configuration.

        # Returns
            Python dictionary.

        From: keras.layers.Layer.get_config documentation.
        """
        config = {'pool_size': self.pool_size,
                  'padding': self.padding,
                  'strides': self.strides
                  }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MaxUnpooling2D(Layer):
    """
    TODO
    """

    def __init__(self, size=(2, 2), **kwargs):
        super().__init__(**kwargs)
        self.size = size

    def call(self, inputs, output_shape=None):
        """
        TODO

        :param inputs:
        :param output_shape:
        :return:
        """
        updates, mask = inputs[0], inputs[1]
        with K.tf.variable_scope(self.name):
            mask = K.cast(mask, 'int32')
            input_shape = K.tf.shape(updates, out_type='int32')

            #  calculation new shape
            if output_shape is None:
                output_shape = (
                    input_shape[0],
                    input_shape[1] * self.size[0],
                    input_shape[2] * self.size[1],
                    input_shape[3])
            self.output_shape1 = output_shape

            # Calculation indices for batch, height, width and feature maps
            one_like_mask = K.ones_like(mask, dtype='int32')
            batch_shape = K.concatenate(
                [[input_shape[0]], [1], [1], [1]],
                axis=0)
            batch_range = K.reshape(
                K.tf.range(output_shape[0], dtype='int32'),
                shape=batch_shape)
            b = one_like_mask * batch_range
            y = mask // (output_shape[2] * output_shape[3])
            x = (mask // output_shape[3]) % output_shape[2]
            feature_range = K.tf.range(output_shape[3], dtype='int32')
            f = one_like_mask * feature_range

            # Transpose indices & reshape update values to one dimension
            updates_size = K.tf.size(updates)
            indices = K.transpose(K.reshape(
                K.stack([b, y, x, f]),
                [4, updates_size]))
            values = K.reshape(updates, [updates_size])
            ret = K.tf.scatter_nd(indices, values, output_shape)
            return ret

    def compute_output_shape(self, input_shape):
        """
        TODO

        :param input_shape:
        :return:
        """
        mask_shape = input_shape[1]
        return (
            mask_shape[0],
            mask_shape[1] * self.size[0],
            mask_shape[2] * self.size[1],
            mask_shape[3]
        )

    def get_config(self):
        """
        Returns the config of the layer.

        A layer config is a Python dictionary (serializable)
        containing the configuration of a layer.
        The same layer can be reinstantiated later
        (without its trained weights) from this configuration.

        # Returns
            Python dictionary.

        From: keras.layers.Layer.get_config documentation.
        """
        config = {'size': self.size,
                  }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


if __name__ == "__main__":
    print("ERROR: this is not the main file of this program.")
