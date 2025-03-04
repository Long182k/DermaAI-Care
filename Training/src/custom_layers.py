import tensorflow as tf

class CustomScaleLayer(tf.keras.layers.Layer):
    def __init__(self, scale=1.0, **kwargs):
        super(CustomScaleLayer, self).__init__(**kwargs)
        self.scale_value = scale
        
    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=(1,),
            initializer=tf.keras.initializers.Constant(self.scale_value),
            trainable=True
        )
        
    def call(self, inputs):
        return inputs * self.scale
        
    def get_config(self):
        config = super(CustomScaleLayer, self).get_config()
        config.update({
            'scale': float(self.scale.numpy()) if hasattr(self, 'scale') else self.scale_value
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config) 