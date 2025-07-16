import tensorflow as tf

def preprocessing_function(data):
    
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    def preprocess(image, label):
        image = tf.image.convert_image_dtype(image, tf.float32) 
        return image, label
    
    X_data = (
        data
        .map(preprocess, num_parallel_calls=AUTOTUNE, deterministic=True)
        .batch(16)
        .cache()
        .prefetch(buffer_size=AUTOTUNE)
    )
    return X_data