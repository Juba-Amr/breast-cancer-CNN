import tensorflow as tf
from tensorflow.keras.models import load_model
from preprocessing import preprocessing_function
from separate import crop_slice_image
import numpy as np

def patches_to_tf_dataset(patches_with_names):

    patches = np.array([patch for patch, _ in patches_with_names])
    print(np.shape(patches))
    patches_np = np.stack(patches)  # shape: (num_patches, 50, 50, 3)

    names = np.array([name for _, name in patches_with_names])

    ds = tf.data.Dataset.from_tensor_slices((patches_np, np.zeros(len(patches_np))))
    return ds, names


def pred_results(input_image, model_path, patient_id):
    model = load_model(model_path)

    
    images = crop_slice_image(input_image, patient_id)
    ds, names = patches_to_tf_dataset(images)
    ds = preprocessing_function(ds)
    y = model.predict(ds)

    final_result = [(y[i-1], names[i-1]) for i in range(len(y))]
    return final_result
    

