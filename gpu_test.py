import tensorflow as tf
print("cuDNN version: ", tf.sysconfig.get_build_info()['cudnn_version'])
print("\033[93mNum GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

if tf.config.experimental.list_physical_devices('GPU'):
    print("TensorFlow is using the GPU")
else:
    print("TensorFlow is not using the GPU")