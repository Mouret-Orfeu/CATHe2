import tensorflow as tf
from tensorflow.python.client import device_lib

# import tensorrt as trt
# print("TensorRT version: ",trt.__version__)

import tensorflow as tf
print("cuDNN version:", tf.sysconfig.get_build_info()['cudnn_version'])


# Function to get GPU device details
def get_device_name():
    local_device_protos = device_lib.list_local_devices()
    for device in local_device_protos:
        if device.device_type == 'GPU':
            return device.physical_device_desc

print("cuDNN version: ", tf.sysconfig.get_build_info()['cudnn_version'])
print("\033[93mNum GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

if tf.config.experimental.list_physical_devices('GPU'):
    print("TensorFlow is using the GPU")
    gpu_name = get_device_name()
    print(f"GPU in use: {gpu_name}")
else:
    print("TensorFlow is not using the GPU")
