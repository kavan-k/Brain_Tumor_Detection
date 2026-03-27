import tensorflow as tf

print("TensorFlow version:", tf.__version__)
print("Built with CUDA:", tf.test.is_built_with_cuda())

gpus = tf.config.list_physical_devices('GPU')
print("GPUs detected:", gpus)

if gpus:
    print("GPU name:", tf.config.experimental.get_device_details(gpus[0])['device_name'])
