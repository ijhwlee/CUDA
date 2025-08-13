import tensorflow as tf

print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

print(tf.test.is_built_with_cuda())
#print(tf.test.is_gpu_available())

print(tf.config.list_physical_devices('GPU'))

