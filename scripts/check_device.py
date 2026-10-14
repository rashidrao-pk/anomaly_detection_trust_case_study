
import platform
import tensorflow as tf

print("Operating system:", platform.system())
print("Architecture:", platform.machine())
print("TensorFlow:", tf.__version__)

print("Available CPUs:", tf.config.list_physical_devices("CPU"))
print("Available GPUs:", tf.config.list_physical_devices("GPU"))

a = tf.random.normal((1024, 1024))
b = tf.random.normal((1024, 1024))
c = tf.matmul(a, b)

print("Matrix multiplication device:", c.device)
print("Result shape:", c.shape)
