#create a floating point tensor
tensor_float = tf.constant([10.1, 11.2, 12.3])
tensor_float, tensor_float.dtype

#create an integer tensor
tensor_int = tf.constant([21, 22, 23])
tensor_int, tensor_int.dtype

#type cast the tensor from int to float
tf.cast(tensor_int, dtype = tf.float32)
