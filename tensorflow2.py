#create a floating point tensor
tensor_float = tf.constant([10.1, 11.2, 12.3])
tensor_float, tensor_float.dtype

#create an integer tensor
tensor_int = tf.constant([21, 22, 23])
tensor_int, tensor_int.dtype

#type cast the tensor from int to float
tf.cast(tensor_int, dtype = tf.float32)

tf.cast(tensor_float, dtype = tf.int32)

tf.cast(tensor_float, dtype = tf.float16)

tf.cast(tensor_int, dtype= tf.int16)

tf.constant(["x", "y", "z"])

#addition : tf.add()

x = tf.constant([10., 20., 30.])
y = tf.constant([1., 2., 3.])
x, y

x + 10

x + y

add = tf.add(x,y)
add

x - 10

x - y

sub = tf.subtract(x, y)
sub

x * 10

x * y

mul = tf.multiply(x, y)
mul

x / 10

y / x

div = tf.divide(x, y)
div

square = tf.square(x)
square

x*x

power = tf.pow(x, 3)
power

x ** 3

sq_diff = tf.math.squared_difference(x,y)
sq_diff

tensor = tf.constant([30, 10, 70, 25, 15, 5])
tensor

min_ele_index = tf.argmin(tensor)
max_ele_index = tf.argmax(tensor)
min_ele_index, max_ele_index

tensor[min_ele_index].numpy(), tensor[max_ele_index].numpy()

min = tf.reduce_min(tensor)
min

max = tf.reduce_max(tensor)
max

matrix = tf.constant([[10,20,30], [40,50,60]])
matrix

sum = tf.reduce_sum(matrix)
sum

sum_0 = tf.reduce_sum(matrix, axis = 0)
sum_1 = tf.reduce_sum(matrix, axis = 1)
sum_0, sum_1
