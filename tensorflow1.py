import tensorflow as tf
print("Tensorflow version is: ", tf.__version__)

#create a scaler tensor - 0D tensor
scaler = tf.constant(11)
scaler

#check the no. of dimensions of scaler tensor
scaler.ndim

#create a vector tensor - 1D tensor
vector = tf.constant([1,2,3])
vector

#check the no. of dimensions of a vector tensor
vector.ndim

#create a matrix tensor - 2D tensor
matrix = tf.constant([[1,2,3], [4,5,6]])
matrix

#check no. of dimensions of a matrix tensor
matrix.ndim

#create a matrix tensor with float dtype
matrix_float = tf.constant([[1.,2.,3.], [4.,5.,6.]], dtype=tf.float16)
matrix_float

matrix_float.ndim

matrix3 = tf.constant([[[1,2,3], [4,5,6]], [[7,8,9], [10,11,12]]])
matrix3

matrix3.ndim

matrix4 = tf.constant([[[1,2,3], [4,5,6]], [[7,8,9], [10,11,12]], [[13,14,15], [16,17,18]]])
matrix4

tensor = tf.constant([[[1,2,3], [4,5,6]],
                     [[7,8,9], [10,11,12]],
                     [[13,14,15], [16,17,18]]])
tensor

tensor.ndim

#VARIABLES

var_vector = tf.Variable([1,2,3])
var_vector

const_vector = tf.constant([2,3,4])
const_vector

var_vector[0].assign(10)
var_vector

#const_vector[1].assign(5)
const_vector

var_matrix = tf.Variable([[1,2,3], [4,5,6]])
var_matrix

#convert python list,scalers and numpy arrays to tensors

import numpy as np

#!pip install numpy

scaler = 10
vector = [1,2,3,4]
array = np.array([10,12,15,20,16])
matrix = np.array([[10,20,30], [40,50,60]])

scaler = tf.convert_to_tensor(scaler)
scaler

vector = tf.convert_to_tensor(vector)
vector

vector = tf.convert_to_tensor(array)
vector

matrix = tf.convert_to_tensor(matrix)
matrix

#CREATE A RANDOM TENSOR FROM NORMAL DISTRIBUTION
tensor = tf.random.normal(shape = (3,2), mean = 10.0, stddev = 2.0, dtype = tf.float16)
tensor

#using uniform distribution
tensor = tf.random.uniform(shape = (3,2), minval = 0, maxval = 10, dtype = tf.float16)
tensor

#getting basic info from a tensor(data type, shape, rank/number of dimensions, total no. of elements)

tensor = tf.constant([[[1,2,3], [4,5,6]],
                      [[7,8,9], [10,11,12]],
                      [[13,14,15], [16,17,28]]])
tensor

print("Data type of tensor: ", tensor.dtype)

print("Shape of a tensor: ", tensor.shape)

print("Rank/No. of dimensions of a tensor: ", tensor.ndim)

print("Total no. of elements in a tensor: ", tf.size(tensor))

print("Total no. of elements in a tensor: ", tf.size(tensor).numpy())

#indexing and slicing with tensors

vector = tf.constant([1,2,3,4,5,6,7,8,9,10])
vector

vector[0]

#all elements
vector[:]

vector[1:5]

vector[::-1]

#fetching multiple elements from a tensor for multiple indices- tf.gather(data, indices)
# - indices can be repeated.
# - by default, it gathers rows when we're working with a 2D tensor
# - tf.gather(x, indices, axis = 0) => gather along rows
# - tf.gather(x, indices, axis = 1) => gather along columns for 2D data

indices = tf.constant([2,5])
tf.gather(vector,indices)

matrix = tf.constant([[1,2,3], [4,5,6], [7,8,9]])
matrix[0, :] #all elements from row 0

matrix[:, :2] #fetch first 2 elements from all rows

#create a tensor with a sequence of numbers using tf.range()
#syntax-:  tf.range(start, limit, delta = 1, dtype = None, name = 'range)

#create a sequence of numbers -> vector tensor using tf.range()
vec_tensor = tf.range(start = 1, limit = 10)
vec_tensor

#create a vector tensor using tf.range() without giving 'start' input
limit = 10
tf.range(limit)

#vector tensor with even and odd values
even_tensor = tf.range(start = 2, limit = 21, delta = 2)
odd_tensor = tf.range(start = 1, limit = 20, delta = 2)
even_tensor, odd_tensor
