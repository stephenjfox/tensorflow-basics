# pulling the data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# We're using Softmax Regression for its multinominal output that's a probability of a given outcome
# Combined with one-hot encoding, we'll have a smarter, accelerated, informed training process.
import tensorflow as tf

# so given that y = softmax(Wx + b), let's define that in TensorFlow
# Note: None, here, means that we can have any dimensions
x = tf.placeholder(tf.float32, [None, 784]) # 28 x 28 of the MNIST images' pixels

# the variables will change with every pass through the network (epoch)
# this is telling TensorFlow to make these "reassignable" throughout our training process.
W = tf.Variable(tf.zeros([784, 10])) # why "zeros"? We'll decide what they can be later
b = tf.Variable(tf.zeros([10]))

#######
# "How is this tf.zeros() different from tf.placeholder()?"
#
# The tf.placeholder() tells the framework that said value exists, but will be filled in at "compile" time
# The tf.zeros() is more of a logical placeholder and, when in conjunction with tf.Variable(),
#   can be filled in as we please.
#   - Also, those values are going to be regularly updated. Variable in every sense of the word.
#######

y = tf.nn.softmax(tf.matmul(x, W) + b)

# Now for training: defining what is bad, so we can get as far away from it as possible.

y_prime = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_prime * tf.log(y), reduction_indices=[1]))

# In this case, we'll be defining a cost via "cross-entropy" which we will seek reduce to a minimum
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
# run 1000 epochs
for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_prime: batch_ys})


# Testing phase
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_prime, 1))
# that ^^^ is a list of booleans
# Think about the match up in a binary output format, and it starts to make more sense
# TODO: blog long winded explanation

# derived accuracy = how many of our labels were correct
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_prime: mnist.test.labels}))
