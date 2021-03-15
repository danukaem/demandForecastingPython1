import numpy as np
import tensorflow as tf

x = np.array([(3, 3, 3), (3, 3, 3), (3, 3, 3)], dtype='int32')
cx = tf.constant(x)
# print(cx)

y = np.array([(1, 1, 1), (1, 1, 1), (1, 1, 1)], dtype='int32')
cy = tf.constant(y)
# print(cy)

mulXY = tf.matmul(cx, cy)
addXY = tf.add(cx, cy)

with tf.compat.v1.Session() as sess:
    m = sess.run(mulXY)
    n = sess.run(addXY)

print(m)
print(n)
