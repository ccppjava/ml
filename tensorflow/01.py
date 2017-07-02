import tensorflow as tf

node1 = tf.constant(3.0, dtype = tf.float32)
node2 = tf.constant(4.0)    # aslo tf.float32 implicitly
print(node1, node2)

sess = tf.Session()
print(sess.run(node1))
print(sess.run([node1, node2]))

node3 = tf.add(node1, node2)
print("node3: ", node3)
print("sess.run(node3): ", sess.run(node3))

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b # + provides a shortcut for tf.add(a, b)

# side note: http://blog.teamtreehouse.com/operator-overloading-python
# overload operator in python 
# def __add__(self, other):

print(sess.run(adder_node, {a: 3, b: 4.5}))
print(sess.run(adder_node, {a: [1, 3], b: [2, 4]}))

add_and_triple = adder_node * 3.
print(sess.run(add_and_triple, {a: 4, b: 4.4}))

# variable allow us to add trainable parameters to graph, they are constructed with a type and initial value
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b

init = tf.global_variables_initializer()
# it is important to realize init is a handle to the TensorFlow sub-graph that initializes all the global variables.
# Until we call sess.run, the variable are uninitialized
sess.run(init)

print(sess.run(linear_model, {x: [1,2,3,4]}))

# placeholder for desired value
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
# loss function measures how far apart the current model is from the provided data
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss, {x: [1,2,3,4], y: [0, -1, -2, -3]}))

# guess the perfect variable 
fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])
print(sess.run(loss, {x: [1,2,3,4], y: [0, -1, -2, -3]}))

# simplest optimizer is gradient descent, it modifies each variable according to 
# the magnitude of the deivative of loss with respect to that variable
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

sess.run(init) # reset values to incorrect defaults
for i in range(1000):
    sess.run(train, {x: [1,2,3,4], y: [0, -1, -2, -3]})

print(sess.run([W, b]))

print(sess.run(loss, {x: [1,2,3,4], y: [0, -1, -2, -3]}))


