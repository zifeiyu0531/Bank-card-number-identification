import tensorflow as tf
import forward
import os
import ImgHandle as IMG
import random

BATCH_SIZE = 20
REGULARIZER = 0.001
STEPS = 10000
MOVING_AVERAGE_DECAY = 0.01
MODEL_SAVE_PATH="./model/"
MODEL_NAME="train_model"
FILE_NAME="Classification.xlsx"

def backward(data, label):

    x = tf.placeholder(tf.float32, shape = (None, forward.INPUT_NODE))
    y_ = tf.placeholder(tf.float32, shape = (None, forward.OUTPUT_NODE))
    y = forward.forward(x, REGULARIZER)
    global_step = tf.Variable(0, trainable=False)	
	
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cem = tf.reduce_mean(ce)
    loss = cem + tf.add_n(tf.get_collection('losses'))

    train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss, global_step=global_step)

    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    ema_op = ema.apply(tf.trainable_variables())
    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name='train')

    saver = tf.train.Saver()

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        for i in range(STEPS):
            start = (i*BATCH_SIZE)%len(data)
            end = start+BATCH_SIZE
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: data[start:end], y_: label[start:end]})
            if i % 100 == 0:
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


def main():
    data, label = IMG.img_handle()
    for i in range(len(data)):
        x, y = random.randint(0, len(data)-1), random.randint(0, len(data)-1)
        temp_data = data[x]
        data[x] = data[y]
        data[y] = temp_data
        temp_label = label[x]
        label[x] = label[y]
        label[y] = temp_label
    print(len(data), len(label))
    backward(data, label)