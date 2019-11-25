import tensorflow as tf
from random import randint


class Model(object):

    def __init__(self):
        self.input = tf.placeholder(shape=[None, 2], dtype=tf.float32, name='input')
        self.label = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='label')

        self.output = tf.layers.dense(self.input, units=1)

        optimizer = tf.train.AdamOptimizer(0.0001)
        self.global_step = tf.train.get_or_create_global_step()
        self.loss = tf.losses.mean_squared_error(self.label, self.output)
        self.train_op = optimizer.minimize(self.loss, self.global_step)
        self._make_saver()

    def train(self, sess, feed_val):
        _, loss = sess.run([self.train_op, self.loss], feed_dict={self.input: feed_val[0],
                                                       self.label: feed_val[1]})
        return loss

    def predict(self, sess, input_val):
        return sess.run(self.output, feed_dict={self.input: input_val})

    def _make_saver(self):
        self.saver = tf.train.Saver(var_list=tf.trainable_variables() + [self.global_step],
                                    max_to_keep=3)


    def restore_model(self, sess, ckpt_dir):
        latest_ckpt = tf.train.latest_checkpoint(ckpt_dir)
        self.saver.restore(sess, latest_ckpt)


def main():
    train_steps = 100000
    out_dir = 'test/'
    train = 0

    with tf.Graph().as_default() as global_graph:
        model = Model()
        sess_conf = tf.ConfigProto(intra_op_parallelism_threads=8, inter_op_parallelism_threads=8)
        sess_conf.gpu_options.allow_growth = True

        with tf.Session(graph=global_graph, config=sess_conf) as sess:
            try:
                sess.run(tf.global_variables_initializer())
                model.restore_model(sess, out_dir)
            except:
                sess.run(tf.global_variables_initializer())

            if train:
                for i in range(train_steps):
                    data = []
                    label = []
                    for _ in range(64):
                        a = randint(-99, 99)
                        b = randint(-99, 99)
                        data.append([a, b])
                        label.append([a * b])

                    loss = model.train(sess, [data, label])

                    if i % 1000 == 0:
                        with open(out_dir + 'loss.txt', 'a') as f:
                            f.write(str(loss) + '\n')

                        model.saver.save(sess, out_dir + 'ckpt', model.global_step)

            else:
                while 1:
                    a = input('a: ')
                    b = input('b: ')
                    print 'a * b: ', a * b

                    print 'predict: ', model.predict(sess, [[a, b]])


main()
