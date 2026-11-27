def model(self):
    with tf.compat.v1.variable_scope("main_branch") as scope3:
        conb0 = tf.concat(axis=3, values=[self.images, self.images_wb, self.images_ce, self.images_gc])
        conv_wb1 = tf.nn.relu(conv2d(conb0, 16, 128, k_h=7, k_w=7, d_h=1, d_w=1, name="conv2wb_1"))
        conv_wb2 = tf.nn.relu(conv2d(conv_wb1, 128, 128, k_h=5, k_w=5, d_h=1, d_w=1, name="conv2wb_2"))
        conv_wb3 = tf.nn.relu(conv2d(conv_wb2, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2wb_3"))
        conv_wb4 = tf.nn.relu(conv2d(conv_wb3, 128, 64, k_h=1, k_w=1, d_h=1, d_w=1, name="conv2wb_4"))
        conv_wb5 = tf.nn.relu(conv2d(conv_wb4, 64, 64, k_h=7, k_w=7, d_h=1, d_w=1, name="conv2wb_5"))
        conv_wb6 = tf.nn.relu(conv2d(conv_wb5, 64, 64, k_h=5, k_w=5, d_h=1, d_w=1, name="conv2wb_6"))
        conv_wb7 = tf.nn.relu(conv2d(conv_wb6, 64, 64, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2wb_7"))

        conv_wb77 = tf.nn.sigmoid(conv2d(conv_wb7, 64, 3, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2wb_77"))

        conb00 = tf.concat(axis=3, values=[self.images, self.images_wb])
        conv_wb9 = tf.nn.relu(conv2d(conb00, 3, 32, k_h=7, k_w=7, d_h=1, d_w=1, name="conv2wb_9"))
        conv_wb10 = tf.nn.relu(conv2d(conv_wb9, 32, 32, k_h=5, k_w=5, d_h=1, d_w=1, name="conv2wb_10"))
        wb1 = tf.nn.relu(conv2d(conv_wb10, 32, 3, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2wb_11"))

        conb11 = tf.concat(axis=3, values=[self.images, self.images_ce])
        conv_wb99 = tf.nn.relu(conv2d(conb11, 3, 32, k_h=7, k_w=7, d_h=1, d_w=1, name="conv2wb_99"))
        conv_wb100 = tf.nn.relu(conv2d(conv_wb99, 32, 32, k_h=5, k_w=5, d_h=1, d_w=1, name="conv2wb_100"))
        ce1 = tf.nn.relu(conv2d(conv_wb100, 32, 3, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2wb_111"))

        conb111 = tf.concat(axis=3, values=[self.images, self.images_gc])
        conv_wb999 = tf.nn.relu(conv2d(conb111, 3, 32, k_h=7, k_w=7, d_h=1, d_w=1, name="conv2wb_999"))
        conv_wb1000 = tf.nn.relu(conv2d(conv_wb999, 32, 32, k_h=5, k_w=5, d_h=1, d_w=1, name="conv2wb_1000"))
        gc1 = tf.nn.relu(conv2d(conv_wb1000, 32, 3, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2wb_1111"))

        weight_wb, weight_ce, weight_gc = tf.split(conv_wb77, 3, 3)
        output = tf.add(tf.add(tf.multiply(wb1, weight_wb), tf.multiply(ce1, weight_ce)),
                        tf.multiply(gc1, weight_gc))

    return output