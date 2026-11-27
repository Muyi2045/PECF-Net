def model(self):
    with tf.compat.v1.variable_scope("main_branch") as scope3:
        # RMT
        depth_2down = max_pool_2x2(1 - self.images_depth)
        depth_4down = max_pool_2x2(depth_2down)
        depth_8down = max_pool_2x2(depth_4down)
        # first HSV encoder
        HSV = tf.image.rgb_to_hsv(self.images)
        conv2_1_HSV = tf.nn.relu(conv2d(HSV, 3, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_1_HSV"))
        conv2_cb1_1_HSV = tf.nn.relu(conv2d(conv2_1_HSV, 3, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb1_1_HSV"))
        conv2_cb1_2_HSV = tf.nn.relu(
            conv2d(conv2_cb1_1_HSV, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb1_2_HSV"))
        conv2_cb1_3_HSV = conv2d(conv2_cb1_2_HSV, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb1_3_HSV")

        first_add_HSV = tf.add(conv2_1_HSV, conv2_cb1_3_HSV)

        conv2_2_HSV = tf.nn.relu(conv2d(first_add_HSV, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_2_HSV"))
        conv2_cb1_4_HSV = tf.nn.relu(conv2d(conv2_2_HSV, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb1_4_HSV"))
        conv2_cb1_5_HSV = tf.nn.relu(
            conv2d(conv2_cb1_4_HSV, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb1_5_HSV"))
        conv2_cb1_6_HSV = conv2d(conv2_cb1_5_HSV, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb1_6_HSV")

        first_add1_HSV = tf.add(conv2_2_HSV, conv2_cb1_6_HSV)

        encoder1_down2_HSV = max_pool_2x2(first_add1_HSV)

        # first YUV encoder
        YUV = Conv_img.rgb_to_lab(self.images)
        conv2_1_YUV = tf.nn.relu(conv2d(YUV, 3, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_1_YUV"))
        conv2_cb1_1_YUV = tf.nn.relu(conv2d(conv2_1_YUV, 3, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb1_1_YUV"))
        conv2_cb1_2_YUV = tf.nn.relu(
            conv2d(conv2_cb1_1_YUV, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb1_2_YUV"))
        conv2_cb1_3_YUV = conv2d(conv2_cb1_2_YUV, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb1_3_YUV")

        first_add_YUV = tf.add(conv2_1_YUV, conv2_cb1_3_YUV)

        conv2_2_YUV = tf.nn.relu(conv2d(first_add_YUV, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_2_YUV"))
        conv2_cb1_4_YUV = tf.nn.relu(conv2d(conv2_2_YUV, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb1_4_YUV"))
        conv2_cb1_5_YUV = tf.nn.relu(
            conv2d(conv2_cb1_4_YUV, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb1_5_YUV"))
        conv2_cb1_6_YUV = conv2d(conv2_cb1_5_YUV, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb1_6_YUV")

        first_add1_YUV = tf.add(conv2_2_YUV, conv2_cb1_6_YUV)

        encoder1_down2_YUV = max_pool_2x2(first_add1_YUV)

        # first RGB encoder
        conv2_1 = tf.nn.relu(conv2d(self.images, 3, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_1"))
        conv2_cb1_1 = tf.nn.relu(
            conv2d(tf.concat(axis=3, values=[conv2_1, conv2_1_HSV, conv2_1_YUV]), 128, 128, k_h=3, k_w=3, d_h=1, d_w=1,
                   name="conv2_cb1_1"))
        conv2_cb1_2 = tf.nn.relu(
            conv2d(tf.concat(axis=3, values=[conv2_cb1_1, conv2_cb1_1_HSV, conv2_cb1_1_YUV]), 128, 128, k_h=3, k_w=3,
                   d_h=1, d_w=1, name="conv2_cb1_2"))
        conv2_cb1_3 = conv2d(tf.concat(axis=3, values=[conv2_cb1_2, conv2_cb1_3_HSV, conv2_cb1_3_YUV]), 128, 128, k_h=3,
                             k_w=3, d_h=1, d_w=1, name="conv2_cb1_3")
        first_add = tf.add(conv2_1, conv2_cb1_3)
        conv2_2 = tf.nn.relu(
            conv2d(tf.concat(axis=3, values=[first_add, first_add_HSV, first_add_YUV]), 128, 128, k_h=3, k_w=3, d_h=1,
                   d_w=1, name="conv2_2"))
        conv2_cb1_4 = tf.nn.relu(
            conv2d(tf.concat(axis=3, values=[conv2_2, conv2_2_HSV, conv2_2_YUV]), 128, 128, k_h=3, k_w=3, d_h=1, d_w=1,
                   name="conv2_cb1_4"))
        conv2_cb1_5 = tf.nn.relu(
            conv2d(tf.concat(axis=3, values=[conv2_cb1_4, conv2_cb1_4_HSV, conv2_cb1_4_YUV]), 128, 128, k_h=3, k_w=3,
                   d_h=1, d_w=1, name="conv2_cb1_5"))
        conv2_cb1_6 = conv2d(tf.concat(axis=3, values=[conv2_cb1_5, conv2_cb1_5_HSV, conv2_cb1_5_YUV]), 128, 128, k_h=3,
                             k_w=3, d_h=1, d_w=1, name="conv2_cb1_6")
        first_add1 = tf.add(conv2_2, conv2_cb1_6)

        encoder1_down2 = max_pool_2x2(first_add1)

        # second hsv encoder
        conv2_2_1_HSV = tf.nn.relu(conv2d(encoder1_down2_HSV, 3, 256, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_2_1_HSV"))
        conv2_cb2_1_HSV = tf.nn.relu(conv2d(conv2_2_1_HSV, 3, 256, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb2_1_HSV"))
        conv2_cb2_2_HSV = tf.nn.relu(
            conv2d(conv2_cb2_1_HSV, 256, 256, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb2_2_HSV"))
        conv2_cb2_3_HSV = conv2d(conv2_cb2_2_HSV, 256, 256, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb2_3_HSV")
        second_add_HSV = tf.add(conv2_2_1_HSV, conv2_cb2_3_HSV)
        conv2_2_2_HSV = tf.nn.relu(conv2d(second_add_HSV, 256, 256, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_2_2_HSV"))
        conv2_cb2_4_HSV = tf.nn.relu(
            conv2d(conv2_2_2_HSV, 256, 256, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb2_4_HSV"))
        conv2_cb2_5_HSV = tf.nn.relu(
            conv2d(conv2_cb2_4_HSV, 256, 256, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb2_5_HSV"))
        conv2_cb2_6_HSV = conv2d(conv2_cb2_5_HSV, 256, 256, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb2_6_HSV")
        second_add1_HSV = tf.add(conv2_2_2_HSV, conv2_cb2_6_HSV)

        encoder2_down2_HSV = max_pool_2x2(second_add1_HSV)

        # second YUV encoder
        conv2_2_1_YUV = tf.nn.relu(conv2d(encoder1_down2_YUV, 3, 256, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_2_1_YUV"))
        conv2_cb2_1_YUV = tf.nn.relu(conv2d(conv2_2_1_YUV, 3, 256, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb2_1_YUV"))
        conv2_cb2_2_YUV = tf.nn.relu(
            conv2d(conv2_cb2_1_YUV, 256, 256, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb2_2_YUV"))
        conv2_cb2_3_YUV = conv2d(conv2_cb2_2_YUV, 256, 256, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb2_3_YUV")
        second_add_YUV = tf.add(conv2_2_1_YUV, conv2_cb2_3_YUV)
        conv2_2_2_YUV = tf.nn.relu(conv2d(second_add_YUV, 256, 256, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_2_2_YUV"))
        conv2_cb2_4_YUV = tf.nn.relu(
            conv2d(conv2_2_2_YUV, 256, 256, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb2_4_YUV"))
        conv2_cb2_5_YUV = tf.nn.relu(
            conv2d(conv2_cb2_4_YUV, 256, 256, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb2_5_YUV"))
        conv2_cb2_6_YUV = conv2d(conv2_cb2_5_YUV, 256, 256, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb2_6_YUV")
        second_add1_YUV = tf.add(conv2_2_2_YUV, conv2_cb2_6_YUV)

        encoder2_down2_YUV = max_pool_2x2(second_add1_YUV)

        # second RGB encoder
        conv2_2_1 = tf.nn.relu(conv2d(encoder1_down2, 3, 256, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_2_1"))
        conv2_cb2_1 = tf.nn.relu(
            conv2d(tf.concat(axis=3, values=[conv2_2_1, conv2_2_1_HSV, conv2_2_1_YUV]), 3, 256, k_h=3, k_w=3, d_h=1,
                   d_w=1, name="conv2_cb2_1"))
        conv2_cb2_2 = tf.nn.relu(
            conv2d(tf.concat(axis=3, values=[conv2_cb2_1, conv2_cb2_1_HSV, conv2_cb2_1_YUV]), 256, 256, k_h=3, k_w=3,
                   d_h=1, d_w=1, name="conv2_cb2_2"))
        conv2_cb2_3 = conv2d(tf.concat(axis=3, values=[conv2_cb2_2, conv2_cb2_3_HSV, conv2_cb2_3_YUV]), 256, 256, k_h=3,
                             k_w=3, d_h=1, d_w=1, name="conv2_cb2_3")
        second_add = tf.add(conv2_2_1, conv2_cb2_3)
        conv2_2_2 = tf.nn.relu(
            conv2d(tf.concat(axis=3, values=[second_add, second_add_HSV, second_add_YUV]), 256, 256, k_h=3, k_w=3,
                   d_h=1, d_w=1, name="conv2_2_2"))
        conv2_cb2_4 = tf.nn.relu(
            conv2d(tf.concat(axis=3, values=[conv2_2_2, conv2_2_2_HSV, conv2_2_2_YUV]), 256, 256, k_h=3, k_w=3, d_h=1,
                   d_w=1, name="conv2_cb2_4"))
        conv2_cb2_5 = tf.nn.relu(
            conv2d(tf.concat(axis=3, values=[conv2_cb2_4, conv2_cb2_4_HSV, conv2_cb2_4_YUV]), 256, 256, k_h=3, k_w=3,
                   d_h=1, d_w=1, name="conv2_cb2_5"))
        conv2_cb2_6 = conv2d(tf.concat(axis=3, values=[conv2_cb2_5, conv2_cb2_5_HSV, conv2_cb2_5_YUV]), 256, 256, k_h=3,
                             k_w=3, d_h=1, d_w=1, name="conv2_cb2_6")
        second_add1 = tf.add(conv2_2_2, conv2_cb2_6)

        encoder2_down2 = max_pool_2x2(second_add1)

        # third hsv encoder
        conv2_3_1_HSV = tf.nn.relu(conv2d(encoder2_down2_HSV, 3, 512, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_3_1_HSV"))
        conv2_cb3_1_HSV = tf.nn.relu(conv2d(conv2_3_1_HSV, 3, 512, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb3_1_HSV"))
        conv2_cb3_2_HSV = tf.nn.relu(
            conv2d(conv2_cb3_1_HSV, 512, 512, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb3_2_HSV"))
        conv2_cb3_3_HSV = conv2d(conv2_cb3_2_HSV, 512, 512, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb3_3_HSV")
        third_add_HSV = tf.add(conv2_3_1_HSV, conv2_cb3_3_HSV)
        conv2_3_2_HSV = tf.nn.relu(conv2d(third_add_HSV, 512, 512, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_3_2_HSV"))
        conv2_cb3_4_HSV = tf.nn.relu(
            conv2d(conv2_3_2_HSV, 512, 512, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb3_4_HSV"))
        conv2_cb3_5_HSV = tf.nn.relu(
            conv2d(conv2_cb3_4_HSV, 512, 512, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb3_5_HSV"))
        conv2_cb3_6_HSV = conv2d(conv2_cb3_5_HSV, 512, 512, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb3_6_HSV")

        third_add1_HSV = tf.add(conv2_3_2_HSV, conv2_cb3_6_HSV)

        # third YUV encoder
        conv2_3_1_YUV = tf.nn.relu(conv2d(encoder2_down2_YUV, 3, 512, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_3_1_YUV"))
        conv2_cb3_1_YUV = tf.nn.relu(conv2d(conv2_3_1_YUV, 3, 512, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb3_1_YUV"))
        conv2_cb3_2_YUV = tf.nn.relu(
            conv2d(conv2_cb3_1_YUV, 512, 512, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb3_2_YUV"))
        conv2_cb3_3_YUV = conv2d(conv2_cb3_2_YUV, 512, 512, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb3_3_YUV")
        third_add_YUV = tf.add(conv2_3_1_YUV, conv2_cb3_3_YUV)
        conv2_3_2_YUV = tf.nn.relu(conv2d(third_add_YUV, 512, 512, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_3_2_YUV"))
        conv2_cb3_4_YUV = tf.nn.relu(
            conv2d(conv2_3_2_YUV, 512, 512, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb3_4_YUV"))
        conv2_cb3_5_YUV = tf.nn.relu(
            conv2d(conv2_cb3_4_YUV, 512, 512, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb3_5_YUV"))
        conv2_cb3_6_YUV = conv2d(conv2_cb3_5_YUV, 512, 512, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb3_6_YUV")

        third_add1_YUV = tf.add(conv2_3_2_YUV, conv2_cb3_6_YUV)

        # third RGB encoder
        conv2_3_1 = tf.nn.relu(conv2d(encoder2_down2, 3, 512, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_3_1"))
        conv2_cb3_1 = tf.nn.relu(
            conv2d(tf.concat(axis=3, values=[conv2_3_1, conv2_3_1_HSV, conv2_3_1_YUV]), 3, 512, k_h=3, k_w=3, d_h=1,
                   d_w=1, name="conv2_cb3_1"))
        conv2_cb3_2 = tf.nn.relu(
            conv2d(tf.concat(axis=3, values=[conv2_cb3_1, conv2_cb3_1_HSV, conv2_cb3_1_YUV]), 512, 512, k_h=3, k_w=3,
                   d_h=1, d_w=1, name="conv2_cb3_2"))
        conv2_cb3_3 = conv2d(tf.concat(axis=3, values=[conv2_cb3_2, conv2_cb3_2_HSV, conv2_cb3_2_YUV]), 512, 512, k_h=3,
                             k_w=3, d_h=1, d_w=1, name="conv2_cb3_3")
        third_add = tf.add(conv2_3_1, conv2_cb3_3)
        conv2_3_2 = tf.nn.relu(
            conv2d(tf.concat(axis=3, values=[third_add, third_add_HSV, third_add_YUV]), 512, 512, k_h=3, k_w=3, d_h=1,
                   d_w=1, name="conv2_3_2"))
        conv2_cb3_4 = tf.nn.relu(
            conv2d(tf.concat(axis=3, values=[conv2_3_2, conv2_3_2_HSV, conv2_3_2_YUV]), 512, 512, k_h=3, k_w=3, d_h=1,
                   d_w=1, name="conv2_cb3_4"))
        conv2_cb3_5 = tf.nn.relu(
            conv2d(tf.concat(axis=3, values=[conv2_cb3_4, conv2_cb3_4_HSV, conv2_cb3_4_YUV]), 512, 512, k_h=3, k_w=3,
                   d_h=1, d_w=1, name="conv2_cb3_5"))
        conv2_cb3_6 = conv2d(tf.concat(axis=3, values=[conv2_cb3_5, conv2_cb3_5_HSV, conv2_cb3_5_YUV]), 512, 512, k_h=3,
                             k_w=3, d_h=1, d_w=1, name="conv2_cb3_6")

        third_add1 = tf.add(conv2_3_2, conv2_cb3_6)

        # concate
        third_con = tf.concat(axis=3, values=[third_add1, third_add1_HSV, third_add1_YUV])
        channle_weight_third_con_temp = Squeeze_excitation_layer(third_con, out_dim=1536, ratio=16,
                                                                 layer_name='channle_weight_third_con_temp')
        third_con_ff = tf.nn.relu(
            conv2d(channle_weight_third_con_temp, 1536, 512, k_h=3, k_w=3, d_h=1, d_w=1, name="third_con_ff"))
        second_con = tf.concat(axis=3, values=[second_add1, second_add1_HSV, second_add1_YUV])
        channle_weight_second_con_temp = Squeeze_excitation_layer(second_con, out_dim=768, ratio=16,
                                                                  layer_name='channle_weight_second_con_temp')
        second_con_ff = tf.nn.relu(
            conv2d(channle_weight_second_con_temp, 768, 256, k_h=3, k_w=3, d_h=1, d_w=1, name="second_con_ff"))
        first_con = tf.concat(axis=3, values=[first_add1, first_add1_HSV, first_add1_YUV])
        channle_weight_first_con_temp = Squeeze_excitation_layer(first_con, out_dim=384, ratio=16,
                                                                 layer_name='channle_weight_first_con_temp')
        first_con_ff = tf.nn.relu(
            conv2d(channle_weight_first_con_temp, 384, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="first_con_ff"))

        # first decoder
        decoder_input = tf.add(third_con_ff, tf.multiply(third_con_ff, depth_4down))
        conv2_1_1_dc = tf.nn.relu(conv2d(decoder_input, 512, 512, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_1_1_dc"))
        conv2_dc1_1 = tf.nn.relu(conv2d(conv2_1_1_dc, 512, 512, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_dc1_1"))
        conv2_dc1_2 = tf.nn.relu(conv2d(conv2_dc1_1, 512, 512, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_dc1_2"))
        conv2_dc1_3 = conv2d(conv2_dc1_2, 512, 512, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_dc1_3")
        first_dadd = tf.add(conv2_1_1_dc, conv2_dc1_3)
        conv2_1_2_dc = tf.nn.relu(conv2d(first_dadd, 512, 512, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_1_2_dc"))
        conv2_dc1_4 = tf.nn.relu(conv2d(conv2_1_2_dc, 512, 512, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_dc1_4"))
        conv2_dc1_5 = tf.nn.relu(conv2d(conv2_dc1_4, 512, 512, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_dc1_5"))
        conv2_dc1_6 = conv2d(conv2_dc1_5, 512, 512, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_dc1_6")
        first_dadd1 = tf.add(conv2_1_2_dc, conv2_dc1_6)
        decoder1_down2 = tf.image.resize(first_dadd1, [second_con_ff.get_shape().as_list()[1],
                                                       second_con_ff.get_shape().as_list()[2]],
                                         method=tf.image.ResizeMethod.BILINEAR)

        # second decoder
        decoder_input1 = tf.add(second_con_ff, tf.multiply(second_con_ff, depth_2down))
        concate_1 = tf.concat(axis=3, values=[decoder1_down2, decoder_input1])
        conv2_1_3_dc = tf.nn.relu(conv2d(concate_1, 512, 256, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_1_3_dc"))
        conv2_dc2_1 = tf.nn.relu(conv2d(conv2_1_3_dc, 256, 256, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_dc2_1"))
        conv2_dc2_2 = tf.nn.relu(conv2d(conv2_dc2_1, 256, 256, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_dc2_2"))
        conv2_dc2_3 = conv2d(conv2_dc2_2, 256, 256, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_dc2_3")
        second_dadd = tf.add(conv2_1_3_dc, conv2_dc2_3)
        conv2_1_4_dc = tf.nn.relu(conv2d(second_dadd, 256, 256, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_1_4_dc"))
        conv2_dc2_4 = tf.nn.relu(conv2d(conv2_1_4_dc, 256, 256, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_dc2_4"))
        conv2_dc2_5 = tf.nn.relu(conv2d(conv2_dc2_4, 256, 256, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_dc2_5"))
        conv2_dc2_6 = conv2d(conv2_dc2_5, 256, 256, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_dc2_6")
        second_dadd1 = tf.add(conv2_1_4_dc, conv2_dc2_6)
        decoder2_down2 = tf.image.resize(second_dadd1, [first_con_ff.get_shape().as_list()[1],
                                                        first_con_ff.get_shape().as_list()[2]],
                                         method=tf.image.ResizeMethod.BILINEAR)

        # third decoder
        decoder_input2 = tf.add(first_con_ff, tf.multiply(first_con_ff, (1 - self.images_depth)))
        concate_2 = tf.concat(axis=3, values=[decoder2_down2, decoder_input2])
        conv2_1_5_dc = tf.nn.relu(conv2d(concate_2, 256, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_1_5_dc"))
        conv2_dc3_1 = tf.nn.relu(conv2d(conv2_1_5_dc, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_dc3_1"))
        conv2_dc3_2 = tf.nn.relu(conv2d(conv2_dc3_1, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_dc3_2"))
        conv2_dc3_3 = conv2d(conv2_dc3_2, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_dc3_3")
        third_dadd = tf.add(conv2_1_5_dc, conv2_dc3_3)
        conv2_1_6_dc = tf.nn.relu(conv2d(third_dadd, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_1_6_dc"))
        conv2_dc3_4 = tf.nn.relu(conv2d(conv2_1_6_dc, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_dc3_4"))
        conv2_dc3_5 = tf.nn.relu(conv2d(conv2_dc3_4, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_dc3_5"))
        conv2_dc3_6 = conv2d(conv2_dc3_5, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_dc3_6")
        third_dadd1 = tf.add(conv2_1_6_dc, conv2_dc3_6)
        conv2_refine = conv2d(third_dadd1, 128, 3, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_refine")

        output = conv2_refine

    return output