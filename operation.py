import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow_probability as tfp

# SELayer 实现
class SELayer(tf.keras.layers.Layer):
    def __init__(self, num_filter):
        super(SELayer, self).__init__()
        self.global_pool = layers.GlobalAveragePooling2D()  # 全局平均池化
        self.conv_double = models.Sequential([
            layers.Conv2D(num_filter // 16, kernel_size=1, strides=1, padding='valid', use_bias=True),
            layers.LeakyReLU(),
            layers.Conv2D(num_filter, kernel_size=1, strides=1, padding='valid', use_bias=True),
            layers.Activation('sigmoid')  # 使用 Activation 层来应用 Sigmoid
        ])

    def call(self, x):
        mask = self.global_pool(x)
        mask = self.conv_double(tf.expand_dims(tf.expand_dims(mask, 1), 1))  # 扩展维度以适应广播
        return x * mask  # 按通道进行缩放

# 自定义的反射填充
def reflection_pad(x, pad_value=1):
    return tf.image.resize_with_crop_or_pad(x, x.shape[1] + 2 * pad_value, x.shape[2] + 2 * pad_value)

# ResBlock 实现
class ResBlock(tf.keras.Model):
    def __init__(self, num_filter):
        super(ResBlock, self).__init__()
        self.body = models.Sequential([
            layers.Lambda(reflection_pad),  # 使用 Lambda 实现反射填充
            layers.Conv2D(num_filter, kernel_size=3, padding='valid'),  # padding='valid' 等价于 padding=0
            layers.LeakyReLU(alpha=0.2),
            layers.Lambda(reflection_pad),  # 使用 Lambda 实现反射填充
            layers.Conv2D(num_filter, kernel_size=3, padding='valid'),  # padding='valid' 等价于 padding=0
            SELayer(num_filter)
        ])

    def call(self, x):
        res = self.body(x)
        return res + x  # 残差连接

class Up(tf.keras.layers.Layer):
    def __init__(self):
        super(Up, self).__init__()
        self.up = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')

    def call(self, x):
        x = self.up(x)
        return x


class ConvBlock(tf.keras.Model):
    def __init__(self, ch_in, ch_out):
        super(ConvBlock, self).__init__()
        # 反射填充可以通过 Lambda 层实现自定义操作
        self.conv = models.Sequential([
            layers.Lambda(lambda x: tf.image.resize_with_crop_or_pad(x, x.shape[1] + 2, x.shape[2] + 2)),  # ReflectionPad2d
            layers.Conv2D(ch_out, kernel_size=3, padding='valid', input_shape=(None, None, ch_in)),
            layers.LeakyReLU(),
            layers.Lambda(lambda x: tf.image.resize_with_crop_or_pad(x, x.shape[1] + 2, x.shape[2] + 2)),  # ReflectionPad2d
            layers.Conv2D(ch_out, kernel_size=3, padding='valid'),
            layers.LeakyReLU(),
        ])

    def call(self, x):
        return self.conv(x)

class Compute_z(tf.keras.layers.Layer):
    def __init__(self, latent_dim):
        super(Compute_z, self).__init__()
        self.latent_dim = latent_dim
        self.u_conv_layer = tf.keras.layers.Conv2D(2 * self.latent_dim, kernel_size=1, padding='valid')
        self.s_conv_layer = tf.keras.layers.Conv2D(2 * self.latent_dim, kernel_size=1, padding='valid')

    def call(self, x):
        # Compute u_mu and u_log_sigma
        u_encoding = tf.reduce_mean(x, axis=[2, 3], keepdims=True)
        u_mu_log_sigma = self.u_conv_layer(u_encoding)
        u_mu_log_sigma = tf.squeeze(u_mu_log_sigma, axis=[2, 3])  # Remove height and width dims
        u_mu = u_mu_log_sigma[:, :, :, :self.latent_dim]
        u_log_sigma = u_mu_log_sigma[:, :, :, self.latent_dim:]

        # Create the distribution for u
        u_dist = tfp.distributions.Independent(
            tfp.distributions.Normal(loc=u_mu, scale=tf.exp(u_log_sigma)),
            1
        )

        # Compute s_mu and s_log_sigma
        s_encoding = tf.math.reduce_std(x, axis=[2, 3], keepdims=True)
        s_mu_log_sigma = self.s_conv_layer(s_encoding)
        s_mu_log_sigma = tf.squeeze(s_mu_log_sigma, axis=[2, 3])  # Remove height and width dims
        s_mu = s_mu_log_sigma[:, :, :, :self.latent_dim]
        s_log_sigma = s_mu_log_sigma[:, :, :, self.latent_dim:]

        # Create the distribution for s
        s_dist = tfp.distributions.Independent(
            tfp.distributions.Normal(loc=s_mu, scale=tf.exp(s_log_sigma)),
            1
        )

        return u_dist, s_dist, u_mu, s_mu, tf.exp(u_log_sigma), tf.exp(s_log_sigma)

'''
class Encoder(tf.keras.layers.Layer):
    def __init__(self, ch):
        super(Encoder, self).__init__()
        self.conv1 = ConvBlock(ch_in=ch, ch_out=64)
        self.conv2 = ConvBlock(ch_in=64, ch_out=3)

    def call(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        return x2

class Decoder(tf.keras.Model):
    def __init__(self, device):
        super(Decoder, self).__init__()
        self.device = device
        # 假设Encoder类是另一个自定义模型，确保它的构造函数在TensorFlow中也能正确实现
        self.pr_encoder = Encoder(3)
        self.po_encoder = Encoder(6)

        # 残差块：这里使用了自定义的ResBlock，假设它是卷积层与激活函数的组合
        self.pr_conv = ResBlock(64)
        self.po_conv = ResBlock(64)

        # 上采样层
        self.pr_Up3 = Up()
        self.pr_UpConv3 = ConvBlock(ch_in=128, ch_out=64)
        self.pr_Up2 = Up()
        self.pr_UpConv2 = ConvBlock(ch_in=128, ch_out=64)
        self.pr_Up1 = Up()
        self.pr_UpConv1 = ConvBlock(ch_in=128, ch_out=64)

        self.po_Up3 = Up()
        self.po_UpConv3 = ConvBlock(ch_in=128, ch_out=64)
        self.po_Up2 = Up()
        self.po_UpConv2 = ConvBlock(ch_in=128, ch_out=64)

        # 输出层
        self.out_conv = tf.keras.Sequential([
            ResBlock(64),
            ResBlock(64),
            layers.Conv2D(3, kernel_size=1, padding='same')
        ])

        z = 20

        # z的计算层
        self.compute_z_pr = Compute_z(z)
        self.compute_z_po = Compute_z(z)

        # 1x1卷积层
        self.conv_u = layers.Conv2D(128, kernel_size=1, padding='same')
        self.conv_s = layers.Conv2D(128, kernel_size=1, padding='same')

        # 实例归一化
        self.insnorm = layers.BatchNormalization()  # 替代InstanceNorm
        self.sigmoid = layers.Activation('sigmoid')

    def call(self, config, inputs, training=True):
        Input, Target = inputs

        pr_x1, pr_x2, pr_x3, pr_x4 = self.pr_encoder(Input)

        if config.is_train:
            # 输入与目标拼接
            po_x1, po_x2, po_x3, po_x4 = self.po_encoder(tf.concat([Input, Target], axis=-1))

            # x4 -> x3
            pr_x4 = self.pr_conv(pr_x4)
            po_x4 = self.po_conv(po_x4)
            pr_d3 = self.pr_Up3(pr_x4)
            po_d3 = self.po_Up3(po_x4)

            # x3 -> x2
            pr_d3 = tf.concat([pr_x3, pr_d3], axis=-1)
            po_d3 = tf.concat([po_x3, po_d3], axis=-1)
            pr_d3 = self.pr_UpConv3(pr_d3)
            po_d3 = self.po_UpConv3(po_d3)

            # x2 -> x1
            pr_d2 = self.pr_Up2(pr_d3)
            po_d2 = self.po_Up2(po_d3)
            pr_d2 = tf.concat([pr_x2, pr_d2], axis=-1)
            po_d2 = tf.concat([po_x2, po_d2], axis=-1)
            pr_d2 = self.pr_UpConv2(pr_d2)
            po_d2 = self.po_UpConv2(po_d2)

            pr_d1 = self.pr_Up1(pr_d2)
            po_d1 = self.po_Up1(po_d2)

            # 拼接最终输出
            pr_d1 = tf.concat([pr_x1, pr_d1], axis=-1)
            po_d1 = tf.concat([po_x1, po_d1], axis=-1)

            # 计算分布
            pr_u_dist, pr_s_dist, _, _, _, _ = self.compute_z_pr(pr_d1)
            po_u_dist, po_s_dist, _, _, _, _ = self.compute_z_po(po_d1)

            po_latent_u = po_u_dist.sample()
            po_latent_s = po_s_dist.sample()

            po_latent_u = tf.expand_dims(po_latent_u, -1)
            po_latent_s = tf.expand_dims(po_latent_s, -1)

            po_u = self.conv_u(po_latent_u)
            po_s = self.conv_s(po_latent_s)

            pr_d1 = self.insnorm(pr_d1) * tf.abs(po_s) + po_u

            # 输出卷积
            pr_d1 = self.pr_UpConv1(pr_d1)
            out = self.out_conv(pr_d1)

            return out, pr_u_dist, pr_s_dist, po_u_dist, po_s_dist

        else:
            # 同训练时的过程，但省略了目标数据部分
            pr_x4 = self.pr_conv(pr_x4)
            pr_d3 = self.pr_Up3(pr_x4)
            pr_d3 = tf.concat([pr_x3, pr_d3], axis=-1)
            pr_d3 = self.pr_UpConv3(pr_d3)
            pr_d2 = self.pr_Up2(pr_d3)
            pr_d2 = tf.concat([pr_x2, pr_d2], axis=-1)
            pr_d2 = self.pr_UpConv2(pr_d2)
            pr_d1 = self.pr_Up1(pr_d2)
            pr_d1 = tf.concat([pr_x1, pr_d1], axis=-1)

            pr_u_dist, pr_s_dist, _, _, _, _ = self.compute_z_pr(pr_d1)
            pr_latent_u = pr_u_dist.sample()
            pr_latent_s = pr_s_dist.sample()

            pr_latent_u = tf.expand_dims(pr_latent_u, -1)
            pr_latent_s = tf.expand_dims(pr_latent_s, -1)

            pr_u = self.conv_u(pr_latent_u)
            pr_s = self.conv_s(pr_latent_s)

            pr_d1 = self.insnorm(pr_d1) * tf.abs(pr_s) + pr_u
            pr_d1 = self.pr_UpConv1(pr_d1)
            out = self.out_conv(pr_d1)

            return out
'''
