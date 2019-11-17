import tensorflow as tf


def get_ssim(
        img_arr1,
        img_arr2,
        max_val=1.0,
        filter_size=11,
        filter_sigma=1.5,
        k1=0.01,
        k2=0.03
):
    """计算两张图片的结构相似度指数SSIM的封装函数

    参数
    img_arr1 : numpy.ndarray
        代表第一张图片的numpy数组
    img_arr2 : numpy.ndarray
        代表第二张图片的numpy数组
    max_val : float
        图片每个像素点的最大值
    其他参数参见tensorflow.image.ssim

    返回值
    float 两张图片的SSIM，默认为(-1, 1]
    """
    img1 = tf.convert_to_tensor(img_arr1)
    img2 = tf.convert_to_tensor(img_arr2)
    ssim = tf.image.ssim(
        img1,
        img2,
        max_val=max_val,
        filter_size=filter_size,
        filter_sigma=filter_sigma,
        k1=k1,
        k2=k2
    )
    with tf.Session() as sess:
        ssim = sess.run(ssim)
    return ssim
