import numpy as np


def sp_noise(image, prob):
    """给图片添加椒盐噪声

    参数
    image : numpy.ndarray
        代表图片的numpy数组

    返回值
    numpy.ndarray 添加噪声后的图片
    """
    output = np.zeros(image.shape)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = np.random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 1
            else:
                output[i][j] = image[i][j]
    return output.astype("float32")


def gauss_noise(image, mean=0, var=0.001):
    """给图片添加高斯噪声

    参数
    image : numpy.ndarray
        代表图片的numpy数组
    mean : float
        均值
    var : float
        方差

    返回值
    numpy.ndarray 添加噪声后的图片
    """
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    return out.astype("float32")
