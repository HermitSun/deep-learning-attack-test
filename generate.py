# images: 传入的测试样本数据(fashion-mnist数据集的shape的子集)，类型为numpy.ndarray。
# shape: images对应的shape，类型为tuple，例如image为1000张mnist的图片数据 (1000,28,28,1) 默认shape为(n, 28, 28 , 1), n为图片数据的数量
# return：返回基于images生成的对抗样本集合generate_images，二者shape一致，且一一对应（原始样本与对抗样本一一对应）
def generate(images, shape):
    return generate_images
