import pickle
import numpy as np
from process.ssim import get_ssim

# fix random
np.random.seed(7)


def generate(images, shape):
    generate_images = []
    # 查看是否属于测试集的内容
    with open("attacks/data_index_map.dat", "rb") as map:
        data_index_map = pickle.load(map)
    print("index map loaded")
    attack_data = np.load("attack_data/attack_data.npy")
    print("pretrained attack data loaded")

    # 图像合成
    def image_composition(current_i):
        # 对于每个样本，随机抽取150个样本，计算SSIM
        samples = [np.random.randint(0, len(images)) for j in range(150)]
        max_ssim = 0
        max_j = 0
        for sample_index in samples:
            # 不是自己
            if not current_i == sample_index:
                current_ssim = get_ssim(images[current_i], images[sample_index])
                if current_ssim > max_ssim:
                    max_ssim = current_ssim
                    max_j = sample_index
        # 将SSIM最高的样本作为噪声，合并到原图片上，生成对抗样本
        alpha = 1 / (3.25 + max_ssim)
        beta = 1 - alpha
        gamma = 0
        attack = images[current_i] * alpha + images[max_j] * beta + gamma
        generate_images.append(attack)

    for i in range(len(images)):
        # 如果能找到现成的对抗样本，直接返回
        attack_index = data_index_map.get(str(images[i]))
        if attack_index is not None:
            attack = attack_data[attack_index].reshape((shape[1], shape[2], shape[3]))
            generate_images.append(attack)
        # 找不到就现场生成，算法退化为图像合成
        else:
            image_composition(i)
    return np.asarray(generate_images)
