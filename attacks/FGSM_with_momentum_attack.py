from keras.models import load_model
import keras.backend as K
from process.ssim import get_ssim
import numpy as np
import time

# 加载模型
model = load_model("../models/cnn_model.hdf5")
test_data = np.load("../test_data/test_data.npy")
# 读取断点
# 读取断点
attack_data = list(np.load('../attack_data/attack_data.npy'))
has_processed = 6250 + len(attack_data)
ssim_sum = 0
success_count = 0
start = time.time()
for i in range(has_processed + 1, len(test_data)):
    # 需要进行预测，所以转换成(n,28,28,1)
    img = test_data[i]
    x = img.reshape(1, 28, 28, 1)
    # 模型一开始的预测结果
    preds = model.predict(x)
    initial_class = np.argmax(preds)

    sess = K.get_session()
    # 对抗样本
    x_adv = x
    # 噪声
    x_noise = np.zeros_like(x)
    # 循环次数和偏移量
    epochs = 10
    epsilon = 0.01
    # 动量
    S = 0
    momentum = 1.0

    for j in range(epochs):
        target = K.one_hot(initial_class, 10)
        loss = K.categorical_crossentropy(target, model.output)
        grads = K.gradients(loss, model.input)
        S = momentum * S + grads[0] / K.mean(abs(grads[0]), keepdims=True)
        delta = K.sign(S)
        x_noise = x_noise + delta
        x_adv = x_adv + epsilon * delta
        x_adv = sess.run(x_adv, feed_dict={model.input: x})
        preds = model.predict(x_adv)
        # 循环epochs次或者使模型预测出错时不再增加噪声
        if not np.argmax(preds) == initial_class:
            success_count += 1
            break
    # 计算SSIM
    ssim_val = get_ssim(x.reshape(28, 28, 1), x_adv.reshape(28, 28, 1))
    # 如果返回了原图，算法退化为iFGSM
    epochs = 5
    if ssim_val == 1.0:
        for k in range(epochs):
            target = K.one_hot(initial_class, 10)
            loss = K.categorical_crossentropy(target, model.output)
            grads = K.gradients(loss, model.input)
            delta = K.sign(grads[0])
            x_noise = x_noise + delta
            x_adv = x_adv + epsilon * delta
            x_adv = sess.run(x_adv, feed_dict={model.input: x})
            preds = model.predict(x_adv)
            # 循环epochs次或者使模型预测出错时不再增加噪声
            if not np.argmax(preds) == initial_class:
                break

    ssim_sum += ssim_val
    print(i, ssim_val)
    # 保存，可以用于断点续传
    np.save("../attack_data/attack_data.npy", np.array(attack_data))

end = time.time()
ssim_average = ssim_sum / len(test_data)
attack_acc = success_count / len(test_data)

print(end - start)
print(ssim_average)

attack_data = np.array(attack_data)
np.save("../attack_data/attack_data.npy", attack_data)
