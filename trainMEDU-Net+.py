import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as plb
import tensorflow as tf
from keras.layers import Input
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.utils import multi_gpu_model
from sklearn.model_selection import KFold

from util.getdata3 import GetData
from model.MEDU-Net import googlenet1
#from model.multiunet import MultiResUnet
from util.evaluation import the_iou, the_dice_coef, standard_mean_iou, dice_coef
import random
from util.logger import Logger

# 配置GPU运行方式
os.environ["CUDA_VISIBLE_DEVICES"] = '1'  # 指定可用GPU
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.99  # 程序最多只能占用指定gpu80%的显存
config.gpu_options.allow_growth = True  # 程序按需申请内存
sess = tf.Session(config=config)

# 参数设置
path_train = './input/'
input_img = Input((512, 512, 1))
image_size = (512, 512, 1)

model = googlenet1(input_img)

# 显示模型各层的摘要
model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', standard_mean_iou, dice_coef])


# 分割数据集
train_set_input, train_set_gt, test_set_input, test_set_gt = GetData(image_size).get_data(
    path_train)
print("数据集分割完毕!")

print("开始训练！！！")

kfold = KFold(n_splits=5)
miou = []
mdice = []
all_scores = []
i = 1
for train, viald in kfold.split(train_set_input, train_set_gt):
    # 定义回调
    print("这是第", i, '折')
    callback_path = 'degoog_' + str(i) + '.h5'
    i = i + 1
    callbacks = [
        EarlyStopping(patience=10, verbose=1),
        ReduceLROnPlateau(factor=0.5, patience=3, min_lr=0.0000001, verbose=1),
        ModelCheckpoint(filepath=callback_path, verbose=1, save_best_only=True,
                                save_weights_only=True)
    ]

    # 这个 `fit` 调用将分布在 2 个 GPU 上。
    # 由于 batch size 是 4, 每个 GPU 将处理 2 个样本。
    results = model.fit(train_set_input[train], train_set_gt[train], batch_size=1, epochs=500,
                       callbacks=callbacks,
                       validation_data=(train_set_input[viald], train_set_gt[viald]))

    scores = model.evaluate(test_set_input, test_set_gt, batch_size=1, verbose=1)
    all_scores.append(scores)
    miou.append(scores[2])
    mdice.append(scores[3])

sys.stdout = Logger('goognet1.txt')
print("训练完成！！！")
print('iou=', miou)
print('miou=', np.mean(miou), np.std(miou))
print('mdice', mdice)
print('mdice=', np.mean(mdice), np.std(mdice))
print('all scores=', all_scores)

plt.figure(figsize=(8, 8))
plt.title("Learning curve")
plt.plot(results.history["loss"], label="loss")
plt.plot(results.history["val_loss"], label="val_loss")
plt.plot(np.argmin(results.history["val_loss"]), np.min(results.history["val_loss"]), marker="x", color="r",
         label="best model")
plt.xlabel("Epochs")
plt.ylabel("log_loss")
plt.legend();