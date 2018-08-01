from keras.optimizers import Adam
from keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dropout, Dense
from keras.models import Model
from keras.callbacks import ModelCheckpoint
import os
import cv2
import numpy as np

chars = ['C', '津', '闽', '7', '沪', 'T', '浙', '辽', 'V', '2', '冀',
         '晋', 'P', '0', 'Q', 'Z', '赣', 'A', '皖', '桂', 'U', '粤',
         '豫', '琼', 'E', '6', '鄂', 'K', '黑', '甘', '1', 'J', '京',
         '蒙', '陕', '5', 'B', 'F', '澳', '8', '4', '鲁', 'R', '青',
         'H', '9', '藏', 'W', '云', '苏', 'G', 'O', '贵', 'N', '川',
         '3', 'S', 'M', '学', 'D', '港', '宁', 'X', '新', '渝', '吉',
         '湘', 'Y', 'L']

M_strIdx = dict(zip(chars, range(len(chars))))

adam = Adam(lr=0.001)

input_tensor = Input((72, 272, 3))
x = input_tensor
for i in range(3):
    x = Conv2D(32*2**i, (3, 3), activation='relu')(x)
    x = Conv2D(32*2**i, (3, 3), activation='relu')(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
x = Flatten()(x)
x = Dropout(0.25)(x)

n_class = len(chars)
x = [Dense(n_class, activation='softmax', name='c%d'%(i+1))(x) for i in range(7)]
model = Model(inputs=input_tensor, outputs=x)
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])
model.summary()

best_model = ModelCheckpoint('carno_01.h5',monitor='val_loss',verbose=0,save_best_only=True)

remove_list = []
s_root_str = '/Users/zhangmin/marco/carno/s/'
s_file_list = []
for root, dir, files in os.walk(s_root_str):
    s_file_list = files

for name in s_file_list:
    file_name = str(name)
    carno = file_name.split('-')[0]
    if len(carno) > 7:
        remove_list.append(file_name)
for name in remove_list:
    s_file_list.remove(name)

remove_list = []
ss_root_str = '/Users/zhangmin/marco/carno/ss/'
ss_file_list = []
for root, dir, files in os.walk(ss_root_str):
    ss_file_list = files

for name in ss_file_list:
    file_name = str(name)
    carno = file_name.split('-')[0]
    if len(carno)>7:
        remove_list.append(file_name)
for name in remove_list:
    ss_file_list.remove(name)

count_s_file_list = len(s_file_list)
count_ss_file_list = len(ss_file_list)

def gen(root_path, file_list, file_list_len, batch_size=20):
    while True:
        X = []
        y = []
        count = 0
        for i in np.arange(0,file_list_len):
            file_name = str(file_list[i])
            carno = file_name.split('-')[0]
            pic = cv2.imread(root_path+file_name)
            pic = cv2.resize(pic, (272, 72),  interpolation=cv2.INTER_CUBIC)
            X.append(pic)
            y.append(carno)
            count += 1
            if count % batch_size == 0:
                ytmp = np.array(list(map(lambda x: [M_strIdx[a] for a in list(x)], y)), dtype=np.uint8)
                y = np.zeros([ytmp.shape[1], batch_size, len(chars)])
                for batch in range(batch_size):
                    for idx, row_i in enumerate(ytmp[batch]):
                        y[idx, batch, row_i] = 1
                XX = X
                yy = y
                X = []
                y = []
                count = 0
                yield np.array(XX,dtype=np.uint8), [yyy for yyy in yy]

model.fit_generator(
    gen(root_path=s_root_str,file_list=s_file_list,
                        file_list_len=count_s_file_list, batch_size=20),
                    steps_per_epoch=count_s_file_list, epochs=5,
                    validation_data=gen(root_path=ss_root_str,
                                        file_list=ss_file_list,file_list_len=count_ss_file_list,
                                        batch_size=20),
                    validation_steps=count_ss_file_list,
                    callbacks=[best_model])

print('end,,,')