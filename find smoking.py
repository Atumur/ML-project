import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import MobileNetV2
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras import regularizers
from keras.optimizers import Adamax
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score


def dataset_processing(df: pd.DataFrame, path: os.path):
    new_record = pd.DataFrame()
    for img in os.listdir(path):
        if img[0] == 's':
            new_record = pd.DataFrame({"path": [path + '/' + img], "label": ['smoking'], "class_id": [1]})
        elif img[0] == 'n':
            new_record = pd.DataFrame({"path": [path + '/' + img], "label": ['notsmoking'], "class_id": [0]})
        df = pd.concat([df, new_record], ignore_index=True)

    # Поменяем тип признака class_id с float на int
    df[["class_id"]] = df[["class_id"]].astype(int)

    return df


def check_model_predictions(y_true, y_pred, n):
    plt.figure(figsize=(15, 15))  # Понижаем высоту фигуры
    rand_nums = np.random.randint(0, len(y_true), n)

    for i in range(n):
        if y_true['class_id'][rand_nums[i]] == y_pred[rand_nums[i]]:
            color = 'green'
        else:
            color = 'red'

        full_path = y_true['path'][rand_nums[i]]
        plt.subplot(3, 4, i+1)  # 3 ряда и 5 столбцов для графиков
        plt.imshow(plt.imread(full_path))
        true_label = y_true['label'][rand_nums[i]]
        predicted_label = 'notsmoking' if  y_pred[rand_nums[i]] == 0 else 'smoking'
        plt.axis('off')
        plt.title(f'Предполагалось: {true_label}\nПолучилось: {predicted_label}', color=color)

    plt.tight_layout()
    plt.show()


train_df = pd.DataFrame({"path": [], "label": [], "class_id": []})
train_df = dataset_processing(train_df, '/opt/anaconda3/envs/env/archive-2/Training/Training')
print(train_df)
train_df.info()


plt.figure(figsize=(8, 6))
ax = sns.countplot(x=train_df["label"], hue=train_df["label"], palette="Spectral")
plt.xlabel("Labels", fontsize=12)
plt.xticks([0, 1], ["Некурящие", "Курящие"])
plt.ylabel("Count", fontsize=12)
plt.title("Подсчет курящих и некурящих людей в train выборке", fontsize=14)


images = np.random.randint(0, len(train_df), size=15)
fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(15, 10))
# axes.flatten преобразует двумерный массив в одномерный
axes = axes.flatten()
for i, ax in enumerate(axes):
    full_path = train_df.loc[images[i]]['path']
    ax.imshow(plt.imread(full_path))
    ax.set_title(train_df.loc[images[i]]['label'])
    ax.set_axis_off()
plt.show()

img_size = (224, 224)
batch_size = 16
# ImageDataGenerator() - подготавливает фотографию для машинного обучения
tr_gen = ImageDataGenerator()
# shuffle - этот параметр указывает, что изображения должны быть перемешаны перед каждой итерацией
train_gen = tr_gen.flow_from_dataframe(train_df, x_col='path', y_col='label', target_size=img_size,
                                       class_mode='categorical', color_mode='rgb', shuffle=True, batch_size=batch_size)

gen_dict = train_gen.class_indices
classes = list(gen_dict.keys())
images, labels = next(train_gen)

num_class = len(classes)
base_model = MobileNetV2(input_shape=(img_size[0], img_size[1], 3), include_top=False, weights='imagenet', pooling='max')

model = Sequential([
    base_model,
    BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001),
    Dense(256, kernel_regularizer=regularizers.l2(0.016), 
          activity_regularizer=regularizers.l1(0.006),
          bias_regularizer=regularizers.l1(0.006),
          activation='sigmoid'),
    Dropout(rate=0.4, seed=75),
    Dense(num_class, activation='softmax')
])

model.compile(Adamax(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

print("Model Summary:")
model.summary()


val_df = pd.DataFrame({"path": [], "label": [], "class_id": []})
val_df = dataset_processing(train_df, '/opt/anaconda3/envs/env/archive-2/Validation')

test_df = pd.DataFrame({"path": [], "label": [], "class_id": []})
test_df = dataset_processing(train_df, '/opt/anaconda3/envs/env/archive-2/Testing')

ts_gen = ImageDataGenerator()
valid_gen = ts_gen.flow_from_dataframe(val_df, x_col='path', y_col='label', target_size=img_size,
                                       class_mode='categorical', color_mode='rgb', shuffle=True, batch_size=batch_size)

test_gen = ts_gen.flow_from_dataframe(test_df, x_col='path', y_col='label', target_size=img_size,
                                      class_mode='categorical', color_mode='rgb', shuffle=False, batch_size=batch_size)


history = model.fit(x= train_gen , epochs = 30, verbose = 1, validation_data= valid_gen,validation_steps = None , shuffle = False)


loss = history.history['loss']
val_loss = history.history['val_loss']
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
epochs = range(1, len(loss) + 1)

plt.figure(figsize=(10, 6))
plt.plot(epochs, loss, label='loss', marker='o', markersize=6)
plt.plot(epochs, val_loss, 'r-', label='Val loss', marker='o', markersize=6)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Ошибки в обучении')
plt.legend()
plt.grid(True)
plt.show()


y_true = test_gen.classes
y_pred = model.predict(test_gen)
y_pred = np.argmax(y_pred, axis=1)

accuracy = accuracy_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Recall: {recall}')
print(f'Precision: {precision}')
print(f'F1-score: {f1}')


model = Sequential([
    base_model,
    BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001),
    Dense(256, kernel_regularizer=regularizers.l2(0.016), 
          activity_regularizer=regularizers.l1(0.006),
          bias_regularizer=regularizers.l1(0.006),
          activation='relu'),
    Dropout(rate=0.4, seed=75),
    Dense(num_class, activation='softmax')
])

model.compile(Adamax(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

print("Model Summary:")
model.summary()

history = model.fit(x= train_gen , epochs = 30, verbose = 1, validation_data= valid_gen,validation_steps = None , shuffle = False)


loss = history.history['loss']
val_loss = history.history['val_loss']
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

epochs = range(1, len(loss) + 1)

plt.figure(figsize=(10, 6))
plt.plot(epochs, loss, label='loss', marker='o', markersize=6)
plt.plot(epochs, val_loss, 'r-', label='Val loss', marker='o', markersize=6)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Ошибки в обучении')
plt.legend()
plt.grid(True)
plt.show()


y_true = test_gen.classes
y_pred = model.predict(test_gen)
y_pred = np.argmax(y_pred, axis=1)

accuracy = accuracy_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Recall: {recall}')
print(f'Precision: {precision}')
print(f'F1-score: {f1}')

check_model_predictions(test_df, y_pred, 12)

