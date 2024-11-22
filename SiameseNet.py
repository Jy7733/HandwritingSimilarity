import os
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
import numpy as np
from tensorflow.keras.preprocessing import image
from sklearn.model_selection import train_test_split
import random
from tqdm import tqdm
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler
import warnings

import os
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

def scheduler(epoch, lr):
    if epoch > 10:
        lr = lr * float(tf.math.exp(-0.1))
    return float(lr) 

# 이미지 크기를 전역 변수로 정의
IMG_SIZE = (128, 128)

def load_pairs_and_labels(data_dir, max_images_per_class=400, max_pairs_per_class=200):
    print("이미지 쌍 및 라벨 생성 시작")
    same_pairs = []
    diff_pairs = []

    class_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    print("클래스 디렉토리:", class_dirs)

    for class_dir in tqdm(class_dirs, desc="같은 클래스 쌍 생성 진행률"):
        class_path = os.path.join(data_dir, class_dir)
        images = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.jpeg', '.png'))][:max_images_per_class]

        # 같은 클래스 내의 서로 다른 두 이미지 선택하여 쌍으로 묶음 
        for i in range(len(images)):
            for j in range(i + 1, min(i + 1 + max_pairs_per_class, len(images))):
                img1_path = os.path.join(class_path, images[i])
                img2_path = os.path.join(class_path, images[j])
                same_pairs.append((img1_path, img2_path))

    total_diff_pairs = len(class_dirs) * (len(class_dirs) - 1) // 2
    with tqdm(total=total_diff_pairs, desc="다른 클래스 쌍 생성 진행률") as pbar:
        for i in range(len(class_dirs)):
            for j in range(i + 1, len(class_dirs)):
                class1_path = os.path.join(data_dir, class_dirs[i])
                class2_path = os.path.join(data_dir, class_dirs[j])
                images1 = [f for f in os.listdir(class1_path) if f.endswith(('.jpg', '.jpeg', '.png'))][:max_images_per_class]
                images2 = [f for f in os.listdir(class2_path) if f.endswith(('.jpg', '.jpeg', '.png'))][:max_images_per_class]

                for _ in range(min(max_pairs_per_class, len(images1), len(images2))):
                    img1_path = os.path.join(class1_path, random.choice(images1))
                    img2_path = os.path.join(class2_path, random.choice(images2))
                    diff_pairs.append((img1_path, img2_path))

                pbar.update(1)

    # 같은 쌍의 수를 다른 쌍의 수에 맞게 줄임
    same_pairs = random.sample(same_pairs, len(diff_pairs))
                
    # 같은 쌍과 다른 쌍의 비율 출력
    same_len = len(same_pairs)
    diff_len = len(diff_pairs)
    print(f"같은 클래스 쌍: {same_len}, 다른 클래스 쌍: {diff_len}")
    print(f"비율 (같은 쌍:다른 쌍) = {same_len}:{diff_len} = {same_len / diff_len:.2f}")


    same_labels = np.ones(len(same_pairs))
    diff_labels = np.zeros(len(diff_pairs))

    X_pairs = np.array(same_pairs + diff_pairs)
    y_labels = np.concatenate((same_labels, diff_labels))

    print("이미지 쌍 및 라벨 생성 완료")
    return X_pairs, y_labels


def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    return img_array / 255.0

def euclidean_distance(vectors):
    (featA, featB) = vectors
    sum_squared = K.sum(K.square(featA - featB), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_squared, K.epsilon()))

def build_siamese_model(input_shape):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    
    for layer in base_model.layers[:-10]:
        layer.trainable = False
    
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation='relu')(x)

    feature_extractor = models.Model(base_model.input, x)

    inputA = layers.Input(shape=input_shape)
    inputB = layers.Input(shape=input_shape)

    featsA = feature_extractor(inputA)
    featsB = feature_extractor(inputB)

    distance = layers.Lambda(euclidean_distance)([featsA, featsB])
    outputs = layers.Dense(1, activation="sigmoid")(distance)

    model = models.Model(inputs=[inputA, inputB], outputs=outputs)
    return model

def siamese_generator(X_pairs, y_labels, batch_size, is_training=True):
    num_samples = len(X_pairs)
    datagen = ImageDataGenerator(
        rotation_range=40,  # 더 넓은 회전 범위
        # width_shift_range=0.3,  # 더 넓은 수평 이동
        # height_shift_range=0.3,  # 더 넓은 수직 이동
        # shear_range=0.2,  # 전단 변형 추가
        # zoom_range=0.2,  # 줌 변형 추가
        horizontal_flip=True,
        vertical_flip=True,  # 수직 뒤집기 추가
        # fill_mode='nearest'
    )
    while True:
        indices = np.random.permutation(num_samples)
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_indices = indices[start:end]
            batch_pairs = X_pairs[batch_indices]
            batch_labels = y_labels[batch_indices]

            imagesA = np.array([preprocess_image(img[0]) for img in batch_pairs])
            imagesB = np.array([preprocess_image(img[1]) for img in batch_pairs])

            if is_training:
                imagesA = next(datagen.flow(imagesA, batch_size=batch_size, shuffle=False))
                imagesB = next(datagen.flow(imagesB, batch_size=batch_size, shuffle=False))

            batch_labels = np.array(batch_labels, dtype=np.float32)

            yield ((imagesA, imagesB), batch_labels)

def train_or_load_model(data_dir, model_path):
    if os.path.exists(model_path):
        print("모델이 이미 존재합니다. 모델을 불러옵니다.")
        model = tf.keras.models.load_model(model_path, custom_objects={'euclidean_distance': euclidean_distance})
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model, 0

    print("모델을 학습합니다.")
    X_pairs, y_labels = load_pairs_and_labels(data_dir)

    X_train, X_val, y_train, y_val = train_test_split(X_pairs, y_labels, test_size=0.2, random_state=42)
    input_shape = IMG_SIZE + (3,)
    siamese_model = build_siamese_model(input_shape)

    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
    siamese_model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    batch_size = 64

    # 데이터셋 준비
    train_dataset = tf.data.Dataset.from_generator(
        lambda: siamese_generator(X_train, y_train, batch_size=batch_size, is_training=True),
        output_signature=(
            (tf.TensorSpec(shape=(None,) + IMG_SIZE + (3,), dtype=tf.float32),
             tf.TensorSpec(shape=(None,) + IMG_SIZE + (3,), dtype=tf.float32)),
            tf.TensorSpec(shape=(None,), dtype=tf.float32)
        )
    ).prefetch(tf.data.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_generator(
        lambda: siamese_generator(X_val, y_val, batch_size=batch_size, is_training=False),
        output_signature=(
            (tf.TensorSpec(shape=(None,) + IMG_SIZE + (3,), dtype=tf.float32),
             tf.TensorSpec(shape=(None,) + IMG_SIZE + (3,), dtype=tf.float32)),
            tf.TensorSpec(shape=(None,), dtype=tf.float32)
        )
    ).prefetch(tf.data.AUTOTUNE)

    steps_per_epoch = len(X_train) // batch_size
    validation_steps = len(X_val) // batch_size

    siamese_model.fit(
        train_dataset,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_dataset,
        validation_steps=validation_steps,
        epochs=15,  
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3),
            LearningRateScheduler(scheduler)   # 학습률 스케줄러 추가
        ]
    )

    siamese_model.save(model_path)
    print(f"모델이 '{model_path}'에 저장되었습니다.")
    return siamese_model, 1


def predict_similarity(model, img1_path, img2_path):
    img1 = preprocess_image(img1_path)
    img2 = preprocess_image(img2_path)
    img1 = np.expand_dims(img1, axis=0)
    img2 = np.expand_dims(img2, axis=0)
    prediction = model.predict([img1, img2])
    return prediction[0][0]

def main():
    data_dir = './training'
    model_path = 'siamese_model.h5'

    model, status = train_or_load_model(data_dir, model_path)

    img_pairs = [
        # ('./pre/00140002003.jpg', './pre/00140079001.jpg'), 
        # ('./pre/00140079002.jpg', './pre/00340002003.jpg'),
        # ('./pre/00340002006.jpg', './pre/00440002008.jpg'),
        # ('./pre/테스트1-1.jpeg','./pre/테스트1-2.jpeg')
        # ('/Users/jaeyoung/Desktop/handwiritngSimilarity/pre/테스트1-2.jpeg', '/Users/jaeyoung/Desktop/handwiritngSimilarity/pre/image2.JPG')
        ('/Users/jaeyoung/Desktop/handwiritngSimilarity/pre/image3.JPG','/Users/jaeyoung/Desktop/handwiritngSimilarity/pre/image4.JPG')

    ]

    for i, (img1_path, img2_path) in enumerate(img_pairs, 1):
        similarity_score = predict_similarity(model, img1_path, img2_path)
        if(similarity_score >= 0.7) :
            print(f"Similarity Score {i}: {similarity_score:.2f} --  필체가 유사합니다. ")
        else :
            print(f"Similarity Score {i}: {similarity_score:.2f} --  필체가 유사하지 않습니다. ")
        

if __name__ == "__main__":
    main()