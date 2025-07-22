import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from anchor_dfa_learning import AUTO_INSTANCE, get_precision_samples, add_position_to_sample, get_pos_samples, get_neg_samples, get_covered_samples
# from alibi.datasets import load_cats
from alibi.explainers import AnchorImage
from skimage.color import gray2rgb

# 載入資料、model
# model = tf.keras.applications.InceptionV3(weights="imagenet")
# image_shape = (299, 299, 3)
# data, labels = load_cats(target_size=image_shape[:2], return_X_y=True)
# print(f'Images shape: {data.shape}')
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data() # MNIST: (28, 28, 1)
x_train = x_train / 255.0
x_test = x_test / 255.0
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=3, validation_data=(x_test, y_test))
x_test_rgb = np.stack([gray2rgb(x) for x in x_test], axis=0) # 灰階轉彩色 (AnchorImage 預設用 RGB)

# 進行預測
# images = tf.keras.applications.inception_v3.preprocess_input(data)
# preds = model.predict(images)
# label = tf.keras.applications.imagenet_utils.decode_predictions(preds, top=3)
# print(label[0])
# predict_fn = lambda x: model.predict(x)
def predict_fn(images): # images: shape (n,28,28,3)
    gray_images = images[..., 0]  # 轉成灰階，取第一個 channel
    return model.predict(gray_images)

# 解釋句子
segmentation_fn = 'slic'
kwargs = {'n_segments': 15, 'compactness': 20, 'sigma': .5, 'start_label': 0}
explainer = AnchorImage(predict_fn, image_shape=(28,28,3), segmentation_fn=segmentation_fn, segmentation_kwargs=kwargs, images_background=None)

i = 1
# image = images[i]
image = x_test_rgb[0]
np.random.seed(0)
coverage_samples = 1000
batch_size = 50
min_samples_start = 30
n_covered_ex = 1000
threshold = 0.95
explanation = explainer.explain(image, coverage_samples = coverage_samples, batch_size=batch_size, min_samples_start=min_samples_start, n_covered_ex=n_covered_ex, threshold=threshold, p_sample=.5, delta=0.1, tau=0.25)

# Anchor 結果
# print('Anchor: %s' % explanation.anchor)
print('Anchor Precision: %.4f' % explanation.precision)
print('Anchor Coverage: %.4f' % explanation.coverage)
# print('Segments: %s' % explanation.segments)
sampler = explainer.sampler # 取出 sampler
mab = explainer.mab # 取出學習紀錄

# 取出被動學習樣本
precision_samples = get_precision_samples('Image', explanation, mab)
precision_samples = add_position_to_sample(precision_samples)
passive_data = AUTO_INSTANCE.convert_to_alergia_format(precision_samples)
print('被動學習樣本數量: %d' % len(passive_data))

# 生成 dfa
dfa = AUTO_INSTANCE.learn_dfa(list(passive_data))

# 計算 DFA Precision
pos_samples = get_pos_samples('Image', explanation, mab)
pos_samples = add_position_to_sample(pos_samples)
pos_samples = AUTO_INSTANCE.convert_to_alergia_format(pos_samples)

pos_prob = sum(AUTO_INSTANCE.calculate_path_probability(sample) for sample in pos_samples) # 計算 DFA 機率
passive_data_prob = sum(AUTO_INSTANCE.calculate_path_probability(sample) for sample in passive_data)
dfa_precision = pos_prob / passive_data_prob if len(passive_data) > 0 else 0
print(f"MC Precision: {dfa_precision:.4f}")

# 計算 DFA Coverage
all_covered_samples = mab.state['coverage_data']
all_covered_samples = add_position_to_sample(all_covered_samples)
all_covered_samples = AUTO_INSTANCE.convert_to_alergia_format(all_covered_samples)

anchor_covered_samples = get_covered_samples('Image', explanation, mab)
anchor_covered_samples = add_position_to_sample(anchor_covered_samples)
anchor_covered_samples = AUTO_INSTANCE.convert_to_alergia_format(anchor_covered_samples)

accepted_prob = sum(AUTO_INSTANCE.calculate_path_probability(sample) for sample in anchor_covered_samples) 
coverage_data_prob = sum(AUTO_INSTANCE.calculate_path_probability(sample) for sample in all_covered_samples)
dfa_coverage = accepted_prob / coverage_data_prob if coverage_samples > 0 else 0
print(f"accepted prob: {accepted_prob:.4f}")
print(f"MC Coverage: {dfa_coverage:.4f}")

# plt.imshow(data[i])
# plt.savefig('sample_grid.png')
# plt.imshow(explanation.anchor)
# plt.savefig('anchor.png')
# plt.imshow(explanation.segments)
# plt.savefig('segments.png')

