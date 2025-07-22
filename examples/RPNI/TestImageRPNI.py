import sys
sys.path.insert(0, './src')
sys.path.insert(0, './modified_packages')
from tee import Tee
import tensorflow as tf
import numpy as np
from alibi.explainers import AnchorImage
from skimage.color import gray2rgb
from dfa_operatopn import dfa_intersection, get_base_dfa, merge_parallel_edges, merge_linear_edges

np.random.seed(0)
tf.random.set_seed(0)

# 載入資料
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data() # MNIST: (28, 28, 1)
x_train = x_train / 255.0
x_test = x_test / 255.0

# 訓練模型
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=3, validation_data=(x_test, y_test))
x_test_rgb = np.stack([gray2rgb(x) for x in x_test], axis=0) # 灰階轉彩色 (AnchorImage 預設用 RGB)

# 進行預測
def predict_fn(images): # images: shape (n,28,28,3)
    gray_images = images[..., 0]
    probs = model.predict(gray_images)
    return np.argmax(probs, axis=1)  # shape: (n,)

#　fit anchor explainer
segmentation_fn = 'slic'
n_segments = 3 # 分割的區塊數量
compactness = 10 # 分割區塊的緊湊度
sigma = .5 # 分割區塊的平滑度
start_label = 0 # 分割區塊的起始標籤
kwargs = {'n_segments': n_segments, 'compactness': compactness, 'sigma': sigma, 'start_label': start_label}
explainer = AnchorImage(predict_fn, image_shape=(28,28,3), segmentation_fn=segmentation_fn, segmentation_kwargs=kwargs, images_background=None)

# 解釋圖片
image = x_test_rgb[0]
image_input = np.expand_dims(image, axis=0)  # # 增加 batch 維度（模型預測需要 4D）shape: (1, 28, 28, 3)
pred = predict_fn(image_input)

coverage_samples = 1000
batch_size = 50
min_samples_start = 30
n_covered_ex = 50
threshold = 0.95

with open("TestImageRPNI.txt", "w", encoding="utf-8") as log_file:
    sys.stdout = Tee(sys.stdout, log_file) # 同時輸出到終端和檔案
    print("Image: %s" % image)
    print("Prediction: %s" % pred)
    explanation = explainer.explain('Image', image, coverage_samples = coverage_samples, batch_size=batch_size, min_samples_start=min_samples_start, n_covered_ex=n_covered_ex, threshold=threshold, p_sample=.5, tau=0.25)

    # Anchor 結果
    print('Anchor: %s' % explanation.anchor)
    # print('Anchor Precision: %.4f' % explanation.precision)
    # print('Anchor Coverage: %.4f' % explanation.coverage)
    # print('Segments: %s' % explanation.segments)
    mab = explainer.mab # 取出學習紀錄

    # 計算 DFA Intersection
    alphabet_map = {} # 建立 dfa 的字母表映射
    segments_num = explanation.segments.max() + 1 # 實際分割區塊數量
    for i in range(segments_num):
        alphabet_map[i] = [0, 1]

    sub_dfa = get_base_dfa(alphabet_map)
    print("sub dfa:", sub_dfa)
    inter_dfa = dfa_intersection(mab.dfa, sub_dfa)
    print("intersection dfa:", inter_dfa)
    dfa = merge_parallel_edges(inter_dfa)
    dfa = merge_linear_edges(dfa)
    print("final dfa:", dfa)

    sys.stdout = sys.__stdout__ # 恢復 stdout

# plt.imshow(data[i])
# plt.savefig('sample_grid.png')
# plt.imshow(explanation.anchor)
# plt.savefig('anchor.png')
# plt.imshow(explanation.segments)
# plt.savefig('segments.png')
