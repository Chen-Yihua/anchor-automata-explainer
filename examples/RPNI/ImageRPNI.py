import sys
sys.path.insert(0, './src')
from tee import Tee
import tensorflow as tf
import numpy as np
from alibi.explainers import AnchorImage
from learner.dfa_learner import dfa_intersection, get_base_dfa, merge_parallel_edges, merge_linear_edges
from data.dataset_loader import fetch_custom_dataset

np.random.seed(0)
tf.random.set_seed(0)

# 載入資料
(X_tr, y_tr), (X_te, y_te) = tf.keras.datasets.mnist.load_data()
b_train = fetch_custom_dataset(
    source=(X_tr, y_tr),
    mode="image",
    color_mode="rgb", # 建議 RGB，SLIC 超像素較穩
    target_size=(28, 28), # 可省略；這裡保持原尺寸
    normalize=False # 回傳 uint8，讓模型內做 Rescaling
)
b_test = fetch_custom_dataset(
    source=(X_te, y_te),
    mode="image",
    color_mode="rgb",
    target_size=(28, 28),
    normalize=False
)
X_train, y_train = b_train.data, b_train.target # (N,28,28,3) uint8
X_test,  y_test  = b_test.data,  b_test.target # (N,28,28,3) uint8

# 預測模型
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 3)),
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax'),
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=3, validation_data=(X_test, y_test), verbose=1)

def predict_fn(images):
    images = np.asarray(images) # (N,H,W,C)
    probs = model.predict(images, verbose=0) # (N,10)
    return np.argmax(probs, axis=1) # (N,)

# sampler 參數
segmentation_fn = 'slic'
n_segments = 3 
compactness = 10 
sigma = .5 
start_label = 0 
kwargs = {'n_segments': n_segments, 'compactness': compactness, 'sigma': sigma, 'start_label': start_label}

#　fit anchor explainer
explainer = AnchorImage(
    predict_fn, 
    image_shape=(28,28,3), 
    segmentation_fn=segmentation_fn, 
    segmentation_kwargs=kwargs, 
    images_background=None
)

# 解釋圖片
image = X_test[0]
image_input = np.expand_dims(image, axis=0)  # 增加 batch 維度（模型預測需要 4D）shape: (1, 28, 28, 3)
pred = predict_fn(image_input)

# explainer 參數
threshold = 0.95
delta = 0.1
tau = 0.15
batch_size = 50
coverage_samples = 1000
beam_size = 1
max_anchor_size = None
min_samples_start = 100
n_covered_ex = 20
p_sample = .5

with open("TestImageRPNI.txt", "w", encoding="utf-8") as log_file:
    sys.stdout = Tee(sys.stdout, log_file) 
    print("Image: %s" % image)
    print("Prediction: %s" % pred)
    explanation = explainer.explain(
        'Image', 
        image, 
        delta = delta,
        tau = tau, 
        beam_size = beam_size, 
        max_anchor_size = max_anchor_size,
        coverage_samples=coverage_samples, 
        batch_size=batch_size, 
        min_samples_start=min_samples_start, 
        n_covered_ex=n_covered_ex, 
        threshold=threshold,
        p_sample=p_sample, 
    )

    # Anchor 結果
    print('Anchor: %s' % explanation.anchor)
    mab = explainer.mab # 取出學習紀錄

    # 計算 DFA Intersection
    alphabet_map = {} # 建立 dfa 的字母表映射
    segments_num = explanation.segments.max() + 1 # 實際分割區塊數量
    for i in range(segments_num):
        alphabet_map[i] = [0, 1]

    sub_dfa = get_base_dfa(alphabet_map)
    # print("sub dfa:", sub_dfa)
    inter_dfa = dfa_intersection(mab.dfa, sub_dfa)
    # print("intersection dfa:", inter_dfa)
    dfa = merge_parallel_edges(inter_dfa)
    dfa = merge_linear_edges(dfa)
    print("final dfa:", dfa)

    sys.stdout = sys.__stdout__ # 恢復 stdout
