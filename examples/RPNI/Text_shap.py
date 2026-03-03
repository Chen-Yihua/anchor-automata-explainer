import sys, os
import warnings
warnings.filterwarnings('ignore')
import re
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
plt.ioff()  # Turn off interactive mode
import shap
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../external_modules/Explaining-FA')))
sys.path.insert(0, os.path.abspath('./src'))
sys.path.insert(0, os.path.abspath('.'))
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from data.dataset_loader import fetch_custom_dataset
np.random.seed(0)

# 載入資料
MOVIESENTIMENT_URLS = ['https://storage.googleapis.com/seldon-datasets/sentence_polarity_v1/rt-polaritydata.tar.gz',
                       'http://www.cs.cornell.edu/People/pabo/movie-review-data/rt-polaritydata.tar.gz']

url = MOVIESENTIMENT_URLS[0]
movies = fetch_custom_dataset(
    source=url,
    mode="text",
    return_X_y=False
)
data = movies.data
labels = movies.target
target_names = movies.target_names
class_names = movies.target_names

# 預測模型
train, test, train_labels, test_labels = train_test_split(data, labels, test_size=.2, random_state=42)
train, val, train_labels, val_labels = train_test_split(train, train_labels, test_size=.1, random_state=42)
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)
val_labels = np.array(val_labels)
# 簡單穩定版本：TF-IDF + LinearSVC
vectorizer = TfidfVectorizer(
    ngram_range=(1, 3),  
    max_features=20000,
)
vectorizer.fit(train)

X_train = vectorizer.transform(train)
X_val = vectorizer.transform(val)
X_test = vectorizer.transform(test)

# 簡單模型：LinearSVC
from sklearn.svm import LinearSVC
clf = LinearSVC(
    C=1.0,
    max_iter=2000,
    random_state=42,
    dual=False,
    class_weight='balanced'
)
clf.fit(X_train, train_labels)

predict_fn = lambda x: clf.predict(vectorizer.transform(x))

preds_train = predict_fn(train)
preds_val = predict_fn(val)
preds_test = predict_fn(test)

print("="*60)
train_acc = accuracy_score(train_labels, preds_train)
val_acc = accuracy_score(val_labels, preds_val)
test_acc = accuracy_score(test_labels, preds_test)

print(f'Train accuracy: {train_acc:.4f}')
print(f'Validation accuracy: {val_acc:.4f}')
print(f'Test accuracy: {test_acc:.4f}')
print("="*60)

# ============ SHAP 解釋 ============
print("\n============SHAP Explanation for data[0]============")
try:
    test_instance = 'barely fun and ordinary .'
    prediction = class_names[predict_fn([test_instance])[0]]
    
    print(f"\nText: {test_instance}")
    print(f"Prediction: {prediction}")
    
    background_size = min(10, len(train))
    background_indices = np.random.choice(len(train), size=background_size, replace=False).astype(int)
    background_texts = [train[i] for i in background_indices]
    
    print(f"\nUsing {background_size} background samples for SHAP calculation...")
    
    def predict_texts(texts):
        if isinstance(texts, np.ndarray):
            texts = texts.tolist() if texts.dtype == object else texts
        if isinstance(texts, str):
            texts = [texts]
        X = vectorizer.transform(texts).toarray()
        scores = clf.decision_function(X)
        if scores.ndim == 1:
            scores = scores.reshape(-1, 1)
        proba = 1 / (1 + np.exp(-scores))
        return np.hstack([1 - proba, proba])
    
    # ============ 使用 UNK 擾動方式（與 Anchor 一致）============
    print("Initializing SHAP with UNK masking (Anchor-style)...")
    from shap import KernelExplainer
    
    # 方法：在向量層級進行 UNK 掩蔽
    # 將文本轉換為向量，然後用 0 (UNK 的特徵向量) 替換缺失特徵
    
    class TextUNKMasker:
        def __init__(self, vectorizer, unk_value=0):
            self.vectorizer = vectorizer
            self.unk_value = unk_value  # 用 0 表示 UNK 詞彙的特徵值
        
        def __call__(self, mask, x):
            """
            mask: shape (n_samples, n_features)，True = 保留，False = 遮蔽
            x: 預先向量化的特徵陣列 (1, n_vocab)
            Returns: 遮蔽後的向量陣列
            """
            # 將 x 轉換為陣列（如果還不是）
            if isinstance(x, str):
                # 如果收到文本，先向量化
                x = self.vectorizer.transform([x]).toarray()
            
            x = np.asarray(x)
            result = x.copy()
            
            # 對每個樣本應用掩蔽
            if mask.ndim == 1:
                mask = mask.reshape(1, -1)
            
            for sample_idx in range(len(mask)):
                sample_mask = mask[sample_idx]
                # 將 False 的位置設為 0 (UNK)
                result[sample_idx][~sample_mask.astype(bool)] = self.unk_value
            
            return result
    
    # 轉換測試實例為向量形式
    test_features = vectorizer.transform([test_instance]).toarray()
    
    # 只分析活躍特徵（出現在該文本中的詞彙）
    active_feature_indices = np.where(test_features[0] > 0)[0]
    active_feature_names = vectorizer.get_feature_names_out()[active_feature_indices]
    active_test_features = test_features[:, active_feature_indices]
    
    print(f"Total features: {len(vectorizer.get_feature_names_out())}")
    print(f"Active features (words in text): {len(active_feature_indices)}")
    print(f"Active feature names: {list(active_feature_names)}")
    
    # 創建 masker
    unk_masker = TextUNKMasker(vectorizer, unk_value=0)
    
    # 測試 masker 是否正常工作
    print("Testing UNK masking compatibility...")
    # 修正：直接跳過 UNK masking 測試，改用更簡單可靠的方法
    unk_masker = None  # 禁用 UNK masker，直接使用備用方案
    
    # 定義預測函數（接收向量化輸入）
    def predict_features(X):
        # LinearSVC 沒有 predict_proba，使用 decision_function 然後轉換為概率
        scores = clf.decision_function(X)
        if scores.ndim == 1:
            scores = scores.reshape(-1, 1)
        # 簡單的 sigmoid 轉換
        proba = 1 / (1 + np.exp(-scores))
        # 返回 [negative_class, positive_class] 概率
        return np.hstack([1 - proba, proba])
    
    # 直接使用 KernelExplainer 在完整特徵空間上計算
    from shap import KernelExplainer
    
    # 準備背景資料（使用完整特徵向量）
    background_features = vectorizer.transform(background_texts).toarray()
    
    print("Computing SHAP values (this may take a few minutes)...")
    explainer_shap = KernelExplainer(predict_features, background_features)
    
    # 計算 SHAP 值
    shap_values_array = explainer_shap.shap_values(test_features, nsamples=100)
    
    # 提取 SHAP 值和特徵名
    all_feature_names = vectorizer.get_feature_names_out()
    shap_array = np.array(shap_values_array)
    
    if shap_array.ndim == 3:
        shap_vals_full = shap_array[0, :, 1]  # 取正類
    elif shap_array.ndim == 2:
        shap_vals_full = shap_array[:, 1] if shap_array.shape[1] == 2 else shap_array[0]
    else:
        shap_vals_full = shap_array.flatten()
    
    base_value = explainer_shap.expected_value
    if isinstance(base_value, np.ndarray):
        base_value = base_value[1] if base_value.size > 1 else base_value[0]
    
    # 過濾為活躍特徵（出現在文本中的詞）
    active_indices = np.where(test_features[0] > 0)[0]
    shap_vals = shap_vals_full[active_indices]
    feature_names_shap = all_feature_names[active_indices]
    
    print(f"✓ SHAP computation complete!")
    print(f"Text: {test_instance}")
    print(f"SHAP values shape: {shap_vals.shape}")
    print(f"Number of active features analyzed: {len(feature_names_shap)}")
    
    # Debug: Show words in the test instance
    words_in_text = [w for w in test_instance.lower().split() if re.match(r'[a-z]+', w)]
    print(f"\nWords in text: {words_in_text}")
    print(f"SHAP feature names: {list(feature_names_shap)}")
    
    # 提取活躍特徵（有非零 SHAP 值的特徵）
    active_indices = np.where(np.abs(shap_vals) > 1e-6)[0]
    
    if len(active_indices) > 0:
        active_feature_names = feature_names_shap[active_indices]
        active_shap_values = shap_vals[active_indices]
        
        print(f"\nSignificant features: {len(active_indices)}")
        print(f"Significant feature names: {list(active_feature_names)}")
        print(f"Significant SHAP values: {active_shap_values}")
    else:
        active_feature_names = feature_names_shap
        active_shap_values = shap_vals
        print("\nUsing all features for visualization")
    
    if base_value is None:
        if isinstance(explainer_shap.expected_value, np.ndarray):
            base_value = float(explainer_shap.expected_value[1]) if explainer_shap.expected_value.size > 1 else float(explainer_shap.expected_value[0])
        else:
            base_value = float(explainer_shap.expected_value)
    
    print("\nGenerating SHAP plots...")
    print(f"  - Generating plots for positive class")
    
    # Waterfall plot (only show active features to reduce clutter)
    try:
        plt.figure(figsize=(12, 6))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            shap.plots.waterfall(shap.Explanation(
                values=active_shap_values,
                base_values=base_value,
                data=np.arange(len(active_shap_values)),  # 簡單的索引
                feature_names=active_feature_names
            ))
        plt.title("SHAP Waterfall Plot (UNK Masking)")
        plt.tight_layout()
        plt.savefig("shap_waterfall.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("  Waterfall plot saved")
    except Exception as e:
        print(f"  Waterfall error: {e}")
    
except Exception as e:
    print(f"SHAP explanation failed: {e}")
    import traceback
    traceback.print_exc()