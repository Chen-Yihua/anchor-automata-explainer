# anchor-automata-explainer

![](https://img.shields.io/badge/python-3.10%2B-blue)

**Anchor-Automata-Explainer** 是一個結合 Anchor 解釋器與 DFA/RPNI 學習的解釋型 AI 工具，支援 Tabular、Text、Image 類型資料，並能將模型行為以可視化自動機（DFA）規則呈現，幫助理解模型決策依據。

本專案在原 anchor 進行以下增強與記錄：

- 支援完整 anchor 解釋流程中的 intermediate state 與訓練紀錄。
- 每輪 anchor 搜尋與樣本覆蓋率、精度等評估皆可存取。
- 自定義 anchor 狀態結構如下（參數說明與 anchor 內部更動見 [DEV.md](./DEV.md)）。

---

## 目錄
- [安裝方式](#安裝方式)
- [資料夾結構](#資料夾結構)
- [快速開始](#快速開始)
- [自定義模型](#自定義模型)
- [進階開發/參數詳解](#進階開發/參數詳解)
- [引用與感謝](#引用與感謝)

---

## 安裝方式

1. **Clone 專案**
   ```bash
   git clone https://github.com/Chen-Yihua/anchor-automata-explainer.git
   cd anchor-automata-explainer
   ```
2. **建立虛擬環境**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows
   ```
4. **安裝依賴套件**
   * 本專案有自訂修改過的 alibi 套件，需指定安裝路徑，請勿安裝官方 PyPI 版本的 alibi，要用這裡的修改版！
   ```bash
   pip install -r requirements.txt
   ```

---

## 資料夾結構
```bash
├── src/                    # 主程式碼與 utils
├── examples/               # Tabular, Text, Image 各種範例程式
├── modified_packages/      # 修改過的 alibi 套件等
├── requirements.txt        # 依賴套件列表
├── README.md
├── DEV.md
```
* src/：主要 class、功能模組（如 automaton、robot_operation 等）
* examples/：各種 anchor + automaton 應用範例（如 TestRobotTabularRPNI.py、TestTextRPNI.py 等）
* modified_packages/：存放你自行修改過的套件，務必安裝這裡的版本
* requirements.txt：請不要 pip install 原版 alibi，要用這裡的 wheel 檔

---

## 快速開始

**以 tabular 為例，解釋並產生 DFA：**

```bash
python examples/RPNI/TestRobotTabularRPNI.py
```
執行結果會輸出 anchor explanation、anchor coverage、訓練過程 log，以及建構出的 DFA。

---

## 自定義模型

依資料類型，請把原始資料整理成以下格式，以供 fetch_xxx 讀取

1. **Tabular（表格）**
   * 檔案：CSV（UTF-8），首列為欄名。
   * 欄位：
      * 其中一欄是 標籤（分類：文字或整數皆可；回函式中會轉成 0…K-1）。
      * 其他欄為 特徵；若有類別型特徵，保持原文字（函式內會 LabelEncoder，並將對應寫入 category_map（key = 欄位 index, value = 類別字串列表））。
   * 缺值：建議以空字串或 ?；我們會在函式內處理（例如轉成眾數/特定符號），類似 fetch_adult 的做法。
   * 檔案擺放：
      * 單檔：data.csv
      * 多檔：可放 train.csv / test.csv（函式要能合併或分別回傳）。
            
2. **Text（文本分類）**
   * 檔案：
      A. 一個 tar.gz 裡面放兩個資料夾（或兩個檔），分別代表兩個 label（或多個 label）。
      B. 或一個 TSV/CSV：欄位 text,label。
   * 內容：每行一個樣本；label 建議用可讀字串（函式內轉 0…K-1），做法可參考 fetch_movie_sentiment 的分兩類讀法。
     
3. **Image（影像分類）**
   * 資料夾結構（建議）：
     ```bash
      dataset/
        train/
          class_a/ *.jpg
          class_b/ *.jpg
          ...
        test/
          class_a/ *.jpg
          class_b/ *.jpg
          ...
     ```
   * 尺寸：不需事先統一；函式可提供 target_size 參數做 resize（如 load_cats）。
   * 標籤：用資料夾名當類別名；函式內會建立 str_to_int 與 int_to_str 對應，類似 fetch_imagenet_10 風格。

本模組提供 fetch_xxx 函式，幫助載入不同類型的資料集 (tabular、text、image) : 

```bash
return_X_y=True
→ (X, y)，其中 X 為特徵，y 為整數化標籤
return_X_y=False
→ 回傳 Bunch 物件，包含：
   * data：特徵
   * target：標籤
   * feature_names（tabular 用）
   * target_names（分類標籤名稱）
   * category_map（類別特徵的對應表）
   * 其他依 dataset 而定（例如：int_to_str_labels、mean_channels）
```

以下為可擴充的 fetch_custom_dataset 範例，可依資料類型進行修改，輸出格式一致即可。

```bash
def fetch_custom_dataset(
    source: str,
    mode: str,  # "tabular" | "text" | "image"
    return_X_y: bool = False,
    target_col: Optional[str] = None,
    target_size: Optional[Tuple[int, int]] = None,
) -> Union[Bunch, Tuple[np.ndarray, np.ndarray]]:
    """
    通用 dataset 載入器樣板
    - source: URL 或本地路徑
    - mode:   tabular / text / image
    - return_X_y: True → (X, y)，False → Bunch
    - target_col: tabular 的標籤欄位
    - target_size: image resize 用
    """
    ...
    # 可參考 fetch_adult / fetch_movie_sentiment / load_cats 的實作
```

使用範例
```bash
# 1. Tabular
X, y = fetch_custom_dataset("data.csv", mode="tabular", target_col="label", return_X_y=True)

# 2. Text
bunch = fetch_custom_dataset("reviews.csv", mode="text")
print(bunch.data[:3], bunch.target[:3])

# 3. Image
X, y = fetch_custom_dataset("dataset", mode="image", target_size=(224, 224), return_X_y=True)
```

**以 tabular 為例，解釋自定義預測模型並產生 DFA：**

  1. **準備 Tabular 資料**
     * 可用 DataFrame 或 numpy array，特徵可含類別型、數值型
     * Tabular 資料須包含以下物件：
       * data ((N, M) array)
       * target ((N,) 1-D array)
       * feature_names：欄位名稱（list，長度為 M）
       * category_map：每個類別型欄位的所有可能值 (dict：key=欄位 index，value=此欄所有類別的 list)
  2. **建立分類器**
     * 你可以用任何 scikit-learn 類模型，或自定義規則 function
     * 必須提供 predict_fn = lambda x: model.predict(x)（確保 shape=(N, M)），或自定義 function，輸入 2D array，回傳 label list
  3. **AnchorTabular 解釋流程**
     * 參考 TestTabularBinary.py、TestTextRPNI.py 的寫法
     * 將你的 data、feature_names、category_map、predict_fn 帶入 AnchorTabular，以製作 explainer
     * explainer.fit(data) 後，直接呼叫：
       ```python
       explanation = explainer.explain('Tabular', test_instance, ...) # anchor 類型可選 'Text', 'Tabular', 'Image'
       ```
  4. **計算 DFA Intersection**
     * 訓練後，可用 explainer.mab 取得 anchor 學習紀錄
     * 計算 DFA intersection 可參考下列程式片段：
       ```python
       from dfa_operatopn import dfa_intersection, get_base_dfa, merge_linear_edges, merge_parallel_edges

        alphabet_map = {i: [0, 1, 2] for i in range(len(feature_names))}  # 依你的特徵型態調整
        sub_dfa = get_base_dfa(alphabet_map)
        inter_dfa = dfa_intersection(explainer.mab.dfa, sub_dfa)
        dfa = merge_parallel_edges(inter_dfa)
        dfa = merge_linear_edges(dfa)
        print("final dfa:", dfa)
       ```
**範例參考**

| 類型           | 範例檔案                      |
| ------------ | ------------------------- |
| Tabular      | `TestTabularBinary.py`    |
| Tabular（自訂）  | `TestRobotTabularRPNI.py` |
| Text         | `TestTextRPNI.py`         |
| Image        | `TestImageRPNI.py`        |
* 建議複製上述檔案結構，調整資料/模型部分即可

---

## 進階開發/參數詳解

- anchor 各類參數與內部狀態說明，請見 [DEV.md](./DEV.md)
- alibi 主要修改記錄、anchor 內部實作細節，請見 [DEV.md](./DEV.md)
---
