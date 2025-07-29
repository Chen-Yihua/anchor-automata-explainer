# 開發者文件

本文件說明 Anchor-Automata-Explainer 的進階開發資訊，包括 anchor 內部 state 結構、sample_fcn 功能、explain 主要參數，以及專案自訂 anchor 實作的修改紀錄，方便二次開發、debug、客製任務。

1. **anchor state（各輪學習過程主要紀錄）**
   
    **Text 類型 anchor state 範例**
    ```python
    't_idx':         # dict，每個候選 anchor 覆蓋到的樣本 index
    't_nsamples':    # dict，每個候選 anchor 已抽樣本數（估 precision 用）
    't_positives':   # dict，每個候選 anchor 抽到 label 與原預測一致的數量
    'data':          # 遮罩後的 data（0 為遮、1 為不遮）
    'prealloc_size': # data 預設空間
    'raw_data':      # 原始資料（局部樣本）
    'labels':        # 每個樣本的預測 label
    'current_idx':   # 下一筆樣本存放 data[] 的 index
    'n_features':    # 特徵數量
    't_coverage_idx':# 每個候選 anchor 覆蓋到哪些樣本的 index (估 coverage 用）
    't_coverage':    # 每個 anchor 覆蓋率
    'coverage_data': # 用於估算 coverage 的全域擾動樣本集合
    't_order':       # 對應候選 anchor 的樣本順序
    ```
    
    **Tabular 類型 anchor state 範例**
    ```python
    't_coverage':      # defaultdict(lambda: 0.)        # anchor 覆蓋率
    't_coverage_idx':  # defaultdict(set)               # anchor 覆蓋到的樣本索引
    't_covered_true':  # defaultdict(None)              # anchor 命中且 label=預測
    't_covered_false': # defaultdict(None)              # anchor 命中但 label≠預測
    't_idx':           # defaultdict(set)               # 每 anchor 覆蓋的 index
    't_nsamples':      # defaultdict(lambda: 0.)        # 每 anchor 抽樣數
    't_order':         # defaultdict(list)              # feature 順序
    't_positives':     # defaultdict(lambda: 0.)        # anchor 命中 label=預測
    'prealloc_size':   # 初始記憶體分配（樣本總筆數 = batch * cache_size)
    'data':            # 所有抽樣樣本
    'labels':          # 樣本的預測 label
    'current_idx':     # 下一筆樣本 index
    'n_features':      # 特徵數
    'coverage_data':   # 所有抽樣資料（算覆蓋率用）
    ```

2. **sample_fcn 物件屬性說明（只有 tabular state 中有）**
   
    ```python
    feature_names : 欄位名稱（如 ['Age', 'Workclass', 'Education', ...]）
    n_records : 總樣本數
    train_data : one hot 之後的資料
    d_train_data : 原始資料（未編碼）
    enc2feat_idx : 將 one hot 後特徵 index 對應回原始特徵
    val2idx : 原始特徵值對應到 one hot index
    
    # 類別資料
    categorical_features : 哪些欄位為類別特徵 (用 index 表示）
    feature_values : 每個類別特徵所有值（如 {2: ['Bachelors', ...]})
    cat_lookup : 編碼值與原始特徵值對應
    
    # 數值資料
    ord_lookup : bin 編碼與原始區間對應
    disc : 分箱器

    # 例子
    feature_names = ['Age', 'Workclass', 'Education', ..., 'Occupation', 'Sex', 'Hours-per-week']
    categorical_features = [1, 2, 3, 4, 5, 6, 7, 8]  # 用 index 表示
    feature_values[2] = ['Bachelors', 'HS-grad', 'Masters', 'Doctorate', 'Some-college']
    ord_lookup[0] = ['(17,25]', '(25,35]', '(35,45]', '(45,55]', ...]
    # Age 在第 3 個 bin，例如 (35,45]（用 ord_lookup[0][2]）
    ```

3. **explain 主要參數**
   
     | 參數名稱                  | 預設值   | 說明                  |
   | --------------------- | ----- | ------------------- |
   | **threshold**         | 0.95  | Anchor precision 門檻 |
   | delta                 | 0.1   | 精度的信賴區間(90%)        |
   | tau                   | 0.15  | 容忍精度誤差              |
   | **batch\_size**       | 100   | 每批重抽樣數              |
   | **coverage\_samples** | 10000 | 用於 coverage 的全域樣本數  |
   | beam\_size            | 1     | 每輪保留 anchor 數       |
   | stop\_on\_first       | False | 是否找到就停              |
   | max\_anchor\_size     | None  | 最大 anchor 長度        |
   | min\_samples\_start   | 100   | 初始精度估計數量            |
   | **n\_covered\_ex**    | 10    | 每 anchor 儲存的正反例數    |
   | binary\_cache\_size   | 10000 | 抽樣 cache 大小         |
   | cache\_margin         | 1000  | 抽樣 cache 預留         |
   | verbose               | False | 顯示詳情                |
   | verbose\_every        | 1     | 幾輪輸出一次              |

4. **Precision、Accuracy、Coverage 定義**
   
   * Anchor
     - Coverage : 符合 anchor 條件的樣本 / 所有 Coverage data
     - 訓練集 Precision : 符合 anchor 且 label 為 1 / 符合某 anchor 的訓練樣本
     - 測試集 Precision : 符合 anchor 且 label 為 1 / 符合某 anchor 的 Coverage data
     - 測試集 Accuracy : 符合 anchor 且 label 為對的 / 所有 Coverage data
       
   * Automaton
     - Coverage : 自動機接受的樣本 / Coverage data
     - 訓練集 Precision : 自動機接受且 label 為 1 / 符合某 anchor 的訓練樣本
     - 測試集 Precision : 自動機接受且 label 為 1 / 符合某 anchor 的 Coverage data
     - 測試集 Accuracy : 自動機接受且 label 為對的 / 所有 Coverage data
     
  - True label 為 1 : 模型預測與原始預測一致
  - True label 為對的 :  True label 是 0 (自動機拒絕) + True label 是  1 (自動機接受)

6. **anchor_base.py 主要修改紀錄**
   
     1. `anchor_base.py` 中的 `anchor_beam()` line 748 - line 749 : 
          * 將 `coverage_raw` (初始抽樣樣本的原始格式)、`coverage_label` (預測結果) 加入 state
     2. `anchor_base.py` 中的 `draw_samples()` line 379 - line 427 : 
        * 更新學習過程中的抽樣紀錄 (每次抽樣的二元值、原始值與其對應 label)
        * 於學習過程中生成自動機，並計算自動機的 Coverage/Precision/Accuracy
     3. `anchor_base.py` 中的 `update_state()` line 557 - line 560 : 
        * 於學習過程中，計算 Anchor 的 Coverage/Precision/Accuracy

7. **進階開發提醒**
   
   * 修改 anchor 行為請直接修改 `modified_packages/alibi/` 下相關 .py
   * `explainer.mab.state` 可取得所有學習/精度/覆蓋等狀態
   * 如需完整追蹤 anchor 行為、樣本擾動、label、precision 等統計，可直接輸出 state 物件
   * 若需重寫解釋流程、加自動機結合分析，可參考 `examples/RPNI/` 內的各型範例
   * 資料集製作可參考 `alibi.datasets` 內的相關方法 及 `src/robot_operation` 的 `fetch_robot` 方法(自定義 tabular 例子)
     
