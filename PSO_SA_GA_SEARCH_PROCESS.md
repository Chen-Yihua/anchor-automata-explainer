# PSO、SA、GA 在 DFA 搜尋中的流程說明

## 目錄
1. [概述](#概述)
2. [預備知識](#預備知識)
3. [SA - 模擬退火搜尋](#sa--模擬退火搜尋)
4. [GA - 遺傳演算法搜尋](#ga--遺傳演算法搜尋)
5. [PSO - 粒子群最佳化搜尋](#pso--粒子群最佳化搜尋)
6. [三者對比](#三者對比)

---

## 概述

在 DFA（確定性有限自動機）優化問題中，需要在保持 訓練準確度 的前提下，最小化狀態數。SA、GA、PSO 是三種經典啟發式演算法，被適配用於此任務。

### 核心思想
- 目標：找到最小的 DFA（最少狀態數），同時訓練準確度 ≥ 閾值
- 搜尋空間：DFA 狀態空間（通過刪除、合併、修改轉移來修改 DFA）
- 評估函數：基於準確度和狀態數的損失函數
- 停止條件：DFA 評估次數達到 `max_evaluations`

---

## 預備知識

### 1. DFA 修改操作（propose_automata）

從初始 DFA 生成候選 DFA 有三種操作：
- DELETE: 刪除一個狀態
- MERGE: 合併兩個狀態
- DELTA: 修改轉移函數

這些操作由 `learner._propose_delete()`、`learner._propose_merge()`、`learner._propose_delta()` 實現。

### 2. DFA 評估（_compute_accuracy）

計算 DFA 在給定資料集上的準確度：
```python
accepts = [learner.check_path_accepted(dfa, path) for path in training_data]
accuracy = (True_Positives + True_Negatives) / len(training_data)
```

### 3. 損失函數（_compute_loss）

```python
if accuracy < threshold:
    loss = 1000 + (threshold - accuracy) × 100  # 強烈懲罰低準確度
else:
    loss = num_states  # 獎勵最小化狀態數
```

### 4. 共享初始化（SharedInit）

所有三種演算法從相同的起點開始：
- 初始 DFA（通過 RPNI 生成）
- 訓練資料（擾動採樣）
- 驗證資料（用於評估最終 DFA）

---

## SA - 模擬退火搜尋

### 標準 SA 演算法流程

```
1. 初始化：選擇初始解（初始 DFA），設置初始溫度 T = T_max
2. 重複直到 T < T_min 或達到迭代限制：
   a) 生成鄰域解（即鄰近的 DFA）
   b) 計算能量差 ΔE = Energy(新DFA) - Energy(當前DFA)
   c) 如果 ΔE < 0，接受新解
      否則，以概率 exp(-ΔE/T) 接受新解
   d) 溫度遞減: T = T × cooling_rate
3. 返回全局最優解
```

### 適配到 DFA 的改進

#### 類：DFAAnnealer

繼承自: `simanneal.Annealer`（Python 的 SA 庫）

關鍵屬性:
```python
self.current_dfa          # 當前 DFA
self.best_dfa             # 全局最優 DFA
self.evaluations_count    # DFA 評估計數（重要：停止條件）
self.max_evaluations      # 評估預算
```

#### DFA-SA 搜尋流程

```
DFAAnnealer.anneal() 主迴圈：
│
├─ 初始化（iteration=0）
│  └─ 呼叫 _get_candidates(initial_dfa, state, 0)
│     → propose_automata 初始化內部指標
│     → 返回 initial_dfa 本身
│
└─ 重複（iteration=1, 2, 3, ...）
   │
   ├─ 邊界檢查
   │  └─ if evaluations_count >= max_evaluations:
   │     → 拋出 RuntimeError（"SA budget exhausted"）停止
   │
   ├─ 生成鄰域
   │  ├─ 呼叫 _get_candidates(current_dfa, state, iteration, beam_size=1)
   │  │  → propose_automata 生成 1 個候選（隨機的 DELETE/MERGE/DELTA）
   │  │  → 返回 [candidate_dfa]
   │  │
   │  └─ evaluations_count += 1（計數增加）
   │
   ├─ 評估鄰域
   │  ├─ 計算 candidate_acc = _compute_accuracy(candidate_dfa, training_data)
   │  │  → 在 training_data 上測試 DFA
   │  │  → 返回準確度（0.0 ~ 1.0）
   │  │
   │  ├─ 記錄歷史：_add_to_history(candidate, accuracy)
   │  └─ 追蹤全局最優：if accuracy > best_acc: best_dfa = candidate
   │
   ├─ Metropolis 準則（simanneal 內部處理）
   │  ├─ energy(current_dfa) = -accuracy(current_dfa)
   │  │  → 能量 = 負準確度（SA 最小化能量）
   │  │
   │  ├─ energy(candidate_dfa) = -accuracy(candidate_dfa)
   │  │
   │  ├─ ΔE = energy(candidate) - energy(current)
   │  │     = -acc(candidate) - (-acc(current))
   │  │     = acc(current) - acc(candidate)
   │  │
   │  ├─ 如果 ΔE < 0（candidate 準確度更好）→ 接受
   │  └─ 否則 → 以概率 exp(-ΔE/T) 接受（允許暫時下降）
   │
   └─ 溫度遞減（simanneal 自動處理）
      └─ T = T × (1 - cooling_rate)

返回值：
├─ best_dfa       # 全局最優 DFA
└─ best_energy    # 對應的最小能量
```

#### DFA-SA 的適應性設計

| 純 SA 概念 | 映射到 DFA | 說明 |
|-----------|-----------|------|
| 初始解 | 初始 DFA | RPNI 生成的初始聚类結果 |
| 鄰域生成 | propose_automata | 三種修改操作（DELETE/MERGE/DELTA） |
| 能量函數 | -accuracy | 最小化能量 = 最大化準確度 |
| 溫度影響 | 接受概率 | 高溫時容易接受壞解（探索），低溫時傾向於接受好解（利用） |
| 停止條件 | max_evaluations | 當評估的 DFA 數達到預算上限 |

#### 程式碼流程範例

```python
# 初始化
annealer = DFAAnnealer(
    initial_dfa, training_data, training_labels,
    validation_data, validation_labels, state,
    output_dir, beam_size=1, max_evaluations=500
)
annealer.Tmax = 10.0
annealer.Tmin = 0.001
annealer.steps = 500  # 最多 500 次迭代（但通常會因預算用盡而提前停止）

# 執行（會在 evaluations_count >= 500 時停止）
best_dfa, best_energy = annealer.anneal()
```

---

## GA - 遺傳演算法搜尋

### 標準 GA 演算法流程

```
1. 初始化：生成初始種群（N 個個體）
2. 重複直到收斂：
   a) 評估種群中每個個體的適應度（fitness）
   b) 選擇階段：根據適應度選擇高品質個體作為親本
   c) 交叉階段：兩個親本交叉產生後代（可能結合雙方特徵）
   d) 變異階段：後代隨機變異（引入多樣性）
   e) 替換：新種群替代舊種群
3. 返回最優個體
```

### 適配到 DFA 的改進

#### GA 配置

注意: DFA 沒有良好定義的交叉操作（兩個 DFA 如何交叉？），因此使用 變異之外的演算法。

```python
population_size = 20      # 每代種群大小
tournament_size = 2       # 錦標賽選擇時對比 2 個個體
mutation_rate = 100%      # 總是變異（無交叉）
```

#### DFA-GA 搜尋流程（標準代際替換）

```
GA 主迴圈：
│
├─ 初始化
│  ├─ population = [initial_dfa]
│  ├─ 評估 initial_dfa→得到 fitness
│  ├─ all_history = [initial_dfa]
│  └─ evaluations_count = 1
│
└─ 世代迴圈（generation = 1, 2, 3, ...）
   │
   ├─ 邊界檢查
   │  └─ if evaluations_count >= max_evaluations:
   │     → 停止進化，跳出迴圈
   │
   ├─ 代際生成（生成 population_size 個後代）
   │  │
   │  └─ for individual_idx in range(population_size):
   │     │
   │     ├─ 邊界檢查（預算用盡時跳出內迴圈）
   │     │  └─ if evaluations_count >= max_evaluations:
   │     │     → break 停止生成
   │     │
   │     ├─ 選擇階段- 錦標賽選擇
   │     │  ├─ 隨機選擇 tournament_size 個個體進行對比
   │     │  └─ parent = 適應度最高的個體
   │     │
   │     ├─ 變異階段- 生成鄰域
   │     │  ├─ candidates = _get_candidates(parent_dfa, state, generation, beam_size=1)
   │     │  │  → propose_automata 生成 1 個變異候選
   │     │  │  → 返回 [mutated_dfa]
   │     │  │
   │     │  └─ evaluations_count += len(candidates)
   │     │
   │     ├─ 評估候選
   │     │  ├─ scored_candidates = _rank_candidates(candidates, training_data, ...)
   │     │  └─ best_candidate = scored_candidates[0]（最高準確度）
   │     │
   │     ├─ 記錄並創建個體
   │     │  ├─ _add_to_history(best_candidate, accuracy)
   │     │  ├─ child = DEAP 個體
   │     │  ├─ child.dfa = best_candidate.dfa
   │     │  ├─ child.fitness.values = (accuracy,)
   │     │  └─ offspring.append(child)
   │     │
   │     └─ 下一個個體
   │
   ├─ 代際替換（標準 GA 策略）
   │  └─ population[:] = offspring
   │     → 整個種群被新一代替代（不保留舊個體）
   │
   ├─ 統計和日誌
   │  ├─ best_accuracy_this_gen = max(fitness in population)
   │  ├─ avg_accuracy_this_gen = mean(fitness in population)
   │  └─ generation_stats.append({generation, best_acc, avg_acc, num_states, ...})
   │
   └─ 下一世代
      └─ generation += 1，繼續迴圈...

返回值：
└─ _select_final(all_history, select_by="accuracy", threshold, ...)
   → 從所有評估過的 DFA 中選擇最優
   → select_by="accuracy" 模式：accuracy ≥ threshold 時，返回最小狀態數的 DFA
```

#### DFA-GA 的適應性設計

| 純 GA 概念 | 映射到 DFA | 說明 |
|-----------|-----------|------|
| 個體 | DFA | 每個 DFA 是一個個體 |
| 適應度 | accuracy | 準確度越高，適應度越好 |
| 選擇 | tournament selection | 選擇種群中準確度高的 DFA 作為親本 |
| 交叉 | 不使用 | DFA 沒有清晰的交叉定義 |
| 變異 | propose_automata | DELETE/MERGE/DELTA 操作作為變異 |
| 停止條件 | max_evaluations | 評估 DFA 數達到預算 |

#### 程式碼流程範例

```python
result = ga_dfa_search(
    sampler_fn=lambda num_samples, compute_labels=True: ...,
    data_type="DFA",
    shared_init=SharedInit(...),
    population_size=20,
    tournament_size=2,
    max_evaluations=500
)
```

關鍵特點:
- 代際替換: 每代生成 population_size 個後代，然後整個替換舊種群（標準 GA 策略）
- 無交叉: 只有選擇 + 變異，保持多樣性通過變異的隨機性
- 固定種群大小: 每代始終保持 population_size 個個體
- 全局歷史: 追蹤所有評估過的 DFA（用於最終選擇）
- 預算感知: 在生成後代迴圈中檢查 evaluations_count，預算用盡時立即停止

---

## PSO - 粒子群最佳化搜尋

### 標準 PSO 演算法流程

```
1. 初始化：建立 N 個粒子（每個粒子的位置代表解的位置向量）
2. 重複直到收斂：
   a) 評估每個粒子的 fitness (accuracy)
   b) 更新「每個粒子的過去最優 pbest」
   c) 更新「全局最優粒子 gbest」
   d) 更新速度：v = w×v + c1×rand()×(pbest - x) + c2×rand()×(gbest - x)
   e) 更新位置：x = x + v
   每輪粒子會計算一個移動向量，試圖往 pbest 和 gbest 的中間地帶靠攏
3. 返回 gbest（全局最優解）
```

參數:
- `w` (慣性權重): 控制速度的動量
- `c1` (認知係數): 粒子向其過去最優靠近的傾向
- `c2` (社交係數): 粒子向全局最優靠近的傾向

### map 到 DFA 

關鍵: 粒子的位置 position 不直接表示一個 DFA，而是表示一組可能的 refinement 操作，再映射成實際的 DFA

- `particle` : (current_dfa, position, velocity, pbest, pbest_accuracy)
   - 每個粒子對應一個目前的候選 DFA，以及其在操作空間中的位置、速度與歷史最佳資訊

- `position` : (operation_type, selector, prob) 
   - 表示一組對 DFA 的修改操作參數
   - `operation_type` ∈ {DELETE, MERGE, DELTA}
   - `selector` ∈ [0, k]，從候選修改中選擇哪 k 個，其中 k 同步 beam_size
   - `prob` ∈ [0,1]，此操作被執行的機率

   position → 經過 mapping (將執行步驟套到 DFA 上) → DFA

- `velocity` : 表示修改操作偏好的變化方向
   - 公式不變：`v = wv + c1(pbest - x) + c2(gbest - x)`
   - 語義：粒子的操作偏好會逐步朝 pbest 與 gbest 所對應的修改方向靠近

- `distance` : 在一組輸入樣本上，兩個 DFA 預測不同的比例
   - 實際計算與 pbest 跟 gbest 的距離時，用 1 - accuracy 代替，避免過多的計算 


**Mapping 流程**（_map_position_to_dfa）

1. 將粒子的 position 解碼為一組 DFA refinement 操作
2. 將這些操作套用到目前 DFA，得到新的候選 DFA
3. 計算此 DFA 的 fitness
4. 根據 fitness 更新 pbest 與 gbest
5. 用 PSO 更新公式更新 velocity 與 position
6. 重複上述步驟直到停止條件成立


#### DFA-PSO 搜尋流程

```
PSOAutomataOptimizer.optimize() 主迴圈：
│
├─ 初始化
│  ├─ 評估 initial_dfa
│  ├─ gbest_dfa = initial_dfa
│  ├─ gbest_fitness = loss(initial_dfa)
│  └─ evaluations_count = 1
│
├─ 建立 PSO 最佳化器（pyswarms.GlobalBestPSO）
│  ├─ n_particles = 10（粒子數）
│  ├─ dimensions = 8（每個粒子的位置向量維度）
│  └─ options = {w: 0.7, c1: 1.5, c2: 1.5}
│
└─ PSO 迭代（iteration = 0, 1, 2, ..., n_iterations）
   │
   ├─ 粒子位置映射　for each particle_id in [0, n_particles]: 
   │  │
   │  ├─ 限制檢查
   │  │  └─ if evaluations_count >= max_evaluations:
   │  │     → 拋出 RuntimeError 停止最佳化
   │  │
   │  ├─ 位置→操作映射　_map_position_to_dfa(particle_id, position) 
   │  │  │
   │  │  ├─ 對所有 `prob` 做 softmax，取得修改操作被執行的機率並依序排序
   │  │  │ 
   │  │  ├─ 依序挑出前 k 個槽位： 
   │  │  │  │  
   │  │  │  ├─ `operation_type` → {DELETE, MERGE, DELTA}
   │  │  │  ├─ `target_state` → 選取 top-k 候選（k 預設同步 beam_size）
   │  │  │  ├─ 執行該修改並更新 current_dfa
   │  │  │  │ 
   │  │  │  └─ 下一個修改操作會在最新 DFA 上重新取樣候選 
   │  │  │ 
   │  │  └─ 返回 modified_dfa（累積執行後的結果） 
   │  │
   │  ├─ 清理　remove_unreachable_states(dfa)
   │  │  → 刪除無法從初始狀態到達的狀態
   │  │
   │  ├─ 評估　_evaluate_and_cache(dfa)
   │  │  ├─ accuracy = _compute_accuracy(dfa, training_data)
   │  │  ├─ loss = _compute_loss(accuracy, num_states)
   │  │  ├─ val_acc = _compute_accuracy(dfa, validation_data)
   │  │  └─ evaluations_count += 1
   │  │
   │  ├─ 追蹤
   │  │  ├─ _add_to_history(dfa, accuracy, val_acc)
   │  │  └─ losses[particle_id] = loss
   │  │
   │  └─ 更新 gbest
   │     └─ if loss < gbest_fitness: gbest_dfa = dfa, ...
   │
   ├─ PSO 更新（pyswarms 內部處理）
   │  ├─ velocity[i] = w×velocity[i] 
   │  │            + c1×rand()×(pbest[i] - position[i])
   │  │            + c2×rand()×(gbest - position[i])
   │  │
   │  └─ position[i] += velocity[i]
   │
   ├─ 進度日誌
   │  └─ print(f"Iteration {iteration}: best loss = {min(losses)}")
   │
   └─ 下一個迭代或停止
      └─ 如果 iteration >= n_iterations 或預算用盡 → 停止

返回值：
├─ best_dfa                # 全局最優 DFA
├─ best_accuracy           # 對應的準確度
├─ best_states             # 狀態數
├─ best_loss               # 最小損失值
├─ all_history             # 所有評估過的 DFA
└─ evaluations            # 實際評估次數
```

#### DFA-PSO 的適應性設計

| 純 PSO 概念 | 映射到 DFA | 說明 |
|-----------|-----------|------|
| 位置向量 | 操作序列編碼 | 8 維浮點向量→3 步修改操作 |
| 適應度 | loss 值 | 損失函數結合準確度和狀態數 |
| 速度更新 | PSO 速度方程 | 粒子向 pbest 和 gbest 靠近 |
| 個人最優 pbest | 該粒子見過的最好 DFA | 追蹤每個粒子的歷史最優 |
| 全局最優 gbest | all particles 中最好的 DFA | 種群中的最優解 |
| 停止條件 | max_evaluations | 評估 DFA 數達到預算 |

#### 程式碼流程範例

```python
optimizer = PSOAutomataOptimizer(
    initial_dfa=initial_dfa,
    threshold=0.9,
    data=training_data,
    labels=training_labels,
    validation_data=validation_data,
    validation_labels=validation_labels,
    learner=learner,
    n_particles=10,
    n_iterations=30,  # 計算為 max_evaluations // n_particles
    w=0.7,
    c1=1.5,
    c2=1.5,
    verbose=True,
    max_evaluations=500
)

result = optimizer.optimize(n_particles=10, n_iterations=30, save_trajectory=True)
```

關鍵特點:
- 離散化映射: 連續位置向量→離散 DFA 操作
- 預算感知: 在最佳化目標函數中檢查預算
- 狀態清理: 每次生成新 DFA 後移除無法到達的狀態
- 歷史追蹤: 全局歷史用於最終選擇

---

## 三者對比

### 搜尋策略對比

| 特性 | SA | GA | PSO |
|------|----|----|-----|
| 搜尋風格 | 貪心+隨機跳躍 | 種群進化 | 種群合作 |
| 中心資料結構 | 單一當前解 | 種群 | 種群+速度向量 |
| 鄰域探索 | 逐步生成單個鄰近解 | 通過變異生成多個候選 | 通過位置更新生成多個候選 |
| 傳播機制 | 概率接受（Metropolis） | 選擇+變異 | 速度更新（pbest+gbest） |
| 多樣性維持 | 高溫時隨機接受差解 | 變異操作的隨機性 | 速度向量的隨機分量 |
| 收斂特性 | 逐步冷卻→逐漸收斂 | 逐代優化→可能陷入局部最優 | 粒子聚集→快速收斂 |

### DFA 適配對比

| 方面 | SA | GA | PSO |
|-------------|-------------|-----------------|----------|
| DFA 修改方式 | 單個→單個（逐步修改） | 種群→後代種群（代際替換） | 連續向量→DFA（映射） |
| 候選生成 | beam_size=1 | 每代生成 population_size 個後代 | 根據位置向量生成 |
| 全局最優追蹤 | best_dfa（顯式） | all_history + best_in_population（隱式）| gbest_dfa（顯式） |
| 適應度評估 | Energy：-accuracy | Fitness：accuracy | Loss：基於準確度+狀態數 |
| 停止機制 | 溫度+預算 | 預算 | 迭代+預算 |

### 效能特徵

| 方面 | SA | GA | PSO |
|------|----|----|-----|
| 計算複雜度 | O(評估數) | O(種群×代數) | O(粒子數×迭代數) |
| 記憶體使用 | 低（單個 DFA） | 中等（種群+歷史） | 中等（種群+速度+歷史） |
| 逃離局部最優 | ✓（概率跳躍） | ✓（變異多樣性） | ✓（速度動量） |
| 求解時間 | 快 | 中等 | 中等 |
| 實現難度 | 低 | 中 | 高 |
| 參數敏感度 | 中（T衰減速率） | 中（種群大小） | 高（w,c1,c2） |

### 預算用盡停止對比

共同目標: 當 DFA 評估次數 ≥ `max_evaluations` 時停止

```
SA:  每次 move() 前檢查 evaluations_count
     → evaluations_count >= max_evaluations 時拋異常停止

GA:  內迴圈 while 條件判斷 evaluations_count < max_evaluations
     → 預算用盡時自然停止生成新候選

PSO: 目標函數開始時檢查 evaluations_count
     → evaluations_count >= max_evaluations 時拋異常停止
     → 由 pyswarms 庫捕獲並中止最佳化
```

---

## 實驗建議

### 何時使用哪種演算法？

1. SA 適合:
   - 問題空間平坦（許多次優解選項）
   - 需要快速初步最佳化
   - 計算資源受限

2. GA 適合:
   - 需要維持高種群多樣性
   - 問題具有 plateau（平台）區域
   - 可以並行評估種群

3. PSO 適合:
   - 需要快速收斂
   - 有明確的全局最優信號
   - 願意投入時間調參（w, c1, c2）

### 參數建議

SA:
```python
steps = 500         # 最多 500 次迭代
T_max = 10.0        # 初始溫度
T_min = 0.001       # 停止溫度
max_evaluations = 500
```

GA:
```python
population_size = 20
tournament_size = 2
max_evaluations = 500
```

PSO:
```python
n_particles = 10
n_iterations = 50        # 計算為 max_evaluations // n_particles
w = 0.7                  # 慣性
c1 = 1.5                 # 認知係數
c2 = 1.5                 # 社交係數
max_evaluations = 500
```

---

## 總結

- SA、GA、PSO 都被適配成基於 DFA 修改操作的啟發式搜尋
- 關鍵適配點:
  1. 鄰域生成 → `propose_automata`（DELETE/MERGE/DELTA）
  2. 適應度評估 → `_compute_accuracy`（訓練準確度）
  3. 停止條件 → `evaluations_count >= max_evaluations`（DFA 評估數預算）
- 三者權衡:
  - SA：簡單快速
  - GA：多樣性好
  - PSO：收斂快
- 通用流程: 初始化 → 生成候選 → 評估 → 歷史追蹤 → 預算檢查 → 最終選擇
