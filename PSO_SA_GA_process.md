# PSO、SA、GA 在 DFA 搜尋中的流程說明

## 目錄
1. [核心思想](#核心思想)
2. [預備知識](#預備知識)
3. [SA - 模擬退火搜尋](#sa--模擬退火搜尋)
4. [GA - 遺傳演算法搜尋](#ga--遺傳演算法搜尋)
5. [PSO - 粒子群最佳化搜尋](#pso--粒子群最佳化搜尋)
6. [三者對比](#三者對比)

---


## 核心思想
- 目標：在固定 evaluation budget 下，比較不同搜尋法是否能在滿足 training accuracy threshold 的前提下，找到更小且具有較高 validation accuracy 的 DFA。
- 搜尋空間：DFA 狀態空間（通過刪除、合併、修改轉移來修改 DFA）。
- 訓練準確度 : 引導搜尋
- 驗證準確度 : 觀察最終自動機與初始自動機的行為是否有偏差

---

## 預備知識

### 1. DFA 修改操作

- DELETE: 刪除一個狀態
- MERGE: 合併兩個狀態
- DELTA: 修改轉移


### 2. DFA 評估

計算 DFA 在給定資料集上的準確度：當預測與初始實例一樣 -> 自動機須接受，否則拒絕
```python
accepts = [learner.check_path_accepted(dfa, path) for path in training_data]
accuracy = (True_Positives + True_Negatives) / len(training_data)
```

### 3. 共享初始化（SharedInit）

所有演算法從相同的起點開始：
- 初始 DFA（通過 RPNI 生成）
- 驗證資料（用於評估最終 DFA）
- SA、GA、PSO 的訓練資料 (呼叫前擾動一批，訓練過程皆用此引導搜尋)
- `max_evaluations` (beam 評估 DFA 的次數)

## Beam Search (Anchor beam)

流程：
- 初始化：使用 RPNI 建構 `initial_dfa` 並計算初始訓練/驗證準確度
- 迭代脈絡：每次 beam 迭代呼叫 `propose_automata`，以 KL-LUCB 排序並保留 top-k。
- 停止條件：beam search 在 state 剩下 2 時，自動停止

負責產生供 SA/GA/PSO 共用的 `SharedInit`，，確保公平比較


# 為何用 「Beam Search 評估 DFA 的次數」當作 GA/SA/PSO 的停止條件

- 在這個任務中，最昂貴的成本是「評估一個 DFA」（計算準確度）。
- SA、GA、PSO 每一輪可產生與評估的候選數不同：
   - SA : 每輪產生一個鄰近候選。
   - GA : 族群式，每代會評估多個個體。
   - PSO : 多粒子並行。
- 如果只用「相同迭代次數」作停止條件，GA/PSO 往往會比 SA 評估更多 DFA，造成比較偏差。
- 因此統一用 `max_evaluations`（Beam Search 實際使用的 DFA 評估數）可讓各方法在近似相同的計算成本下比較效果。

---

## SA - 模擬退火搜尋

### 標準 SA 演算法流程

```
1. 初始化：選擇初始解（初始 DFA），設置初始溫度 T = T_max
2. 重複直到 T < T_min 或達到迭代限制：
   a) 生成鄰域解（即鄰近的 DFA）
   b) 計算能量差 ΔE = Energy(新DFA) - Energy(當前DFA)
      Energy 計算 : -training_accuracy，Negative because Annealer minimizes energy
   c) 如果 ΔE < 0，接受新解
      否則，以概率 exp(-ΔE/T) 接受新解
   d) 溫度遞減: T = T × cooling_rate
3. 返回全局最優解
```

### 應用到 DFA

#### SA 配置
```python
annealer.Tmax = 10.0
annealer.Tmin = 0.001
# ensure enough total moves to fully consume evaluation budget
annealer.steps = max(int(steps), int(max_evaluations) + 1) 

state / solution = DFA
neighbor = 一次 DELETE / MERGE / DELTA
energy = -train_accuracy + current size/initial size  # Negative because Annealer minimizes energy
stopping = max_evaluations
```

#### DFA-SA 搜尋流程

```
DFAAnnealer.anneal() 主迴圈：
│
├─ 初始化（iteration=0）
│  └─ 呼叫 _get_candidates(initial_dfa, state, 0)
│     → 返回 initial_dfa 本身
│
└─ 重複（iteration=1, 2, 3, ... , max_evaluations）
   │
   ├─ 預算檢查
   │  └─ if evaluations_count >= max_evaluations:
   │     → break
   │
   ├─ 隨機生成 1 個鄰域
   │  │
   │  ├─ 隨機選擇操作 ['DELETE', 'MERGE', 'DELTA']
   │  │
   │  ├─ 隨機選擇當前 DFA 中要應用的狀態
   │  │
   │  └─ 生成 1 個新候選
   │     evaluations_count += 1
   │
   ├─ 評估鄰域
   │  └─ 計算 candidate_acc = _compute_accuracy(candidate_dfa, training_data)
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
└─ _select_final(all_history, select_by="accuracy", threshold, ...)
   → 從所有評估過的 DFA 中選擇最優
   → select_by="accuracy" 模式：accuracy ≥ threshold 時，返回最小狀態數的 DFA
```


---

## GA - 遺傳演算法搜尋

### 標準 GA 演算法流程

```
1. 初始化：生成初始種群（N 個個體）
2. 重複直到收斂：
   a) 評估種群中每個個體的適應度（fitness）
      fitness 計算 : training_accuracy
   b) 選擇階段：根據適應度選擇高品質個體作為 parents
   c) 交叉階段：兩個 parents 交叉產生後代（結合雙方特徵）(DFA 不做這步驟)
   d) 變異階段：後代隨機變異
   e) 替換：新種群替代舊種群
3. 返回最優個體
```

### 應用到 DFA

#### GA 配置

注意: DFA 沒有良好定義的交叉操作（兩個 DFA 如何交叉？），因此只使用變異操作。

```python
population_size = 20      # 每代種群大小
tournament_size = 2       # 錦標賽選擇時對比 2 個個體
mutation_rate = 100%      # 總是變異（無交叉）
elite_size = population_size // 10  # 保留前 10%

individual = DFA
selection = tournament selection
variation = mutation-only（沒有 crossover）
fitness = train_accuracy - current size/initial size
stopping = max_evaluations
```


#### DFA-GA 搜尋流程

```
GA 主迴圈：
│
├─ 初始化
│  ├─ population = 通過單一鄰居變異修改 initial_dfa，得到 population_size 個初始個體
│  ├─ 評估所有初始個體 → 得到 fitness
│  └─ evaluations_count = population_size
│
└─ 世代迴圈（generation = 1, 2, 3, ...）
   │
   ├─ 預算檢查
   │  └─ if evaluations_count >= max_evaluations:
   │     → break
   │
   ├─ 計算本代生成個數
   │  └─ offspring_target = min(population_size, max_evaluations - evaluations_count)
   │
   ├─ 一次性選擇父代（tournament selection）
   │  ├─ 選出 offspring_target 個父代
   │  └─ 每次選擇：隨機選 tournament_size (=2) 個個體，取 fitness 最高者
   │
   ├─ 批量變異與評估
   │  └─ for each parent in selected_parents:
   │     ├─ 生成 1 個候選（隨機 DELETE/MERGE/DELTA） 
   │     └─ 評估候選 → fitness = train_accuracy
   │        evaluations_count += 1
   │
   ├─ 精英保留
   │  ├─ elite_size = max(1, population_size // 10)  ← 保留前 10%
   │  └─ elite = sorted(population, ...)[:elite_size]  ← 當前世代最優個體
   │
   ├─ 代際替換（包含精英）
   │  ├─ population[:] = offspring + elite
   │  │  → 新後代 + 上一代精英，保留最優解
   │  │
   │  └─ if len(population) > population_size:
   │     → 按 fitness 排序，保留前 population_size 個
   │
   └─ 下一世代
      └─ generation += 1，繼續迴圈...

返回值：
└─ _select_final(all_history, select_by="accuracy", threshold, ...)
   → 從所有評估過的 DFA 中選擇最優
   → select_by="accuracy" 模式：accuracy ≥ threshold 時，返回最小狀態數的 DFA
```


---

## PSO - 粒子群最佳化搜尋

### 標準 PSO 演算法流程

```
1. 初始化：建立 N 個粒子（每個粒子的位置代表一個解的向量，速度 設為隨機）
2. 重複直到收斂：
   針對這 N 個粒子中的每一個粒子，分別執行：
   a) 評估每個粒子的 fitness (accuracy)
   b) 更新「每個粒子的過去最優 pbest」
   c) 更新「全局最優粒子 gbest」
   d) 更新速度：v = w×v + c1×rand()×(pbest - x) + c2×rand()×(gbest - x) (標準公式)
   e) 更新位置：x = x + v
   每個粒子會計算一個移動向量，試圖往 pbest 和 gbest 的中間地帶靠攏，得到新的一個解
3. 返回 gbest（全局最優解）
```

### 應用到 DFA

#### PSO 配置

```python
n_particles=10,   # 粒子數
n_iterations = max_evaluations // n_particles
w=0.7,            # 慣性權重: 控制速度的動量
c1=1.5,           # 認知係數: 粒子向其過去最優靠近的傾向
c2=1.5,           # 社交係數: 粒子向全局最優靠近的傾向
objective = train_accuracy - current size/initial size
stopping = matched evaluation budget
```

#### DFA edit-space encoding and decoding

**核心概念：** 粒子不是直接存 DFA，而是存一個 position 向量，所以用「修改方式編碼」代表一個 DFA

**編碼例子：** 把數字變回「操作序列」，然後套用到 DFA 上
```
有 position 向量 [0.1, 0.3, 0.0] 時：

1. 第一個數字 → 決定操作类型
   - 0 ~ 0.33 → DELETE
   - 0.34 ~ 0.66 → MERGE  
   - 0.67 ~ 1.0 → DELTA  
2. 第二、三個數字 → 決定欲操作的狀態
   - target_idx = int(normalized * n_states)  
4. 將操作套在初始 dfa 上
5. 得到 DFA
6. 計算 DFA 的 fitness（訓練準確度）
7. 這個 fitness 會被用來：
   - 判斷「這個 position 向量好不好」
   - 更新 pbest_position（個人最好位置）
   - 更新 gbest_position（全局最好位置）
        
8. PSO 根據 pbest 和 gbest 移動粒子
   - 粒子 "往" pbest 靠攏：獲得新的 position 向量
   - 粒子 "往" gbest 靠攏：又獲得新的 position 向量
   - 這些新 position 向量 → 解碼 → 得到新 DFA → 評估 → 重複...
```

**數據結構：**

- `particle` : (position, velocity, pbest_position, gbest_position)
   - **position** 為「修改方式編碼」
                  同一個 position 向量 → 解碼後得到同一個 DFA
   - **pbest_position** 是粒子歷史見過最好的那個 position 向量
   - **gbest_position** 是所有粒子見過最好的那個 position 向量

- `position` : 3 維浮點向量 = [op_type, op_target1, op_target2]
   - `operation_type` ∈ {0~0.33, 0.34~0.66, 0.67~1.0} → {DELETE, MERGE, DELTA}
   - `target_idx` ∈ [0, 1] → 狀態編號

- `velocity` : 3 維速度向量
   - 公式：`v = w×v + c1×rand()×(pbest_position - x) + c2×rand()×(gbest_position - x)`
   - 語義：粒子會朝著歷史最好位置和全局最好位置移動
   - 結果：新的 position = 舊 position + velocity

- `搜尋機制` : 在位置空間中移動
   - 粒子的 position 改變 → 解碼後得到不同的 DFA
   - pbest/gbest 記住最好的位置向量 → 引導粒子往最好的地方靠攏


#### DFA-PSO 搜尋流程

```
PSOAutomataOptimizer.optimize() 主迴圈：
│
├─ 初始化
│  ├─ 評估 initial_dfa
│  ├─ gbest_position = None（初始化）
│  ├─ gbest_fitness = loss(initial_dfa)
│  ├─ evaluations_count = 1
│  └─ n_iterations = max_evaluations // n_particles
│
│
├─ 建立 PSO 最佳化器（pyswarms.GlobalBestPSO）
│  ├─ n_particles = 10（粒子數）
│  ├─ dimensions = 3（每個粒子的位置向量維度：1 slot × 3 dimensions）
│  └─ options = {w: 0.7, c1: 1.5, c2: 1.5}
│
└─ PSO 迭代（iteration = 0, 1, 2, ..., n_iterations）
   │
   ├─ 粒子位置映射　for each particle_id in [0, n_particles]: 
   │  │
   │  ├─ 預算檢查
   │  │  └─ if evaluations_count >= max_evaluations:
   │  │     → break
   │  │
   │  ├─ 位置→DFA 映射 _map_position_to_dfa(position) 
   │  │  │
   │  │  ├─ 始終從 initial_dfa 開始
   │  │  │ 
   │  │  ├─ 提取 1 個槽位（3 維）並直接解碼
   │  │  │  ├─ `operation_type` → {DELETE, MERGE, DELTA}
   │  │  │  ├─ DELETE: 用 target1_idx 選擇要刪除的狀態
   │  │  │  ├─ MERGE: 用 target1_idx 和 target2_idx 獨立選擇兩個要合併的狀態
   │  │  │  ├─ DELTA: 用 target1_idx 為源狀態，target2_idx 為目標狀態
   │  │  │ 
   │  │  └─ 返回 modified_dfa（套用 1 個操作後的結果）
   │  │     同位置→同 DFA（無緩存，每次確定）
   │  │
   │  ├─ 評估 dfa
   │  │  ├─ accuracy = _compute_accuracy(dfa, training_data)
   │  │  ├─ loss = _compute_loss(accuracy)
   │  │  └─ evaluations_count += 1
   │  │
   │  └─ 更新 pbest/gbest
   │     ├─ pbest_position：該粒子見過的最好位置
   │     └─ gbest_position：所有粒子見過的最好位置
   │
   ├─ PSO 更新（pyswarms 內部處理
   │  ├─ velocity[i] = w×velocity[i] 
   │  │            + c1×rand()×(pbest_position[i] - position[i])
   │  │            + c2×rand()×(gbest_position - position[i])
   │  │
   │  └─ position[i] += velocity[i]
   │
   └─ 下一個迭代或停止
      └─ 如果 iteration >= n_iterations 或預算用盡 → 停止

返回值：
└─ _select_final(all_history, select_by="accuracy", threshold, ...)
   → 從所有評估過的 DFA 中選擇最優
   → select_by="accuracy" 模式：accuracy ≥ threshold 時，返回最小狀態數的 DFA
```


# 排名準則

在相同 evaluation budget 下

作法一、
1. 先看 validation accuracy 是否達標
2. 在達標的方法裡，比 states
3. 再看 time

---