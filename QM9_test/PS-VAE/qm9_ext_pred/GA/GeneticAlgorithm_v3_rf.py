"""
遗传算法迭代有效性验证思路（回归模型 + 动态目标 + 物理约束）：
1. 数据准备：将溶剂配比特征归一化。
2. 训练回归模型：使用随机森林预测freeze值。
3. 动态目标：每代设定目标值（从0.9线性降至0），个体适应度 = |预测freeze - 目标| + 物理约束惩罚。
4. 遗传算法：
   - 初始种群从freeze>0.9组随机选取。
   - 迭代200代，种群大小2000。
   - 每代记录种群平均适应度。
5. 结果分析：绘制平均适应度随代数变化曲线，观察是否呈下降趋势。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import random
from sklearn.ensemble import RandomForestRegressor

# ====================== 参数设置 ======================
POP_SIZE = 500  # 种群大小
N_GEN = 2000  # 迭代代数
EPS = 1e-6  # 防止除零的小常数
CROSS_PROB = 0.1  # 交叉概率
MUT_PROB = 0.05  # 变异概率（每个基因的变异概率）
ELITE_SIZE = 5  # 每代保留的最优个体数量（精英数）
MUT_ETA = 20  # 多项式变异的分布指数（越大变异幅度越小）
ORIGINAL_GROUP = 0.95  # 初始种群选取阈值（freeze > 0.9）
VERSION = "0_1"
# 早停设置
PATIENCE = N_GEN / 10  # 容忍连续多少代适应度不改善
best_fitness_so_far = float("inf")  # 记录历史最佳适应度
no_improve_count = 0  # 未改善计数
# ====================== 数据加载与预处理 ======================
df = pd.read_csv("onehot-filtered.csv")

# 溶剂列（根据示例顺序）
solvent_cols = [
    "meoh",
    "dmso",
    "gly",
    "dme",
    "dmac",
    "dox",
    "nmp",
    "dmf",
    "tetraglyme",
    "thf",
    "gbl",
    "diglyme",
]
X_raw = df[solvent_cols].values.astype(float)
y = df["freeze"].values.astype(float)

# 对特征进行Min-Max归一化，使每个特征在[0,1]区间
X_min = X_raw.min(axis=0)
X_max = X_raw.max(axis=0)
X_norm = (X_raw - X_min) / (X_max - X_min + EPS)

# 归一化后的边界（用于遗传算法个体约束）
LB = np.zeros(X_norm.shape[1])  # 下界 0
UB = np.ones(X_norm.shape[1])  # 上界 1

# ====================== 训练回归模型 ======================
print("训练随机森林回归模型...")
model = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42)
model.fit(X_norm, y)
print("模型训练完成")


# ====================== 物理约束惩罚函数 ======================
def calculate_penalty(individual_norm):
    """
    计算物理约束惩罚项
    输入：归一化个体（溶剂比例）
    返回：惩罚值（标量）
    """
    # 反归一化到原始溶剂比例
    ind_raw = individual_norm * (X_max - X_min) + X_min
    solvents = ind_raw  # 个体只包含溶剂比例

    penalty = 0.0

    # 约束1：溶剂总和应接近100（允许±5%误差）
    total = np.sum(solvents)
    if abs(total - 100) > 5:
        penalty += abs(total - 100) * 0.1

    # 约束2：有效溶剂数量（占比>5%）应≥5
    active = np.sum(solvents > 5)
    if active < 5:
        penalty += (5 - active) * 0.2

    # 约束3：构型熵应≥1.0
    valid = solvents[solvents > 0]
    if len(valid) > 0:
        probs = valid / np.sum(valid)
        entropy = -np.sum(probs * np.log(probs))
        if entropy < 1.0:
            penalty += (1.0 - entropy) * 0.3
    else:
        penalty += 1.0  # 无有效溶剂，重罚

    return penalty


# ====================== 适应度函数（含动态目标） ======================
def fitness_with_model(individual, target):
    """
    计算个体适应度 = |预测freeze - 目标| + 惩罚
    """
    pred = model.predict([individual])[0]
    penalty = calculate_penalty(individual)
    return abs(pred - target)  # + penalty


# ====================== 遗传操作函数 ======================
def tournament_selection(pop, fitness, tourn_size=2):
    """锦标赛选择（最小化适应度）"""
    indices = np.random.choice(len(pop), tourn_size, replace=False)
    best_idx = indices[np.argmin(fitness[indices])]
    return pop[best_idx].copy()


def arithmetic_crossover(p1, p2):
    """算术交叉：生成两个子代，并裁剪到边界"""
    alpha = np.random.random()
    c1 = alpha * p1 + (1 - alpha) * p2
    c2 = (1 - alpha) * p1 + alpha * p2
    c1 = np.clip(c1, LB, UB)
    c2 = np.clip(c2, LB, UB)
    return c1, c2


def polynomial_mutation(individual, prob, eta, low, up):
    """
    多项式变异（Polynomial Mutation）
    """
    mutated = individual.copy()
    n = len(mutated)
    for i in range(n):
        if np.random.random() < prob:
            r = np.random.random()
            if r < 0.5:
                delta = (2 * r) ** (1.0 / (eta + 1)) - 1.0
            else:
                delta = 1.0 - (2 * (1.0 - r)) ** (1.0 / (eta + 1))
            mutated[i] += delta * (up[i] - low[i])
            mutated[i] = np.clip(mutated[i], low[i], up[i])
    return mutated


# ====================== 初始化种群（从freeze>0.9组选取） ======================
group_mask = (y - ORIGINAL_GROUP) > EPS
X_group = X_norm[group_mask]

if len(X_group) == 0:
    raise ValueError("数据中没有freeze>0.9的样本，请检查数据")
else:
    print("初始种群候选个数：", len(X_group))

# 有放回随机选择POP_SIZE个个体作为初始种群
population = X_group[np.random.choice(len(X_group), POP_SIZE, replace=True)]


# ====================== 主遗传算法循环 ======================


avg_fitness_history = []  # 记录每代平均适应度

for gen in range(N_GEN):
    # 动态目标：从0.9线性降至0
    target = max(0, 0.9 * (1 - gen / N_GEN))

    # 评估当前种群适应度
    fitness = np.array([fitness_with_model(ind, target) for ind in population])
    avg_fitness = np.mean(fitness)
    avg_fitness_history.append(avg_fitness)

    # 检查是否达到新的最佳适应度（由于我们是最小化）
    if avg_fitness < best_fitness_so_far - 1e-6:  # 有显著改善
        best_fitness_so_far = avg_fitness
        no_improve_count = 0
    else:
        no_improve_count += 1

    # 如果连续 PATIENCE 代没有改善，提前停止
    if no_improve_count >= PATIENCE:
        print(f"\n适应度已连续 {PATIENCE} 代未改善，在第 {gen+1} 代提前停止迭代")
        break

    # 精英保留
    elite_indices = np.argsort(fitness)[:ELITE_SIZE]
    elites = [population[idx].copy() for idx in elite_indices]

    # 生成下一代剩余个体
    offspring = []
    while len(offspring) < POP_SIZE - ELITE_SIZE:
        parent1 = tournament_selection(population, fitness, tourn_size=2)
        parent2 = tournament_selection(population, fitness, tourn_size=2)

        if np.random.random() < CROSS_PROB:
            child1, child2 = arithmetic_crossover(parent1, parent2)
        else:
            child1, child2 = parent1.copy(), parent2.copy()

        child1 = polynomial_mutation(child1, prob=MUT_PROB, eta=MUT_ETA, low=LB, up=UB)
        child2 = polynomial_mutation(child2, prob=MUT_PROB, eta=MUT_ETA, low=LB, up=UB)

        offspring.append(child1)
        if len(offspring) < POP_SIZE - ELITE_SIZE:
            offspring.append(child2)

    population = np.array(elites + offspring)

    # if (gen + 1) % 20 == 0:
    print(f"第 {gen+1:3d} 代，目标={target:.3f}，平均适应度={avg_fitness:.4f}")

# ====================== 结果可视化 ======================
# plt.figure(figsize=(10, 6))
# plt.plot(range(1, len(avg_fitness_history) + 1), avg_fitness_history, "b-", linewidth=2)
# plt.xlabel("Generation", fontsize=12)
# plt.ylabel("Average Fitness (|pred - target| )", fontsize=12)
# plt.title("Evolution of Population Average Fitness with Dynamic Target", fontsize=14)
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.savefig(
#     f"avg_fitness_history_v3_{str(ORIGINAL_GROUP).replace('.','-')}.png", dpi=300
# )
# plt.show()
# ====================== 结果可视化 ======================
plt.figure(figsize=(12, 8))  # 稍微增大图片尺寸以容纳参数文本

# 绘制主曲线
plt.plot(
    range(1, len(avg_fitness_history) + 1),
    avg_fitness_history,
    "b-",
    linewidth=2,
    label="Average Fitness",
)

# 添加参数信息文本
params_text = (
    f"Parameters:\n"
    f"POP_SIZE = {POP_SIZE}\n"
    f"N_GEN = {N_GEN}\n"
    f"CROSS_PROB = {CROSS_PROB}\n"
    f"MUT_PROB = {MUT_PROB}\n"
    f"ELITE_SIZE = {ELITE_SIZE}\n"
    f"MUT_ETA = {MUT_ETA}\n"
    f"ORIGINAL_GROUP = {ORIGINAL_GROUP}"
)

# 在图的左上角添加文本框
plt.text(
    0.02,
    0.98,
    params_text,
    transform=plt.gca().transAxes,
    fontsize=10,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
)

plt.xlabel("Generation", fontsize=12)
plt.ylabel("Average Fitness (|pred - target|)", fontsize=12)
plt.title("Evolution of Population Average Fitness with Dynamic Target", fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig(
    f"avg_fitness_history_v3_{str(ORIGINAL_GROUP).replace('.','-')}_{VERSION}.png",
    dpi=300,
)
plt.show()
# ====================== 保存历史数据 ======================
np.savetxt(
    f"avg_fitness_history_v3_{str(ORIGINAL_GROUP).replace('.','-')}_{VERSION}.txt",
    avg_fitness_history,
)


# ====================== 保存最后一代种群 ======================
def denormalize(individual):
    ind_array = np.array(individual)
    return ind_array * (X_max - X_min) + X_min


last_pop = population
last_individuals_denorm = []
last_fitness = []

for ind in last_pop:
    ind_array = np.array(ind)
    fit_val = fitness_with_model(ind_array, 0.0)  # 最后一代目标为0
    last_fitness.append(fit_val)
    denorm_ind = denormalize(ind_array)
    last_individuals_denorm.append(denorm_ind)

pop_df = pd.DataFrame(last_individuals_denorm, columns=solvent_cols)
pop_df["fitness_freeze"] = last_fitness
pop_df["generation"] = N_GEN
pop_df.to_csv(
    f"final_population_{str(ORIGINAL_GROUP).replace('.','-')}_{VERSION}.csv",
    index=False,
)
print(f"最后一代种群已保存至 final_population.csv，共 {len(pop_df)} 个个体")
