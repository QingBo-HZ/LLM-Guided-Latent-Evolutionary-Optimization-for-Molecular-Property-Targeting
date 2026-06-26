import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# =====================
# 你需要替换的输入
# =====================

# 三种初始化 + 最终结果（latent）
psvae_init = np.load("psvae_init_latent.npy")
psvae_final = np.load("psvae_final_latent.npy")

llm_init = np.load("llm_init_latent.npy")
llm_final = np.load("llm_final_latent.npy")

hybrid_init = np.load("hybrid_init_latent.npy")
hybrid_final = np.load("hybrid_final_latent.npy")

# predictor（你自己的）
predictor = ...   # 必须是 predictor(z) -> gap

# =====================
# Step1: PCA降维
# =====================

all_latent = np.concatenate([
    psvae_init, psvae_final,
    llm_init, llm_final,
    hybrid_init, hybrid_final
], axis=0)

pca = PCA(n_components=2)
pca.fit(all_latent)

def project(z):
    return pca.transform(z)

psvae_init_2d = project(psvae_init)
psvae_final_2d = project(psvae_final)

llm_init_2d = project(llm_init)
llm_final_2d = project(llm_final)

hybrid_init_2d = project(hybrid_init)
hybrid_final_2d = project(hybrid_final)

# =====================
# Step2: 构建等高线网格
# =====================

x_min, x_max = all_latent[:,0].min(), all_latent[:,0].max()
y_min, y_max = all_latent[:,1].min(), all_latent[:,1].max()

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 100),
    np.linspace(y_min, y_max, 100)
)

grid_points = np.stack([xx.ravel(), yy.ravel()], axis=1)

# 逆投影回 latent 空间
latent_grid = pca.inverse_transform(grid_points)

# 预测 gap
pred = predictor(latent_grid)
gap = pred[:, predictor.gap_idx]

# 👉 sigmoid score（你现在用的）
def gap_to_score(g):
    return 1 / (1 + np.exp((g - 0.1) / 0.02))

score = gap_to_score(gap)
Z = score.reshape(xx.shape)

# =====================
# Step3: 绘图
# =====================

plt.figure(figsize=(8, 6))

# 等高线
contour = plt.contourf(xx, yy, Z, levels=30)
plt.colorbar(contour, label="Gap Score")

# ===== 轨迹（中心点）=====
def plot_traj(init_2d, final_2d, color, label):
    init_center = init_2d.mean(axis=0)
    final_center = final_2d.mean(axis=0)

    plt.scatter(init_center[0], init_center[1], c=color, marker='o', s=80)
    plt.scatter(final_center[0], final_center[1], c=color, marker='*', s=150)

    plt.arrow(
        init_center[0], init_center[1],
        final_center[0] - init_center[0],
        final_center[1] - init_center[1],
        color=color, width=0.01, length_includes_head=True
    )

    plt.plot([], [], color=color, label=label)

plot_traj(psvae_init_2d, psvae_final_2d, 'blue', 'PS-VAE')
plot_traj(llm_init_2d, llm_final_2d, 'green', 'LLM')
plot_traj(hybrid_init_2d, hybrid_final_2d, 'red', 'Hybrid')

# ===== 最终点云（看diversity）=====
plt.scatter(hybrid_final_2d[:,0], hybrid_final_2d[:,1],
            c='red', alpha=0.2, s=10)

# =====================
# 图设置
# =====================

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Latent Space Optimization Landscape (Gap Score)")

plt.legend()
plt.tight_layout()

plt.savefig("contour_hybrid_vs_baseline.png", dpi=300)
plt.show()