import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score

# ===========================================================
#                   НАСТРОЙКА СРЕДЫ
# ===========================================================
os.makedirs("figures", exist_ok=True)
plt.style.use("seaborn-v0_8-whitegrid")

# ===========================================================
#                   ЗАГРУЗКА ДАННЫХ
# ===========================================================
df = pd.read_excel("Pumpkin_Seeds_Dataset.xlsx")
df.columns = [c.strip().replace(" ", "_").replace("-", "_") for c in df.columns]

# бинаризация таргета
THRESH = 0.9903
df["target_bin"] = (df["Solidity"] <= THRESH).astype(int)

# два количественных признака
f1 = "Area"
f2 = "Perimeter"

# самый частый класс
most_common_class = df["target_bin"].mode()[0]
df_class = df[df["target_bin"] == most_common_class]

# признаки
x1 = df_class[f1].values
x2 = df_class[f2].values

# параметры нормального распределения
mu1, sigma1 = x1.mean(), x1.std()
mu2, sigma2 = x2.mean(), x2.std()

print(f"[INFO] {f1}: mean={mu1:.4f}, std={sigma1:.4f}")
print(f"[INFO] {f2}: mean={mu2:.4f}, std={sigma2:.4f}")


# ===========================================================
#           ФУНКЦИЯ: График нормального распределения
# ===========================================================
def plot_feature_with_normal(x, mu, sigma, feature_name, filename):
    plt.figure(figsize=(8, 5))
    plt.hist(x, bins=30, density=True, alpha=0.6, label="Гистограмма")
    xs = np.linspace(min(x), max(x), 300)
    plt.plot(xs, norm.pdf(xs, mu, sigma), "r-", label="Нормальное распределение")
    plt.title(f"Параметрический подход — {feature_name}")
    plt.xlabel(feature_name)
    plt.ylabel("Плотность вероятности")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"figures/{filename}")
    plt.close()


# --- Рисуем графики ---
plot_feature_with_normal(x1, mu1, sigma1, f1, "lab4_task1_f1.png")
plot_feature_with_normal(x2, mu2, sigma2, f2, "lab4_task1_f2.png")

print("[INFO] Задание 1 выполнено — графики сохранены.")


# ===========================================================
#                 ЗАДАНИЕ 2: KDE
# ===========================================================
print("\n--- Задание 2: KDE ---")

x1 = df_class[f1].values.reshape(-1, 1)
x2 = df_class[f2].values.reshape(-1, 1)

x1_train, x1_test = train_test_split(x1, test_size=0.25, random_state=42)
x2_train, x2_test = train_test_split(x2, test_size=0.25, random_state=42)

bandwidths = np.logspace(-1, 4, 30)


def kde_log_likelihood(train, test, bw):
    return KernelDensity(kernel="gaussian", bandwidth=bw).fit(train).score(test)


def fit_kde(train, bw):
    return KernelDensity(kernel="gaussian", bandwidth=bw).fit(train)


# визуально выбранные значения
visual_bw_1 = 5000
visual_bw_2 = 30

# оптимальные по ML
best_bw_1 = max(bandwidths, key=lambda bw: kde_log_likelihood(x1_train, x1_test, bw))
best_bw_2 = max(bandwidths, key=lambda bw: kde_log_likelihood(x2_train, x2_test, bw))

print(f"[KDE] {f1}: визуальный={visual_bw_1}, ML={best_bw_1}")
print(f"[KDE] {f2}: визуальный={visual_bw_2}, ML={best_bw_2}")


# --- Графики KDE ---
def plot_kde_comparison(x, feature_name, visual_bw, best_bw, filename):
    xs = np.linspace(x.min(), x.max(), 400).reshape(-1, 1)
    kde_visual = fit_kde(x, visual_bw)
    kde_best = fit_kde(x, best_bw)

    pdf_visual = np.exp(kde_visual.score_samples(xs))
    pdf_best = np.exp(kde_best.score_samples(xs))

    plt.figure(figsize=(8, 5))
    plt.hist(x.flatten(), bins=30, density=True, alpha=0.5)
    plt.plot(xs, pdf_visual, label=f"KDE visual (bw={visual_bw})")
    plt.plot(xs, pdf_best, linestyle="--", label=f"KDE ML (bw={best_bw:.2f})")

    plt.title(f"KDE — {feature_name}")
    plt.xlabel(feature_name)
    plt.ylabel("Плотность")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"figures/{filename}")
    plt.close()


plot_kde_comparison(x1, f1, visual_bw_1, best_bw_1, "lab4_task2_f1.png")
plot_kde_comparison(x2, f2, visual_bw_2, best_bw_2, "lab4_task2_f2.png")

print("[INFO] Задание 2 выполнено.")


# ===========================================================
#               ЗАДАНИЕ 3: Gaussian Mixture
# ===========================================================
print("\n--- Задание 3: Gaussian Mixture ---")

N_COMPONENTS = [3, 4]


def plot_gmm(x, feature_name, n_components, filename):
    gmm = GaussianMixture(
        n_components=n_components, covariance_type="full", random_state=42
    )
    gmm.fit(x)

    # === КОНСОЛЬНЫЙ ВЫВОД ПАРАМЕТРОВ GMM ===
    print("\n=== Gaussian Mixture Model параметры для", feature_name, "===")
    for i in range(n_components):
        mean = gmm.means_[i, 0]
        std = np.sqrt(gmm.covariances_[i, 0, 0])
        weight = gmm.weights_[i]

        print(f"\nКомпонента {i+1}:")
        print(f"  Вес w = {weight:.4f}")
        print(f"  Среднее μ = {mean:.4f}")
        print(f"  Стандартное отклонение σ = {std:.4f}")
        print(f"  Вклад компоненты = {weight:.4f} * N(x | μ, σ)")

    # Остальной графический код
    xs = np.linspace(x.min(), x.max(), 400).reshape(-1, 1)
    total_pdf = np.exp(gmm.score_samples(xs))

    component_pdfs = []
    for i in range(n_components):
        mean = gmm.means_[i, 0]
        std = np.sqrt(gmm.covariances_[i, 0, 0])
        component_pdf = gmm.weights_[i] * norm.pdf(xs, mean, std)
        component_pdfs.append(component_pdf)

    plt.figure(figsize=(8, 5))
    plt.hist(x.flatten(), bins=30, density=True, alpha=0.4, label="Гистограмма данных")
    plt.plot(xs, total_pdf, "k-", linewidth=2, label="Суммарная плотность смеси")

    for i, pdf in enumerate(component_pdfs):
        plt.plot(xs, pdf, "--", linewidth=1.7, label=f"Компонента {i+1}")

    plt.title(f"Gaussian Mixture Model — {feature_name}")
    plt.xlabel(feature_name)
    plt.ylabel("Плотность")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"figures/{filename}")
    plt.close()


plot_gmm(x1, f1, N_COMPONENTS[0], "lab4_task3_f1.png")
plot_gmm(x2, f2, N_COMPONENTS[1], "lab4_task3_f2.png")

print("[INFO] Задание 3 выполнено.")


# ===========================================================
#       ЗАДАНИЕ 4: Наивный Байес для всех 3 подходов
# ===========================================================
print("\n--- Задание 4: Байесовская классификация ---")

EPS = 1e-12


def safe_prob(num, den):
    return num / max(den, EPS)


# --- Разделение ---
X = df[[f1, f2]].values
y = df["target_bin"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

train_c0 = X_train[y_train == 0]
train_c1 = X_train[y_train == 1]


# ---------------------- 1) ПАРАМЕТРИЧЕСКИЙ ----------------------
def normal_pdf(x, mu, sigma):
    sigma = max(sigma, EPS)
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (np.sqrt(2 * np.pi) * sigma)


def predict_parametric(X):
    mu0, mu1 = train_c0.mean(axis=0), train_c1.mean(axis=0)
    s0, s1 = train_c0.std(axis=0), train_c1.std(axis=0)

    p0 = len(train_c0) / len(X_train)
    p1 = 1 - p0

    scores = []
    for a, b in X:
        f1_c0 = normal_pdf(a, mu0[0], s0[0])
        f2_c0 = normal_pdf(b, mu0[1], s0[1])
        f1_c1 = normal_pdf(a, mu1[0], s1[0])
        f2_c1 = normal_pdf(b, mu1[1], s1[1])

        num = f1_c1 * f2_c1 * p1
        den = num + f1_c0 * f2_c0 * p0
        scores.append(safe_prob(num, den))

    return np.array(scores)


param_scores = predict_parametric(X_test)
print(f"AUC Parametric: {roc_auc_score(y_test, param_scores):.4f}")


# ---------------------- 2) KDE ---------------------------------
kde_c0_f1 = KernelDensity(bandwidth=best_bw_1).fit(train_c0[:, 0].reshape(-1, 1))
kde_c0_f2 = KernelDensity(bandwidth=best_bw_2).fit(train_c0[:, 1].reshape(-1, 1))

kde_c1_f1 = KernelDensity(bandwidth=best_bw_1).fit(train_c1[:, 0].reshape(-1, 1))
kde_c1_f2 = KernelDensity(bandwidth=best_bw_2).fit(train_c1[:, 1].reshape(-1, 1))


def predict_kde(X):
    p0 = len(train_c0) / len(X_train)
    p1 = 1 - p0
    scores = []

    for a, b in X:
        f1_c0 = np.exp(kde_c0_f1.score_samples([[a]]))[0]
        f2_c0 = np.exp(kde_c0_f2.score_samples([[b]]))[0]
        f1_c1 = np.exp(kde_c1_f1.score_samples([[a]]))[0]
        f2_c1 = np.exp(kde_c1_f2.score_samples([[b]]))[0]

        num = f1_c1 * f2_c1 * p1
        den = num + f1_c0 * f2_c0 * p0
        scores.append(safe_prob(num, den))

    return np.array(scores)


kde_scores = predict_kde(X_test)
print(f"AUC KDE: {roc_auc_score(y_test, kde_scores):.4f}")


# ---------------------- 3) GMM ---------------------------------
gmm_c0 = GaussianMixture(n_components=2).fit(train_c0)
gmm_c1 = GaussianMixture(n_components=2).fit(train_c1)


def predict_gmm(X):
    p0 = len(train_c0) / len(X_train)
    p1 = 1 - p0
    scores = []

    for a, b in X:
        p_c0 = np.exp(gmm_c0.score_samples([[a, b]]))[0]
        p_c1 = np.exp(gmm_c1.score_samples([[a, b]]))[0]

        num = p_c1 * p1
        den = num + p_c0 * p0
        scores.append(safe_prob(num, den))

    return np.array(scores)


gmm_scores = predict_gmm(X_test)
print(f"AUC GMM: {roc_auc_score(y_test, gmm_scores):.4f}")


# ---------------------- 4) GaussianNB ---------------------------
gnb = GaussianNB().fit(X_train, y_train)
auc_gnb = roc_auc_score(y_test, gnb.predict_proba(X_test)[:, 1])
print(f"AUC GaussianNB: {auc_gnb:.4f}")
