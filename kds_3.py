import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
import warnings
import seaborn as sns
import warnings
import joblib
import pydotplus

from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, validation_curve
from skompiler import skompile
import graphviz

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)



df = pd.read_csv("C:/Users/TR/PycharmProjects/PythonProject/kdsproje/labels_with_split.csv",sep=";")

df.head()
df.describe().T
df.nunique()

def grab_col_names(dataframe, cat_th=10,  car_th=20):
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.

    Parameters
    ----------
    dataframe: dataframe
        değişken isimleri alınmak istenen dataframe'dir.
    cat_th: int, float
        numerik fakat kategorik olan değişkenler için sınıf eşik değeri
    car_th: int, float
        kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    -------
    cat_cols: list
        Kategorik değişken listesi
    num_cols: list
        Numerik değişken listesi
    cat_but_car: list
        Kategorik görünümlü kardinal değişken listesi

    Notes
    ------
    cat_cols + num_cols + cat_but_car = toplam değişken sayısı
    num_but_cat cat_cols'un içerisinde.

    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]

    num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int", "float"]]

    cat_but_car = [col for col in df.columns if
                   df[col].nunique() > 20 and str(df[col].dtypes) in ["category", "object"]]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in df.columns if df[col].dtypes in ["int", "float"]]
    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)
corr = df[num_cols].corr()

sns.set(rc={'figure.figsize': (12, 12)})
sns.heatmap(corr, cmap="RdBu")
plt.show()


def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


for col in num_cols:
    num_summary(df, col, plot=True)

X = df.iloc[:,1:7]
kmeans = KMeans(n_clusters=7, random_state=17,).fit(X)
kmeans.get_params()


kmeans.n_clusters
kmeans.cluster_centers_
kmeans.labels_
kmeans.inertia_

kmeans = KMeans()
ssd = []
K = range(1, 10)

for k in K:
    kmeans = KMeans(n_clusters=k,).fit(X)
    ssd.append(kmeans.inertia_)



plt.plot(K, ssd, "bx-")
plt.xlabel("Farklı K Değerlerine Karşılık SSE/SSR/SSD")
plt.title("Optimum Küme sayısı için Elbow Yöntemi")
plt.show()

kmeans = KMeans() #  optimum küme sayısı 4
elbow = KElbowVisualizer(kmeans, k=(2, 10))
elbow.fit(X)
elbow.show()

kmeans = KMeans(n_clusters=elbow.elbow_value_).fit(X)

kmeans.n_clusters
kmeans.cluster_centers_
kmeans.labels_




clusters_kmeans = kmeans.labels_
df["cluster"] = clusters_kmeans

df["cluster"] = df["cluster"] + 1  #kümelerin 1 den başlaması için
df.groupby("cluster").agg({"cluster":"count"})

plt.scatter(x=df["cluster"],y=df["width"],color="red",)

df.groupby("cluster").agg({"width":"mean","xmax":"mean","xmin":"mean","ymax":"mean","ymin":"mean"})

df.to_csv("ucak kümeleri")


#cart ile doğrulama

y= df["cluster"]
X = df.drop("class", axis=1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=45)


cart_model = DecisionTreeClassifier(random_state=17).fit(X_train, y_train)

y_pred = cart_model.predict(X_train)
y_prob = cart_model.predict_proba(X_train)[:, 1]
print(classification_report(y_train, y_pred))


# Test Hatası
y_pred = cart_model.predict(X_test)
y_prob = cart_model.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))
roc_auc_score(y_test, y_prob)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import warnings

warnings.filterwarnings("ignore")


X_features = df.iloc[:, 1:7]  # width, height, xmin, ymin, xmax, ymax

"""
========================================================
  K-MEANS DOĞRULAMA VE GÖRSELLEŞTİRME BLOĞU
  Hava Savunma KDS Projesi

  Bu blok mevcut kodda şu satırın HEMEN ARKASINA eklenir:
      df.to_csv("ucak kümeleri")

  CART doğrulama bloğundan (#cart ile doğrulama) ÖNCE gelir.
========================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 0. HAZIRLIK
#    (Bu değişkenler mevcut kodda zaten tanımlı,
#     sadece referans için gösterilmiştir)
# ─────────────────────────────────────────────
# df       → küme sütunu eklenmiş DataFrame
# X        → df.iloc[:,1:7]  (kümeleme için kullanılan özellikler)
# kmeans   → eğitilmiş KMeans modeli
# clusters_kmeans → kmeans.labels_

X_features = df.iloc[:, 1:7]  # width, height, xmin, ymin, xmax, ymax

# ─────────────────────────────────────────────
# 1. İÇ DOĞRULAMA METRİKLERİ
# ─────────────────────────────────────────────
sil_score = silhouette_score(X_features, kmeans.labels_)
db_score = davies_bouldin_score(X_features, kmeans.labels_)
ch_score = calinski_harabasz_score(X_features, kmeans.labels_)

print("=" * 55)
print("       K-MEANS İÇ DOĞRULAMA METRİKLERİ")
print("=" * 55)
print(f"  Silhouette Skoru          : {sil_score:.4f}  (1'e yakın = iyi)")
print(f"  Davies-Bouldin İndeksi    : {db_score:.4f}  (0'a yakın = iyi)")
print(f"  Calinski-Harabasz İndeksi : {ch_score:.2f}  (yüksek = iyi)")
print("=" * 55)

# ─────────────────────────────────────────────
# 2. SİLHOUETTE ANALİZİ GRAFİĞİ
# ─────────────────────────────────────────────
n_clusters = kmeans.n_clusters
sample_silhouette_values = silhouette_samples(X_features, kmeans.labels_)

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim([-0.1, 1])
ax.set_ylim([0, len(X_features) + (n_clusters + 1) * 10])

y_lower = 10
kume_renkleri = cm.tab10(np.linspace(0, 1, n_clusters))

for i in range(n_clusters):
    ith_cluster_silhouette_values = np.sort(
        sample_silhouette_values[kmeans.labels_ == i]
    )
    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i

    color = kume_renkleri[i]
    ax.fill_betweenx(
        np.arange(y_lower, y_upper),
        0,
        ith_cluster_silhouette_values,
        facecolor=color,
        edgecolor=color,
        alpha=0.7,
    )
    ax.text(-0.05, y_lower + 0.5 * size_cluster_i, f"Küme {i + 1}", fontsize=9)
    y_lower = y_upper + 10

ax.axvline(x=sil_score, color="red", linestyle="--", linewidth=1.5,
           label=f"Ortalama Silhouette = {sil_score:.3f}")
ax.set_xlabel("Silhouette Katsayısı", fontsize=12)
ax.set_ylabel("Küme Etiketi", fontsize=12)
ax.set_title("Küme Başına Silhouette Analizi (Hava Hedef Kümeleri)", fontsize=13, fontweight="bold")
ax.set_yticks([])
ax.legend(loc="upper right")
plt.tight_layout()
plt.savefig("silhouette_analizi.png", dpi=150, bbox_inches="tight")
plt.show()
print(">> silhouette_analizi.png kaydedildi.")

# ─────────────────────────────────────────────
# 3. PCA İLE 2D KÜME GÖRSELLEŞTİRMESİ
# ─────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_features)

pca = PCA(n_components=2, random_state=17)
X_pca = pca.fit_transform(X_scaled)

df["pca_1"] = X_pca[:, 0]
df["pca_2"] = X_pca[:, 1]

# Merkez noktaları PCA uzayına dönüştür
centers_scaled = scaler.transform(kmeans.cluster_centers_)
centers_pca = pca.transform(centers_scaled)

fig, ax = plt.subplots(figsize=(10, 7))
palette = sns.color_palette("tab10", n_colors=n_clusters)

for i in range(n_clusters):
    mask = df["cluster"] == (i + 1)
    ax.scatter(
        df.loc[mask, "pca_1"],
        df.loc[mask, "pca_2"],
        s=12,
        color=palette[i],
        alpha=0.55,
        label=f"Küme {i + 1}",
    )

# Merkezleri çiz
ax.scatter(
    centers_pca[:, 0],
    centers_pca[:, 1],
    s=220,
    c="black",
    marker="X",
    zorder=5,
    label="Küme Merkezi",
)
for idx, (cx, cy) in enumerate(centers_pca):
    ax.annotate(
        f"K{idx + 1}",
        (cx, cy),
        textcoords="offset points",
        xytext=(8, 6),
        fontsize=9,
        fontweight="bold",
        color="black",
    )

ax.set_xlabel(f"PCA Bileşen 1 (Açıklanan Varyans: %{pca.explained_variance_ratio_[0] * 100:.1f})", fontsize=11)
ax.set_ylabel(f"PCA Bileşen 2 (Açıklanan Varyans: %{pca.explained_variance_ratio_[1] * 100:.1f})", fontsize=11)
ax.set_title("K-Means Kümeleri – PCA 2D Projeksiyonu (Hava Hedefleri)", fontsize=13, fontweight="bold")
ax.legend(loc="upper right", fontsize=9, markerscale=1.5)
plt.tight_layout()
plt.savefig("pca_kume_gorseli.png", dpi=150, bbox_inches="tight")
plt.show()
print(">> pca_kume_gorseli.png kaydedildi.")

# ─────────────────────────────────────────────
# 4. KÜME BAZINDA ÖZELLIK KUTU GRAFİKLERİ
# ─────────────────────────────────────────────
ozellikler = ["width", "height", "xmin", "ymin", "xmax", "ymax"]
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()

for idx, ozellik in enumerate(ozellikler):
    sns.boxplot(
        data=df,
        x="cluster",
        y=ozellik,
        palette="tab10",
        ax=axes[idx],
        linewidth=0.8,
    )
    axes[idx].set_title(f"{ozellik.upper()} – Kümelere Göre Dağılım", fontsize=11, fontweight="bold")
    axes[idx].set_xlabel("Küme", fontsize=10)
    axes[idx].set_ylabel(ozellik, fontsize=10)

plt.suptitle("Özellik Dağılımları – Küme Bazında Karşılaştırma (Hava Savunma KDS)",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("kutu_grafikleri.png", dpi=150, bbox_inches="tight")
plt.show()
print(">> kutu_grafikleri.png kaydedildi.")

# ─────────────────────────────────────────────
# 5. RADAR (SPIDER) GRAFİĞİ – KÜME PROFİLLERİ
# ─────────────────────────────────────────────
from sklearn.preprocessing import MinMaxScaler

kume_profil = df.groupby("cluster")[ozellikler].mean()
scaler_radar = MinMaxScaler()
kume_profil_norm = pd.DataFrame(
    scaler_radar.fit_transform(kume_profil),
    index=kume_profil.index,
    columns=ozellikler,
)

kategoriler = ozellikler + [ozellikler[0]]  # kapalı polygon için tekrar ilk
N = len(ozellikler)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

for i, row in kume_profil_norm.iterrows():
    degerler = row.tolist() + [row.tolist()[0]]
    ax.plot(angles, degerler, "o-", linewidth=2, label=f"Küme {i}", color=palette[i - 1])
    ax.fill(angles, degerler, alpha=0.12, color=palette[i - 1])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(ozellikler, fontsize=11)
ax.set_yticks([0.2, 0.4, 0.6, 0.8])
ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8"], fontsize=8)
ax.set_title("Küme Profil Radarı – Normalize Özellik Ortalamaları\n(Hava Savunma KDS)",
             fontsize=13, fontweight="bold", pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)
plt.tight_layout()
plt.savefig("radar_grafigi.png", dpi=150, bbox_inches="tight")
plt.show()
print(">> radar_grafigi.png kaydedildi.")



# ─────────────────────────────────────────────
# 7. KÜME DAĞILIM PASTA GRAFİĞİ
# ─────────────────────────────────────────────
kume_sayilari = df["cluster"].value_counts().sort_index()
labels_pasta = [f"Küme {k}" for k in kume_sayilari.index]
explode = [0.04] * len(kume_sayilari)

fig, ax = plt.subplots(figsize=(7, 7))
wedges, texts, autotexts = ax.pie(
    kume_sayilari,
    labels=labels_pasta,
    autopct="%1.1f%%",
    explode=explode,
    colors=palette[:len(kume_sayilari)],
    startangle=140,
    pctdistance=0.82,
    textprops={"fontsize": 11},
)
for autotext in autotexts:
    autotext.set_fontweight("bold")

ax.set_title("Küme Dağılımı – Hedef Tipi Oranları (Hava Savunma KDS)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("pasta_grafigi.png", dpi=150, bbox_inches="tight")
plt.show()
print(">> pasta_grafigi.png kaydedildi.")

# ─────────────────────────────────────────────
# 8. ÖZET TABLO
# ─────────────────────────────────────────────

ozet = df.groupby("cluster").agg(
    Adet=("cluster", "count"),
    Ort_Genislik=("width", "mean"),
    Ort_Yukseklik=("height", "mean"),
    Ort_BBox_Alan=("width", lambda x: (
            (df.loc[x.index, "xmax"] - df.loc[x.index, "xmin"]) *
            (df.loc[x.index, "ymax"] - df.loc[x.index, "ymin"])
    ).mean()),
).round(1)
print(ozet.to_string())
print("=" * 70)
print(f"\n  Silhouette : {sil_score:.4f}  |  Davies-Bouldin : {db_score:.4f}  |  CH : {ch_score:.2f}")
print("=" * 70)

# ─────────────────────────────────────────────
# TEMIZLIK: Geçici PCA sütunlarını kaldır
# ─────────────────────────────────────────────
df.drop(columns=["pca_1", "pca_2"], inplace=True, errors="ignore")

def hava_savunma_kds(kume_id):
    """
    Sadece K-Means küme numarasını (0, 1, 2, 3) alarak
    uygun hava savunma aksiyonunu döndüren KDS fonksiyonu.
    """

    print("\n[!] RADAR TESPİTİ - SİSTEM UYARISI [!]")
    print("-" * 40)

    if kume_id == 1:
        print("Hedef Tipi : Küme 1 - Küçük Boyutlu Hedef (İHA veya Kamikaze Drone)")
        print("Aksiyon    : Kısa Menzilli Füze (MANPADS) kilitlensin ve Jammer'lar aktif edilsin.")

    elif kume_id == 2:
        print("Hedef Tipi : Küme 2 - Orta Boyutlu / Yüksek İrtifa (Avcı Uçağı)")
        print("Aksiyon    : Önleyici Avcı Filosu (Interceptors) havalandırılsın.")

    elif kume_id == 3:
        print("Hedef Tipi : Küme 3 - Orta Boyutlu / Alçak İrtifa (Yakın Hava Desteği - CAS)")
        print("Aksiyon    : Mobil Hava Savunma Bataryaları angaje edilsin, uçaksavar ateşi başlasın.")

    elif kume_id == 4:
        print("Hedef Tipi : Küme 4 - Büyük Stratejik Hedef (Ağır Bombardıman / Kargo)")
        print("Aksiyon    : Ağır Anti-Air (AA) bataryaları ve Uzun Menzilli SAM füzeleri ateşlensin.")

    else:
        print("HATA       : Tanımsız Küme! Lütfen 1, 2, 3 veya 4 değerlerinden birini girin.")

    print("-" * 40)


hava_savunma_kds(3)

import streamlit as st

import streamlit as st

st.title("Hava Savunma Karar Destek Sistemi")

kume_id = st.selectbox("Küme Numarası Seçin", [1, 2, 3, 4])

if st.button("Savunma Önerisi Getir"):
    if kume_id == 1:
        st.write("**Hedef Tipi:** Küme 1 - Küçük Boyutlu Hedef (İHA veya Kamikaze Drone)")
        st.write("**Aksiyon:** Kısa Menzilli Füze (MANPADS) kilitlensin ve Jammer'lar aktif edilsin.")
    elif kume_id == 2:
        st.write("**Hedef Tipi:** Küme 2 - Orta Boyutlu / Yüksek İrtifa (Avcı Uçağı)")
        st.write("**Aksiyon:** Önleyici Avcı Filosu (Interceptors) havalandırılsın.")
    elif kume_id == 3:
        st.write("**Hedef Tipi:** Küme 3 - Orta Boyutlu / Alçak İrtifa (Yakın Hava Desteği - CAS)")
        st.write("**Aksiyon:** Mobil Hava Savunma Bataryaları angaje edilsin, uçaksavar ateşi başlasın.")
    elif kume_id == 4:
        st.write("**Hedef Tipi:** Küme 4 - Büyük Stratejik Hedef (Ağır Bombardıman / Kargo)")
        st.write("**Aksiyon:** Ağır Anti-Air (AA) bataryaları ve Uzun Menzilli SAM füzeleri ateşlensin.")