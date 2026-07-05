import pandas as pd
import numpy as np
import gensim.downloader as api
from sklearn.decomposition import PCA
from pathlib import Path
try:
    from data_processing import get_data
except ImportError:
    get_data = None
import os
import json


players_path = os.environ.get("DISTRAJ_PLAYERS_JSON", os.path.join("data", "raw", "players.json"))
with open(players_path, 'r', encoding='utf-8') as f:
    players = json.load(f)


all_trackable_ids = [p['trackable_object'] for p in players]
unique_trackable_ids = sorted(set(all_trackable_ids))

# 创建一个映射：trackable_object → 连续编号
trackable_to_idx = {tid: i for i, tid in enumerate(unique_trackable_ids)}

print(f"总共 {len(unique_trackable_ids)} 个唯一节点")

df = pd.DataFrame(players)

df['trackable_object'] = df['trackable_object'].astype(int)

df['age'] = 2025 - pd.to_datetime(df['birthday']).dt.year

df['gender_binary'] = df['gender'].map({'male': 1, 'female': 0}).fillna(0).astype(int)

names = df['short_name'].astype(str).tolist()

print("加载 FastText 模型...")
ft = api.load("fasttext-wiki-news-subwords-300")


vecs_300 = np.zeros((len(names), ft.vector_size), dtype=np.float32)
for i, name in enumerate(names):
    tokens = name.lower().split()
    tok_vecs = []
    for t in tokens:
        try:
            tok_vecs.append(ft[t])
        except KeyError:
            continue
    if tok_vecs:
        vecs_300[i] = np.mean(tok_vecs, axis=0)
    else:
        vecs_300[i] = np.zeros(ft.vector_size, dtype=np.float32)

print("PCA 降维到 6 维...")
pca = PCA(n_components=6)
vecs_6 = pca.fit_transform(vecs_300)

# 把所有特征合成 DataFrame
static_features_df = pd.DataFrame({
    'trackable_object': df['trackable_object'],
    'age': df['age'],
    'gender_binary': df['gender_binary']
})

for i in range(6):
    static_features_df[f'vec_{i}'] = vecs_6[:, i]

print(static_features_df.head())
print("静态特征矩阵 shape:", static_features_df.shape)

save_dir = os.environ.get("DISTRAJ_STATIC_FEATURE_DIR", os.path.join("data", "processed", "metadata_players"))
os.makedirs(save_dir, exist_ok=True)

save_path = os.path.join(save_dir, 'static_features.csv')
static_features_df.to_csv(save_path, index=False)
print(f"静态特征保存到 {save_path}")
