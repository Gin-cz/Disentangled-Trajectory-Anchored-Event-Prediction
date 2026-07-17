"""Build static player metadata features."""

import json
import os

import numpy as np
import pandas as pd


def build_player_features(players_path=None, save_dir=None):
    import gensim.downloader as api
    from sklearn.decomposition import PCA

    players_path = players_path or os.environ.get(
        "DISTRAJ_PLAYERS_JSON", os.path.join("data", "raw", "players.json")
    )
    save_dir = save_dir or os.environ.get(
        "DISTRAJ_STATIC_FEATURE_DIR", os.path.join("data", "processed", "metadata_players")
    )

    with open(players_path, "r", encoding="utf-8") as file:
        players = json.load(file)

    dataframe = pd.DataFrame(players)
    dataframe["trackable_object"] = dataframe["trackable_object"].astype(int)
    dataframe["age"] = 2025 - pd.to_datetime(dataframe["birthday"]).dt.year
    dataframe["gender_binary"] = (
        dataframe["gender"].map({"male": 1, "female": 0}).fillna(0).astype(int)
    )

    names = dataframe["short_name"].astype(str).tolist()
    fasttext = api.load("fasttext-wiki-news-subwords-300")
    vectors_300 = np.zeros((len(names), fasttext.vector_size), dtype=np.float32)

    for index, name in enumerate(names):
        token_vectors = []
        for token in name.lower().split():
            try:
                token_vectors.append(fasttext[token])
            except KeyError:
                continue
        if token_vectors:
            vectors_300[index] = np.mean(token_vectors, axis=0)

    vectors_6 = PCA(n_components=6).fit_transform(vectors_300)
    static_features = pd.DataFrame(
        {
            "trackable_object": dataframe["trackable_object"],
            "age": dataframe["age"],
            "gender_binary": dataframe["gender_binary"],
        }
    )
    for index in range(6):
        static_features[f"vec_{index}"] = vectors_6[:, index]

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "static_features.csv")
    static_features.to_csv(save_path, index=False)
    return save_path


if __name__ == "__main__":
    build_player_features()
