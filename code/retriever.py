import os
import joblib
import numpy as np
import pandas as pd
from transformers import AutoModel

class SimCSE_Retriever():
    def __init__(self, config) -> None:
        self.similarities_dict = dict()
        base_path = os.path.join("..", "cache")
        simcse_model_paths = [
            file
            for file in os.listdir(base_path)
            if file.startswith(f"simcse_similarities_{config.feature}")
        ]
        simcse_model_paths = [
            file
            for file in simcse_model_paths
            if file.endswith(f"ratio_{config.ratio_of_source_used}.joblib")
        ]
        for path in simcse_model_paths:
            self.similarities_dict.update(joblib.load(os.path.join(base_path, path)))
        print("Number of files loaded:", len(simcse_model_paths))

    def retrieve_similar_cases(
        self, case: str, num_cases: int = 1, threshold: float = -np.inf
    ):
        sentences_and_similarities = self.similarities_dict[case.strip()].items()
        sentences_and_similarities_sorted = sorted(
            sentences_and_similarities, key=lambda x: x[1][0], reverse=True
        )

        return [
            {"similar_case": x[0], "similar_case_label": x[1][1]}
            for x in sentences_and_similarities_sorted[1 : num_cases + 1]
            if x[1][0] > threshold
        ]

def get_embeddings_simcse(model, text: str):
    return model.encode(text)


def generate_the_simcse_similarities(
    source_file: str,
    target_file_template: str,
    output_file_template: str,
    feature: str,
    ratio_of_source_used: float = 1.0,
):


    model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-roberta-large")

    source_file_df = (
        pd.read_csv(source_file)
        .groupby("label")
        .apply(lambda x: x.sample(frac=ratio_of_source_used))
        .reset_index(drop=True)
    )

    train_sentences = source_file_df[feature].tolist()
    train_labels = source_file_df["label"].tolist()

    train_sentences = [x.strip() for x in train_sentences]

    for split in ["train", "dev", "test"]:
        target_file = target_file_template.replace("split", split)

        all_sentences = pd.read_csv(target_file)[feature].tolist()
        all_sentences = [x.strip() for x in all_sentences]

        similarities = model.similarity(all_sentences, train_sentences)
        similarities_dict = dict()
        for sentence, row in zip(all_sentences, similarities):
            similarities_dict[sentence] = dict(
                zip(train_sentences, list(zip(row.tolist(), train_labels)))
            )

        output_file = output_file_template.replace("split", split)

        joblib.dump(similarities_dict, output_file)


