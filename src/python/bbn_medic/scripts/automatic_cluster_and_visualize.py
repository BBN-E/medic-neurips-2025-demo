import argparse
import ast
import os
import random
import warnings
from itertools import islice

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN, HDBSCAN, MeanShift
from sklearn.manifold import TSNE
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

warnings.simplefilter(action='ignore', category=FutureWarning)


def batched(iterable, n, *, strict=False):
    """Defines itertools' batched to for compatibility with Python versions before 3.12"""
    # batched('ABCDEFG', 3) -> ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        if strict and len(batch) != n:
            raise ValueError('batched(): incomplete batch')
        yield batch


def HDBSCAN_clustering(prompt_embeddings):
    """Returns a numpy array of cluster ids for each prompt embedding using HDBSCAN."""
    hdb = HDBSCAN(min_cluster_size=20)
    hdb.fit(prompt_embeddings)
    return hdb.labels_


def DBSCAN_clustering(prompt_embeddings):
    """Returns a numpy array of cluster ids for each prompt embedding using DBSCAN."""
    db = DBSCAN()
    db.fit(prompt_embeddings)
    return db.labels_


def MeanShift_clustering(prompt_embeddings):
    """Returns a numpy array of cluster ids for each prompt embedding using MeanShift."""
    MS = MeanShift()
    MS.fit(prompt_embeddings)
    return MS.labels_


def cluster(method, prompt_embeddings):
    """Calls the appropriate clustering method and returns a numpy array of cluster ids."""
    if method == "HDBSCAN":
        return HDBSCAN_clustering(prompt_embeddings=prompt_embeddings)
    elif method == "DBSCAN":
        return DBSCAN_clustering(prompt_embeddings=prompt_embeddings)
    elif method == "MeanShift":
        return MeanShift_clustering(prompt_embeddings=prompt_embeddings)
    else:
        return ValueError("Clustering Method not in [HDBSCAN, DBSCAN, MeanShift]")


def visualize(embeddings, out_dir, labels, clusters, model_type="umap", dim=2, desires=None):
    """Applies model_type(UMAP or TSNE) to reduce embeddings to dim dimensional space. Then, 
    the clusters are graphed onto a scatterplot with reduced desire embeddings optionally 
    plotted as well. """
    num_prompts = len(embeddings)
    if desires is not None:
        embeddings = np.concatenate([embeddings, desires], axis=0)
    if model_type.lower() == "umap":
        import umap
        reducer = umap.UMAP(n_components=dim)
        embs = reducer.fit_transform(embeddings)
        embs, prompt_embs = embs[:num_prompts], embs[num_prompts:]
    elif model_type.lower() == "tsne":
        tsne = TSNE(random_state=42, max_iter=1000, metric="euclidean", n_components=dim)
        embs = tsne.fit_transform(embeddings)
        embs, prompt_embs = embs[:num_prompts], embs[num_prompts:]
    else:
        raise ValueError("Unknown modeling type")

    if desires is not None:
        assert len(
            prompt_embs) > 0, "There should be reduced desire embeddings if desire embeddings are passed into this function"

    if dim == 2:
        values = np.hsplit(embs, 1 + np.arange(embs.shape[1] - 1))  # splitting into [x, y]
        assert len(values) == 2
        labels["x"] = values[0]
        labels["y"] = values[1]
    elif dim == 3:
        values = np.hsplit(embs, 1 + np.arange(embs.shape[1] - 1))  # splitting into [x, y, z]
        assert len(values) == 3
        labels["x"] = values[0]
        labels["y"] = values[1]
        labels["z"] = values[2]
    else:
        raise ValueError("Incorrect dimension parameter. The visualization dimension should either be 2 or 3.")

    name = f"{model_type}_{dim}d"
    # the prompt embedding split
    desire_values = np.hsplit(prompt_embs, 1 + np.arange(prompt_embs.shape[1]))
    plot(out_dir, values, labels, name, clusters, desire_values=desire_values)


def plot(out_dir, values, labels, name, clusters, desire_values=None):
    """Creates plots of the 2D or 3D reduced embedding data. Creates 2 PNG images under 
    args.output_directory/args.cluster_method/: The first is a complete visualization of all data
    and the second partitions the data by cluster id. """
    if len(values) == 2:
        plt.scatter(values[0], values[1], color="blue", alpha=0.25, label="All prompts", s=6)
        if desire_values:  # Plot desires if wanted
            plt.scatter(desire_values[0], desire_values[1], color="black", alpha=1, label="Desires", s=3)
        plt.title(f"2D scatterplot")
        plt.savefig(os.path.join(out_dir, f"{name}_complete.png"))
        plt.clf()

        # graphing individual portions in different colors
        for i, id in enumerate(clusters):
            plt.scatter(labels[labels["Cluster_label"] == id]["x"], labels[labels["Cluster_label"] == id]["y"],
                        label=f"Cluster {id}", alpha=0.25, s=6)
        if desire_values:  # Plot desires if wanted
            plt.scatter(desire_values[0], desire_values[1], color="black", alpha=1, label="Desires", s=3)
        plt.title(f"2D scatterplot")

        plt.tight_layout(rect=[0, 0, 0.80, 1])
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1), title="Clusters")
        plt.savefig(os.path.join(out_dir, f"{name}_clusters.png"))
        plt.clf()

    elif len(values) == 3:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        # Scatter plot with color mapping
        sc = ax.scatter3D(values[0], values[1], values[2], c=values[2], cmap='viridis', marker='^')
        if desire_values:  # Plot desires if wanted
            ax.scatter3D(desire_values[0], desire_values[1], desire_values[2], color="black", alpha=1, label="Desires",
                         s=6)

        # Labels
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')
        ax.set_title('3D Scatter Plot with Color Mapping')

        plt.tight_layout()
        plt.colorbar(sc, ax=ax, label='Z Value')
        fig.savefig(os.path.join(out_dir, f"{name}_complete.png"))
        plt.close(fig)

        marker_size = 40
        alpha_value = 0.75
        z_min = labels['z'].min()
        z_max = labels['z'].max()
        x_min = labels['x'].min()
        x_max = labels['x'].max()
        y_min = labels['y'].min()
        y_max = labels['y'].max()

        # Determine the grid size for subplots
        num_clusters = len(clusters)
        nrows = int(np.ceil(np.sqrt(num_clusters)))
        ncols = int(np.ceil(num_clusters / nrows))

        # Create a single figure to hold all subplots
        fig_all_clusters = plt.figure(figsize=(ncols * 6, nrows * 5))
        fig_all_clusters.suptitle(f'3D Scatter Plots of All Clusters', fontsize=16)

        for i, cluster_id in enumerate(clusters):
            # Add a subplot to the figure
            ax_cluster = fig_all_clusters.add_subplot(nrows, ncols, i + 1, projection='3d')

            cluster_data = labels.loc[(labels["Cluster_label"] == cluster_id)]

            # Scatter plot for the current cluster
            sc_cluster = ax_cluster.scatter3D(
                cluster_data["x"],
                cluster_data["y"],
                cluster_data["z"],
                c=cluster_data["z"],
                cmap='viridis',
                s=marker_size,
                alpha=alpha_value,
                vmin=z_min,
                vmax=z_max
            )
            if desire_values:  # Plot desires if wanted
                ax_cluster.scatter3D(desire_values[0], desire_values[1], desire_values[2], color="black", alpha=1,
                                     label="Desires", s=6)
            ax_cluster.set_xlim(x_min, x_max)
            ax_cluster.set_ylim(y_min, y_max)
            ax_cluster.set_zlim(z_min, z_max)

            ax_cluster.set_xlabel('X')
            ax_cluster.set_ylabel('Y')
            ax_cluster.set_zlabel('Z')
            ax_cluster.set_title(f'Cluster {cluster_id}')

        # Adjust layout to prevent overlap
        plt.tight_layout(rect=[0, 0.03, 0.88, 0.95])

        # Add a single colorbar for the entire figure
        cbar_ax = fig_all_clusters.add_axes([0.90, 0.15, 0.02, 0.7])
        fig_all_clusters.colorbar(sc_cluster, cax=cbar_ax, label='Z Value')

        # Save the combined figure
        fig_all_clusters.savefig(os.path.join(out_dir, f"{name}_all_clusters_subplot.png"), dpi=300)
        plt.close(fig_all_clusters)


def generate_hiatus_compatible_file(df, raw_data_path):
    """Copies the original data into a new jsonl that is compatible with
    the HIATUS stylistic embedding script's expected format."""
    # copying original file into HIATUS compatible format inside raw_data
    temp_df = df.copy()
    temp_df["fullText"] = temp_df["text"]  # requires "text" column => "fullText" column
    temp_df["documentID"] = temp_df["id"]  # "id" column => "documentID" column

    raw_stylistic_data_dir = os.path.join(raw_data_path, "stylistic_data")
    os.makedirs(raw_stylistic_data_dir, exist_ok=True)
    data_path = os.path.join(raw_stylistic_data_dir, "hiatus_compatible_jsonl.jsonl")
    temp_df.to_json(data_path, lines=True, orient='records')
    return data_path, raw_stylistic_data_dir


def embed_bert(df):
    """Uses BERT embeddings to embed text (df["text"]). A sentence embedding is represented
    by the embedding of the CLS token. Embeddings are returned as a NumPy array of 
    shape [len(df), hidden_size]."""
    # Load BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    model.cuda()

    hidden_dim = model.config.hidden_size
    temp_embeddings = np.empty([0, hidden_dim])

    for batch in tqdm(batched(df["text"], 256), total=len(df["text"]) // 256 + 1):
        encoding = tokenizer.batch_encode_plus(batch,
                                               padding=True,  # Pad to the maximum sequence length
                                               truncation=True,  # Truncate to the maximum sequence length if necessary
                                               return_tensors='pt',  # Return PyTorch tensors
                                               add_special_tokens=True  # Add special tokens CLS and SEP
                                               )
        token_ids = encoding['input_ids'].to(model.device)
        attention_mask = encoding['attention_mask'].to(model.device)

        # Generate embeddings using BERT model
        with torch.no_grad():
            outputs = model(token_ids, attention_mask=attention_mask)
            prompt_embeddings = outputs.last_hidden_state[:, 0, :].detach().cpu()  # CLS embeddings
            prompt_embeddings = np.array(prompt_embeddings)
            temp_embeddings = np.concatenate((temp_embeddings, prompt_embeddings), axis=0)
    return temp_embeddings


def embed_sbert(df):
    """Uses specialized Sentence Transformer's sBERT embeddings to embed text (df["text"]). 
    Embeddings are returned as a NumPy array of shape [len(df), hidden_size]."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2").cuda()
    hidden_size = model.get_sentence_embedding_dimension()
    temp_embeddings = np.empty([0, hidden_size])

    for batch in tqdm(batched(df["text"], 256), total=len(df["text"]) // 256 + 1):
        # Generate embeddings using sBERT model
        with torch.no_grad():
            outputs = model.encode(batch)
            temp_embeddings = np.concatenate((temp_embeddings, outputs), axis=0)

    return temp_embeddings


def embed_stylistic(data_path, raw_stylistic_data_dir):
    """Uses Hiatus (release R2025_05_21) scripts to embed text. data_path is the path to 
    the jsonl file with text under column "fullText" and raw_stylistic_data_dir is the
    output file for stored embeddings and other diagnostics."""
    # Uses nfs.nimble.projects.hiatus.releases.hiatus.R2025_05_21
    model_mode_name = "decode"
    # from bbn_alert.feature_space_generation.features_luar.run import predict_features_luar
    from datasets import Dataset
    from bbn_alert.nn.trainer import LightningTrainerWrapper

    # config:
    from bbn_alert.common.data_types import ModelMode
    from bbn_alert.common.logger import logger
    from bbn_alert.feature_space_generation.driver import Driver

    params = {
        "data": {
            "datasets": [
                {
                    "dataset_type": "hrs_dataset",
                    "train": {
                        "filelist": f"{data_path}"
                    }
                }
            ],
            "prediction_output_dir": f"{raw_stylistic_data_dir}"
        },
        "extractors": [
            {
                "num_gpu": 1,
                "num_workers": 1,
                "cache_dir": "/nfs/nimble/projects/hiatus/hqiu/hiatus_public/transformers_cache/",
                "extractor_type": "luar_feature_extractor",
                "encoder_name": "sentence-transformers/paraphrase-distilroberta-base-v1",
                "task": "features_luar",
                "type": None,
                "model_path": "/nfs/nimble/projects/hiatus/hqiu/hiatus_public/032624_milestone_1.3_delivery/model/bbn/epoch=19-step=2880.ckpt",
                "hyper-parameters": {
                    "hyperparameters_type": "LUARHyperParameters",
                    "batch_size": 128,
                    "token_max_length": 512
                },
                "seed": 42,
                "text_normalization_processors": [

                ]
            }
        ]
    }

    model_mode = ModelMode[model_mode_name]

    driver = Driver(params, model_mode)

    assert len(params['extractors']) == 1
    extractor_params = params['extractors'][0]
    extractor = driver.get_extractor(extractor_params)

    output_dir = params.get("data", dict()).get("prediction_output_dir", None)
    if output_dir is None:
        raise ValueError(
            "Output dir must have valid value under data/prediction_output_dir. We got {}".format(output_dir))
    doc_ids_sum = list()
    features_sum = list()
    trainer = LightningTrainerWrapper(extractor.extraction_model.config, extractor.extractor_params,
                                      extractor.extraction_model.model_mode, extractor.extraction_model.hyper_params)
    prediction = trainer.predict(extractor.extraction_model)

    for dataloadidx_idx, dataloader_predition in enumerate(prediction):
        doc_ids_sum.extend(dataloader_predition['documentID'])
        features_sum.extend(dataloader_predition['features'])
    dataset_dict = Dataset.from_dict({
        "documentID": doc_ids_sum,
        "features": features_sum,
    })
    prompt_embeddings = np.array(features_sum)
    dataset_dict.save_to_disk(output_dir)

    return prompt_embeddings


def embed_text(embedding_style, df, out_dir):
    """Calls the appropriate embedding function and returns both the NumPy
    prompt embeddings and the modified df."""
    # Embed using specified style
    if embedding_style == "bert":
        print("Generating BERT Semantic embeddings.")
        prompt_embeddings = embed_bert(df)
        df["prompt_embeddings"] = [i.tolist() for i in list(prompt_embeddings)]

    elif embedding_style == "sbert":
        print("Generating sBERT semantic embeddings")

        prompt_embeddings = embed_sbert(df)
        df["prompt_embeddings"] = [i.tolist() for i in list(prompt_embeddings)]

    elif embedding_style == "stylistic":
        print("Generating Stylistic embeddings")
        data_path, style_data_dir = generate_hiatus_compatible_file(df, out_dir)
        prompt_embeddings = embed_stylistic(data_path, style_data_dir)
        df["prompt_embeddings"] = [i.tolist() for i in list(prompt_embeddings)]

    elif embedding_style == "mixed":
        print("Generating concatenated stylistic and semantic embeddings")
        data_path, style_data_dir = generate_hiatus_compatible_file(df, out_dir)
        stylistic_embeddings = embed_stylistic(data_path, style_data_dir)
        semantic_embeddings = embed_sbert(df)

        prompt_embeddings = np.concat((stylistic_embeddings, semantic_embeddings), axis=1)
        df["prompt_embeddings"] = [i.tolist() for i in list(prompt_embeddings)]
    else:
        raise ValueError(
            "Select a embedding style from \"stylistic\", \"bert\", \"sbert\", and \"mixed\". Mixed denotes stylistic embeddings concatenated with sbert embeddings. ")
    return prompt_embeddings, df


def main(args):
    # making output directories
    for cluster_method in args.clustering_methods:
        for embedding_style in args.embedding_styles:
            out_dir = os.path.join(args.output_directory, embedding_style, cluster_method)
            raw_data_dir = os.path.join(out_dir, "raw_data")
            os.makedirs(out_dir, exist_ok=True)
            os.makedirs(raw_data_dir, exist_ok=True)

    # Loop over embedding_style
    for i, embedding_style in enumerate(args.embedding_styles):
        print(f"\n====================================================== \
              \nEmbedding and clustering {embedding_style} embeddings.")
        base_dir = os.path.join(args.output_directory, embedding_style)
        # Loading embeddings 
        if args.embedded_prompts_paths:
            assert len(args.embedded_prompts_paths) == len(args.embedding_styles)
            print(f"Loading pre-embedded prompts from {args.embedded_prompts_paths[i]}.")
            df = pd.read_csv(args.embedded_prompts_paths[i])
            prompt_embeddings = np.array([ast.literal_eval(s) for s in df["prompt_embeddings"]])

            # Embed desires if needed
            if args.desire_path is not None:
                desires_path = args.desire_path
                desires_df = pd.read_json(desires_path, lines=True)
                desire_embeddings, desires_df = embed_text(embedding_style, desires_df, base_dir)
            else:
                desire_embeddings = None
        else:
            # Set a random seed
            random_seed = 42
            random.seed(random_seed)
            torch.manual_seed(random_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(random_seed)

            if args.prompts_path.endswith(".txt"):
                jsonl_splits = []
                with open(args.prompts_path, "r") as f:
                    for i in f:
                        i = i.strip()
                        jsonl_splits.append(pd.read_json(i, lines=True))
                df = pd.concat(jsonl_splits)
            else:
                df = pd.read_json(args.prompts_path, lines=True)
            prompt_embeddings, df = embed_text(embedding_style, df, base_dir)

            # Embed desires if needed
            if args.desire_path is not None:
                desires_path = args.desire_path
                desires_df = pd.read_json(desires_path, lines=True)
                desire_embeddings, desires_df = embed_text(embedding_style, desires_df, base_dir)
            else:
                desire_embeddings = None

        # Loop over clustering method
        for clustering_method in args.clustering_methods:
            out_dir = os.path.join(base_dir, clustering_method)
            raw_data_dir = os.path.join(out_dir, "raw_data")
            print(f"\nBeginning to cluster {embedding_style} embeddings using {clustering_method}.")
            df_embedding = df.copy()
            cluster_labels = cluster(method=clustering_method, prompt_embeddings=prompt_embeddings)
            clusters = np.unique(cluster_labels).tolist()
            df_embedding["Cluster_label"] = cluster_labels

            # Saving cluster data
            df_embedding.to_csv(os.path.join(raw_data_dir, "labeled_original_data.csv"))
            for i, id in enumerate(clusters):
                df_embedding[df_embedding["Cluster_label"] == id].to_csv(os.path.join(raw_data_dir, f"cluster_{i}.csv"))
            print(f"{len(clusters)} clusters identified. Vector Embedding clusters saved to {raw_data_dir}.")

            dim = 2
            if args.visualize_UMAP is True:
                print("Visualizing 2d clusters using UMAP.")
                visualize(prompt_embeddings, out_dir, df_embedding, clusters, model_type="umap", dim=dim,
                          desires=desire_embeddings)
            if args.visualize_TSNE is True:
                print("Visualizing 2d clusters using TSNE.")
                visualize(prompt_embeddings, out_dir, df_embedding, clusters, model_type="tsne", dim=dim,
                          desires=desire_embeddings)

            if args.visualize_3d:
                dim = 3
                if args.visualize_UMAP is True:
                    print("Visualizing 3d clusters using UMAP.")
                    visualize(prompt_embeddings, out_dir, df_embedding, clusters, model_type="umap", dim=dim,
                              desires=desire_embeddings)
                if args.visualize_TSNE is True:
                    print("Visualizing 3d clusters using TSNE.")
                    visualize(prompt_embeddings, out_dir, df_embedding, clusters, model_type="tsne", dim=dim,
                              desires=desire_embeddings)

    print("Tasks completed.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog="Prompt Visualization"
    )
    parser.add_argument("-n", "--clustering_methods", nargs='+', default=["HDBSCAN", "DBSCAN", "MeanShift"],
                        help="Select which clustering methods to visualize. Currently, the supported clustering methods are DBSCAN, HDBSCAN, MeanShift.")
    parser.add_argument("-p", "--prompts_path",
                        help="Assumes this is a dataframe with the prompts saved under \"text\".")
    parser.add_argument("-s", "--embedding_styles", default=["sbert", "bert", "semantic", "mixed"], nargs="+",
                        help="Choose between semantic embedding (BERT or sBERT), stylistic embedding (unimplemented), or concatenated stylistic and semantic embeddings (unimplemented).")
    parser.add_argument("-c", "--embedded_prompts_paths", nargs="+", default=None,
                        help="Loads in a jsonl with precomputed prompt embeddings saved in the \"prompt_embedding\" column")
    parser.add_argument("-o", "--output_directory")
    parser.add_argument("--desire_path", help="Provides the location of a jsonl with desires.")
    parser.add_argument("-v", "--visualize_TSNE", default=True,
                        help="Visualize the clusters using TSNE dimensionality reduction.", type=bool)
    parser.add_argument("-u", "--visualize_UMAP", default=True,
                        help="Visualize the clusters using UMAP dimensionality reduction.", type=bool)
    parser.add_argument("-d", "--visualize_3d", default=True, help="View 3D visualizations of the clusters", type=bool)
    args = parser.parse_args()
    main(args)
