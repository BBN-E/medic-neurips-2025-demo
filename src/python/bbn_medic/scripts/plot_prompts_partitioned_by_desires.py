import numpy as np
import argparse
import pandas as pd
import os
from matplotlib.lines import Line2D
import random
from sklearn.manifold import TSNE
import torch
from transformers import BertTokenizer, BertModel
from matplotlib import pyplot as plt
from itertools import islice
from tqdm import tqdm
import ast
from collections import defaultdict

import seaborn as sns
import colorcet as cc
from collections import Counter
import time

import warnings
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

def generate_hiatus_compatible_file(df, raw_data_path):
    """Copies the original data into a new jsonl that is compatible with
    the HIATUS stylistic embedding script's expected format."""
    # copying original file into HIATUS compatible format inside raw_data
    temp_df = df.copy()
    temp_df["fullText"] = temp_df["text"] # requires "text" column => "fullText" column
    temp_df["documentID"] = temp_df["id"] # "id" column => "documentID" column

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
    temp_embeddings = np.empty([0,hidden_dim])

    for batch in tqdm(batched(df["text"], 256), total=len(df["text"])//256+1):
        encoding = tokenizer.batch_encode_plus( batch,
        padding=True,              # Pad to the maximum sequence length
        truncation=True,           # Truncate to the maximum sequence length if necessary
        return_tensors='pt',       # Return PyTorch tensors
        add_special_tokens=True    # Add special tokens CLS and SEP
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
    temp_embeddings = np.empty([0,hidden_size])

    for batch in tqdm(batched(df["text"], 256), total=len(df["text"])//256+1):
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
    import json
    from bbn_alert.common.data_types import ModelMode
    from bbn_alert.common.logger import logger
    from bbn_alert.feature_space_generation.driver import Driver

    params = {
        "data": {
            "datasets":[
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

        prompt_embeddings = np.concatenate((stylistic_embeddings, semantic_embeddings), axis=1)
        df["prompt_embeddings"] = [i.tolist() for i in list(prompt_embeddings)]
    else:
        raise ValueError("Select a embedding style from \"stylistic\", \"bert\", \"sbert\", and \"mixed\". Mixed denotes stylistic embeddings concatenated with sbert embeddings. ")
    return prompt_embeddings, df

def visualize(embeddings, out_dir, df, args, model_type="umap", dim=2, desires=None, label=-1, mark_styles=None):
    """Applies model_type(UMAP or TSNE) to reduce embeddings to dim dimensional space. Then, 
    the clusters are graphed onto a scatterplot with reduced desire embeddings optionally 
    plotted as well. """
    num_prompts = len(embeddings)
    if desires is not None:
        embeddings = np.concatenate([embeddings, desires], axis=0)
    if model_type.lower() == "umap":
        import umap
        reducer = umap.UMAP(random_state=42, n_components=dim)
        embs = reducer.fit_transform(embeddings)
        embs, prompt_embs = embs[:num_prompts], embs[num_prompts:]
    elif model_type.lower() == "tsne":
        tsne = TSNE(random_state=42, max_iter=1000, metric="euclidean", n_components=dim, perplexity=min(len(embeddings)-1, 30))
        embs = tsne.fit_transform(embeddings)
        embs, prompt_embs = embs[:num_prompts], embs[num_prompts:]
    else:
        raise ValueError("Unknown modeling type")
    
    if desires is not None:
        assert len(prompt_embs) > 0, "There should be reduced desire embeddings if desire embeddings are passed into this function"

    if dim == 2:
        values = np.hsplit(embs, 1+np.arange(embs.shape[1]-1)) # splitting into [x, y]
        assert len(values) == 2
        df["x"] = values[0]
        df["y"] = values[1]
    elif dim == 3:
        values = np.hsplit(embs, 1+np.arange(embs.shape[1]-1)) # splitting into [x, y, z]
        assert len(values) == 3
        df["x"] = values[0]
        df["y"] = values[1]
        df["z"] = values[2]
    else:
        raise ValueError("Incorrect dimension parameter. The visualization dimension should either be 2 or 3.")

    name = f"{model_type}_{dim}d"
    # the prompt embedding split
    desire_values = np.hsplit(prompt_embs, 1+np.arange(prompt_embs.shape[1]))

    if mark_styles != None:
        try: 
            if len(mark_styles) != 2:
                mark_styles = None
        except:
            mark_styles = None
    
    if model_type=="tsne":
        model_type = "t-SNE"
    elif model_type=="umap":
        model_type = "UMAP"

    plot(out_dir, values, df, name, args, desire_values=desire_values, label=label, mark_styles=mark_styles, model_type=model_type)


def plot(out_dir, values, df, name, args, desire_values=None, label=-1, mark_styles=None, model_type=None):
    """Creates plots of the 2D or 3D reduced embedding data. Creates 2 PDF images under
    args.output_directory/desires: The first is a complete visualization of all data
    and the second partitions the data by cluster id. """

    plt.figure(dpi=1200)

    
    desire_df = pd.read_json(args.desire_path, lines=True)
    style_markers = np.full(len(values[0].squeeze()), 'o', dtype=str) # Default marker

    # Group data indices by desire
    indices_by_desire = defaultdict(list)
    for i, item in enumerate(df["metadata"]):
        desire_id = get_desire_source_id(item)
        indices_by_desire[desire_id].append(i)
        if mark_styles is not None:
            # Assign markers based on style
            if mark_styles[0] in item["style"]:
                style_markers[i] = "D" 
            elif mark_styles[1] in item["style"]:
                style_markers[i] = "s"

    x = pd.Series(values[0].squeeze())
    y = pd.Series(values[1].squeeze())
    
    unique_desires_in_data = sorted(list(indices_by_desire.keys()))
    num_desires = len(unique_desires_in_data)
    cmap = sns.color_palette(cc.glasbey, n_colors=num_desires)
    
    # Map desire IDs to colors for consistent plotting
    color_map = {desire_id: cmap[i] for i, desire_id in enumerate(unique_desires_in_data)}

    # Iterate through desires to plot data points
    for desire_id, indices in indices_by_desire.items():
        color = color_map[desire_id]
        
        # Further subset by the marker type for this desire group
        subset_markers = style_markers[indices]
        styles_found = False
        for marker_style in ['o', 's', 'D']:
            # Get the final indices for this specific color and marker
            marker_mask = (subset_markers == marker_style)
            final_indices = np.array(indices)[marker_mask]
            
            # Plot the specific subset
            if marker_style != 'o':
                styles_found=True
                plt.scatter(x[final_indices], y[final_indices], color='none', s=25, marker=marker_style, alpha=1, edgecolors='black',linewidths=0.4)
                plt.scatter(x[final_indices], y[final_indices], color=color, s=25, marker=marker_style, alpha=0.2, edgecolors='black',linewidths=0.4)
            
            else:
                plt.scatter(x[final_indices], y[final_indices], color=color, s=25, marker=marker_style, alpha=0.2, edgecolors='none')

    # Plot the corresponding desire 'X' markers afterwards so that they are on top
    if desire_values is not None and len(desire_values) > 0:
        for i, desire_id in enumerate(unique_desires_in_data):
            row_mask = desire_df["id"] == desire_id
            if not row_mask.any(): continue
            desire_embedding_index = desire_df.loc[row_mask].index[0]
            
            plt.scatter(
                desire_values[0][desire_embedding_index], 
                desire_values[1][desire_embedding_index],
                color=color_map[desire_id],
                alpha=1,
                marker='X',
                s=100,
                edgecolors='black',
                linewidths=0.75
            )


    # Legend creation
    legend_handles = []

    # 1. Add the "Desires" title to the top of the legend
    legend_handles.append(Line2D([0], [0], marker='', color='w', label='Desires', markersize=0, linestyle='None'))

    # 2. Add handles for the actual desires (colors)
    for desire_id, color in color_map.items():
        label_text = desire_df.loc[desire_df["id"] == desire_id, "text"].iloc[0]
        legend_handles.append(Line2D([0], [0], marker='o', color='w', label=label_text,
                                        markerfacecolor=color, markersize=8, linestyle='None'))
    if mark_styles != None and styles_found:
        print("Not adding styles to the legend since there are no prompts in the specified styles.")
        # 3. Add a separator and title for styles
        legend_handles.append(Line2D([0], [0], marker='', color='w', label='\nStyles', markersize=0, linestyle='None'))

        # 4. Add handles for the styles (markers)
        legend_handles.append(Line2D([0], [0], marker='D', color='grey', label=mark_styles[0], markersize=7, linestyle='None'))
        legend_handles.append(Line2D([0], [0], marker='s', color='grey', label=mark_styles[1], markersize=7, linestyle='None'))
        legend_handles.append(Line2D([0], [0], marker='o', color='grey', label='Other', markersize=7, linestyle='None'))

    # 5. Add handle for the large 'Desire' marker
    legend_handles.append(Line2D([0], [0], marker='X', color='grey', label='Desire', markersize=9, linestyle='None', markeredgecolor='black'))

    # Create the final legend
    leg = plt.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1.02, 1.0), fancybox=True)

    # Make the legend markers opaque
    for lh in leg.legend_handles:
        lh.set_alpha(1)

    plt.title(f"{model_type} Projection of sBERT Prompt Embeddings Partitioned By Desires", loc='left')
    
    plt.savefig(os.path.join(out_dir, f"{name}_disorders_partition_{label}.pdf"), bbox_inches='tight')
    plt.clf()



def get_desire_source_id(dict): 
    if dict["style"]=="standard":
        return dict["Desire_source_id"]
    else:
        if "Prompt_metadata" in dict.keys():
            return dict["Prompt_metadata"]["Desire_source_id"]
        else:
            raise ValueError(f"desire source id not found in: {dict}")


def main(args):
    timing_list = []
    timing_list.append(time.time())
    print(f"\n====================================================== \
            \nEmbedding and plotting desires of sbert embeddings.")
    
    # This loop just creates directories
    out_dir = os.path.join(args.output_directory, "desire_visualization")
    for i in range(int(args.n_plots)):
        raw_data_dir = os.path.join(out_dir, f"raw_data_{i}")
        os.makedirs(raw_data_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    random.seed(42)
    data_path = args.labeled_original_data[0]

    all_cluster_data = pd.read_csv(data_path)
    meta = [ast.literal_eval(i) for i in all_cluster_data["metadata"]]
    all_cluster_data["metadata"] = meta
    
    print("Extracting embeddings from data file...")
    embeddings = [ast.literal_eval(i) for i in all_cluster_data["prompt_embeddings"].tolist()]

    n_samples_required = int(args.n_samples) # default 50
    n_desires = int(args.n_desires) # default 16
    n_plots = int(args.n_plots) # default 5

    for label in range(n_plots):
        raw_data_dir = os.path.join(out_dir, f"raw_data_{label}")
        print("Subsetting data for desire-based visualization BEFORE dimensionality reduction.")

        # Get desire IDs for each row from metadata
        collection_desire_repeats = [get_desire_source_id(m) for m in all_cluster_data["metadata"]]
        desire_counts = Counter(collection_desire_repeats)

        # Find desires that meet the sample count requirement
        eligible_desires = [
            desire_id for desire_id, count in desire_counts.items()
            if count >= n_samples_required
        ]

        # Randomly select n_desires of these eligible desires to plot
        if len(eligible_desires) >= n_desires:
            subsample_desires = random.sample(eligible_desires, n_desires)
            break_after_one_plot = False
        else:
            print(f"Warning: Found only {len(eligible_desires)} desires with at least {n_samples_required} samples. Using all of them.")
            subsample_desires = [desire_id for desire_id, count in desire_counts.items()]
            # Only break if there are not enough plots, so there is only 1 generated plot.
            # The rest of the graphs will be identical because of random seeds.
            break_after_one_plot = True
        
        indices_by_desires = defaultdict(list)
        for i, m in enumerate(all_cluster_data["metadata"]):
            indices_by_desires[get_desire_source_id(m)].append(i)

        desire_df = pd.read_json(args.desire_path, lines=True)

        indices_to_keep = []
        for desire_id in subsample_desires:
            # For each chosen desire, sample 50 indices
            if len(indices_by_desires[desire_id])>=n_samples_required:
                sampled_indices = random.sample(indices_by_desires[desire_id], n_samples_required)
            else: 
                sampled_indices = list(range(len(indices_by_desires[desire_id])))
            indices_to_keep.extend(sampled_indices)

            row_mask = desire_df["id"] == desire_id
            true_label = desire_df.loc[row_mask, "text"].iloc[0]
            all_cluster_data.iloc[sampled_indices].to_csv(os.path.join(raw_data_dir, "_".join(true_label.split())+".csv"))
        
        # Create the final subset of data and labels for visualization
        embeddings_to_visualize = np.array(embeddings)[indices_to_keep]
        labels_to_visualize = all_cluster_data.iloc[indices_to_keep].reset_index(drop=True)


        if args.desire_path:
            desires_df = pd.read_json(args.desire_path, lines=True)
            desire_embeddings, _ = embed_text("sbert", desires_df, out_dir)
        else:
            desire_embeddings = None
            
        timing_list.append(time.time())
        print(f"Loading and subsetting time is: {(timing_list[-1]-timing_list[-2]):.2f} seconds")


        # tsne
        print("Running t-SNE...")
        visualize(embeddings_to_visualize, out_dir, labels_to_visualize, args, model_type="tsne", dim=2, desires=desire_embeddings, label=label, mark_styles=args.styles_to_mark)
        timing_list.append(time.time())
        print(f"t-SNE and plotting took: {(timing_list[-1]-timing_list[-2]):.2f} seconds")

        # umap
        print("Running UMAP...")
        visualize(embeddings_to_visualize, out_dir, labels_to_visualize, args, model_type="umap", dim=2, desires=desire_embeddings, label=label, mark_styles=args.styles_to_mark)
        timing_list.append(time.time())
        print(f"UMAP and plotting finished in {(timing_list[-1]-timing_list[-2]):.2f} seconds")
        
        # break if there are not enough desires:
        if break_after_one_plot:
            break
    timing_list.append(time.time())
    print(f"Script finished in {(timing_list[-1]-timing_list[0]):.2f} seconds.")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog="Prompt Visualization"
    )
    
    parser.add_argument("-p", "--prompts_path", help="Assumes this is a dataframe with the prompts saved under \"text\".")
    parser.add_argument("-c", "--labeled_original_data", nargs="+", default=None, help="path with all cluster_id csv files.")
    parser.add_argument("-o", "--output_directory")
    parser.add_argument("--desire_path", help="Provides the location of a jsonl with desires.")

    parser.add_argument("--n_samples", default=50, help="Provide the number of prompts per desire to plot")
    parser.add_argument("--n_desires", default=16, help="Provide the number of desires to visualize")
    parser.add_argument("--n_plots", default=10, help="Provide the number of graphs to plot")
    parser.add_argument("--styles_to_mark", default=None, nargs="*", help="Pass in 2 styles to visualize. These must be the exact names of the style")
    
    args = parser.parse_args()
    main(args)