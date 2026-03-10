# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from enum import Enum
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from accelerate.utils import gather_object
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss, MarginRankingLoss
from transformers.utils import strtobool

from typing import List, Dict, Optional
import json


def cross_entropy_loss_func(outputs, labels, num_items_in_batch=None, **kwargs):
    # You need to return a scalar representing the loss.
    from swift.trainers import per_token_loss_func
    token_loss = per_token_loss_func(outputs, labels)
    if num_items_in_batch is None:
        num_items_in_batch = (labels[:, 1:] != -100).sum()
    return token_loss.sum() / num_items_in_batch


def _parse_pair_sentence(outputs):
    if isinstance(outputs, dict):
        last_hidden_state = outputs['last_hidden_state']
    else:
        last_hidden_state = outputs
    batch_size = last_hidden_state.shape[0]
    shape_len = len(last_hidden_state.shape)
    first_sentence = list(range(0, batch_size, 2))
    second_sentence = list(range(1, batch_size, 2))
    if shape_len == 3:
        sentence1 = last_hidden_state[first_sentence][:, 0].squeeze(dim=1)
        sentence2 = last_hidden_state[second_sentence][:, 0].squeeze(dim=1)
    else:
        sentence1 = last_hidden_state[first_sentence]
        sentence2 = last_hidden_state[second_sentence]
    return sentence1, sentence2


# Code borrowed from sentence_transformers
class SiameseDistanceMetric(Enum):
    """The metric for the contrastive loss"""

    EUCLIDEAN = lambda x, y: F.pairwise_distance(x, y, p=2)  # noqa
    MANHATTAN = lambda x, y: F.pairwise_distance(x, y, p=1)  # noqa
    COSINE_DISTANCE = lambda x, y: 1 - F.cosine_similarity(x, y)  # noqa


def cosine_similarity_func(outputs, labels, loss_scale=None, num_items_in_batch=None, **kwargs) -> torch.Tensor:
    cos_score_transformation = nn.Identity()
    loss_fct = MSELoss()
    sentence1, sentence2 = _parse_pair_sentence(outputs)
    output = cos_score_transformation(torch.cosine_similarity(sentence1, sentence2))
    return loss_fct(output, labels.to(output.dtype).view(-1))


def contrastive_loss(outputs, labels, loss_scale=None, num_items_in_batch=None, **kwargs) -> torch.Tensor:
    sentence1, sentence2 = _parse_pair_sentence(outputs)
    distance_metric = SiameseDistanceMetric.COSINE_DISTANCE
    distances = distance_metric(sentence1, sentence2)
    margin = 0.5
    labels = labels.to(sentence1.dtype)
    losses = 0.5 * (labels * distances.pow(2) + (1 - labels) * F.relu(margin - distances).pow(2))
    return losses.mean()


def calculate_paired_metrics(embeddings, labels):
    from sklearn.metrics.pairwise import (paired_cosine_distances, paired_euclidean_distances,
                                          paired_manhattan_distances)
    from scipy.stats import pearsonr, spearmanr

    embeddings1, embeddings2 = _parse_pair_sentence(embeddings)
    cosine_scores = 1 - (paired_cosine_distances(embeddings1, embeddings2))
    manhattan_distances = -paired_manhattan_distances(embeddings1, embeddings2)
    euclidean_distances = -paired_euclidean_distances(embeddings1, embeddings2)
    dot_products = [np.dot(emb1, emb2) for emb1, emb2 in zip(embeddings1, embeddings2)]

    eval_pearson_cosine, _ = pearsonr(labels, cosine_scores)
    eval_spearman_cosine, _ = spearmanr(labels, cosine_scores)

    eval_pearson_manhattan, _ = pearsonr(labels, manhattan_distances)
    eval_spearman_manhattan, _ = spearmanr(labels, manhattan_distances)

    eval_pearson_euclidean, _ = pearsonr(labels, euclidean_distances)
    eval_spearman_euclidean, _ = spearmanr(labels, euclidean_distances)

    eval_pearson_dot, _ = pearsonr(labels, dot_products)
    eval_spearman_dot, _ = spearmanr(labels, dot_products)

    return {
        'pearson_cosine': eval_pearson_cosine,
        'pearson_euclidean': eval_pearson_euclidean,
        'pearson_manhattan': eval_pearson_manhattan,
        'pearson_dot_product': eval_pearson_dot,
        'spearman_cosine': eval_spearman_cosine,
        'spearman_euclidean': eval_spearman_euclidean,
        'spearman_manhattan': eval_spearman_manhattan,
        'spearman_dot_product': eval_spearman_dot,
    }


def calculate_infonce_metrics(embeddings, labels):
    hard_negatives = os.environ.get('INFONCE_HARD_NEGATIVES', None)
    use_batch = strtobool(os.environ.get('INFONCE_USE_BATCH', 'True'))
    if hard_negatives is not None:
        hard_negatives = int(hard_negatives)
    split_tensors = _parse_multi_negative_sentences(torch.tensor(embeddings), torch.tensor(labels), hard_negatives)
    split_tensors = [t.numpy() for t in split_tensors]
    can_batched = hard_negatives is not None
    if hard_negatives is None and len(set([s.shape[0] for s in split_tensors])) == 1:
        can_batched = True
    all_similarity_matrix = []
    all_labels = []
    pos_neg_margins = []
    if not use_batch:
        if can_batched:
            sentences = np.stack(split_tensors, axis=0)
            similarity_matrix = np.matmul(sentences[:, 0:1], sentences[:, 1:].transpose((0, 2, 1))).squeeze(1)
            all_similarity_matrix.append(similarity_matrix)
            labels = np.zeros_like(similarity_matrix)
            labels[:, 0] = 1
            all_labels.append(labels)
        else:
            for tensor in split_tensors:
                similarity_matrix = np.matmul(tensor[0], tensor[1:].T)
                all_similarity_matrix.append(similarity_matrix)
                labels = np.zeros_like(similarity_matrix)
                labels[0] = 1
                all_labels.append(labels)
                max_neg_scores = np.max(similarity_matrix[labels == 0], axis=-1)
                pos_neg_margins.append(np.mean(similarity_matrix[labels == 1] - max_neg_scores).item())
    else:
        if can_batched:
            sentences = np.stack(split_tensors, axis=0)
            similarity_matrix = np.matmul(sentences[:, 0], sentences[:, 1:].reshape(-1, sentences.shape[2]).T)
            all_similarity_matrix.append(similarity_matrix)
            labels = np.zeros_like(similarity_matrix)
            for row, col in enumerate(range(0, sentences.shape[0] * (sentences.shape[1] - 1), sentences.shape[1] - 1)):
                labels[row, col] = 1
            all_labels.append(labels)
        else:
            all_tensors = []
            for tensor in split_tensors:
                all_tensors.append(tensor[1:])
            sentences = np.concatenate(all_tensors, axis=0)
            length = 0
            for idx, tensor in enumerate(split_tensors):
                similarity_matrix = np.matmul(tensor[0], sentences.T)
                all_similarity_matrix.append(similarity_matrix)
                labels = np.zeros_like(similarity_matrix)
                labels[length] = 1
                all_labels.append(labels)
                length += tensor.shape[0] - 1
                max_neg_scores = np.max(similarity_matrix[labels == 0], axis=-1)
                pos_neg_margins.append(np.mean(similarity_matrix[labels == 1] - max_neg_scores).item())

    similarity_matrix = np.concatenate(all_similarity_matrix, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    if can_batched:
        pos_scores = similarity_matrix[labels == 1].reshape(similarity_matrix.shape[0], -1)
        neg_scores = similarity_matrix[labels == 0].reshape(similarity_matrix.shape[0], -1)
        max_neg_scores = np.max(neg_scores, axis=-1)
        pos_neg_margin = np.mean(pos_scores - max_neg_scores).item()
    else:
        pos_scores = similarity_matrix[labels == 1]
        neg_scores = similarity_matrix[labels == 0]
        pos_neg_margin = np.mean(pos_neg_margins)

    mean_neg = np.mean(neg_scores)
    mean_pos = np.mean(pos_scores)
    return {'margin': pos_neg_margin, 'mean_neg': mean_neg, 'mean_pos': mean_pos}


def calculate_reranker_metrics(logits, labels):
    """
    Calculate MRR and NDCG metrics for reranker.

    This function first groups the data based on query boundaries (identified by
    positive samples), then calculates MRR and NDCG for each group independently,
    and finally returns the mean across all queries.

    Data format:
    - Each query group starts with a positive sample (label=1) followed by negatives (label=0)
    - Example: [1,0,0,1,0,0,0] represents 2 queries: query1=[1,0,0], query2=[1,0,0,0]

    Args:
        logits: Model output scores [batch_size] (numpy array or can be converted to numpy)
        labels: Binary labels (1 for positive, 0 for negative) [batch_size]

    Returns:
        dict: Dictionary containing MRR and NDCG metrics averaged across all queries
    """
    import numpy as np

    # Convert to numpy if needed
    if hasattr(logits, 'numpy'):
        logits = logits.numpy()
    if hasattr(labels, 'numpy'):
        labels = labels.numpy()

    logits = np.array(logits).flatten()
    labels = np.array(labels).flatten()

    # Step 1: Find all positive sample indices (query boundaries)
    positive_indices = np.where(labels == 1)[0]

    if len(positive_indices) == 0:
        return {'mrr': 0.0, 'ndcg': 0.0}

    # Step 2: Split into groups (queries)
    query_groups = []
    for i, pos_idx in enumerate(positive_indices):
        # Each group starts at a positive index
        group_start = pos_idx

        # Group ends at the next positive index or end of data
        if i + 1 < len(positive_indices):
            group_end = positive_indices[i + 1]
        else:
            group_end = len(labels)

        # Extract this query's data
        query_logits = logits[group_start:group_end]
        query_labels = labels[group_start:group_end]

        query_groups.append((query_logits, query_labels))

    # Step 3: Calculate metrics for each query independently
    mrr_scores = []
    ndcg_scores = []

    for query_idx, (query_logits, query_labels) in enumerate(query_groups):
        # Skip groups that are too small (need at least 1 positive + 1 negative)
        if len(query_logits) < 2:
            print(f'Query {query_idx}: Skipped (too small: {len(query_logits)} items)')
            continue

        # Verify that the first sample is positive (data format validation)
        if query_labels[0] != 1:
            print(f'Query {query_idx}: Skipped (first sample not positive)')
            continue

        # Step 3a: Calculate ranking within this query
        ranking = np.argsort(-query_logits)  # Sort by logits descending

        # Step 3b: Find position of positive document (should be at index 0 in query)
        pos_rank = np.where(ranking == 0)[0][0] + 1  # +1 for 1-based ranking

        # Step 3c: Calculate MRR for this query
        mrr = 1.0 / pos_rank
        mrr_scores.append(mrr)

        # Step 3d: Calculate NDCG for this query
        def calculate_ndcg_single_query(relevance_scores, ranking):
            """Calculate NDCG for a single query"""
            # Calculate DCG (Discounted Cumulative Gain)
            dcg = 0.0
            for rank_pos, doc_idx in enumerate(ranking):
                relevance = relevance_scores[doc_idx]
                dcg += (2**relevance - 1) / np.log2(rank_pos + 2)  # rank_pos+2 because log2(1) undefined

            # Calculate IDCG (Ideal DCG)
            ideal_relevance = np.sort(relevance_scores)[::-1]  # Sort relevance descending
            idcg = 0.0
            for rank_pos, relevance in enumerate(ideal_relevance):
                idcg += (2**relevance - 1) / np.log2(rank_pos + 2)

            # NDCG = DCG / IDCG
            if idcg == 0:
                return 0.0
            return dcg / idcg

        # Create relevance scores (1 for positive, 0 for negative)
        relevance_scores = query_labels.astype(float)
        ndcg = calculate_ndcg_single_query(relevance_scores, ranking)
        ndcg_scores.append(ndcg)

    # Step 4: Calculate mean metrics across all valid queries
    if len(mrr_scores) == 0:
        print('No valid queries found for metric calculation')
        return {'mrr': 0.0, 'ndcg': 0.0}

    mean_mrr = np.mean(mrr_scores)
    mean_ndcg = np.mean(ndcg_scores)

    return {
        'mrr': mean_mrr,
        'ndcg': mean_ndcg,
    }


def _parse_multi_negative_sentences(sentences, labels, hard_negatives=None):
    split_indices = torch.nonzero(labels, as_tuple=False).squeeze().tolist()
    if isinstance(split_indices, int):
        split_indices = [split_indices]
    split_indices.append(len(labels))
    split_indices = np.array(split_indices) + np.array(list(range(len(split_indices))))
    split_tensors = []

    for i in range(len(split_indices) - 1):
        start = split_indices[i]
        end = split_indices[i + 1]
        split_part = sentences[start:end]
        if hard_negatives is not None:
            negatives = len(split_part) - 2
            assert negatives > 0
            if negatives > hard_negatives:
                split_part = split_part[:hard_negatives + 2]
            elif negatives < hard_negatives:
                selected = np.random.choice(list(range(negatives)), size=hard_negatives - negatives, replace=True)
                selected += 1  # skip positive
                split_part = torch.cat((split_part, split_part[selected]), dim=0)
        split_tensors.append(split_part)
    return split_tensors


def infonce_loss(outputs, labels, loss_scale=None, num_items_in_batch=None, **kwargs) -> torch.Tensor:
    temperature = float(os.environ.get('INFONCE_TEMPERATURE', '0.01'))  # temperature
    # calculate CE across the batch, meaning all samples will be negative except the matching positive
    use_batch = strtobool(os.environ.get('INFONCE_USE_BATCH', 'True'))
    hard_negatives = os.environ.get('INFONCE_HARD_NEGATIVES', None)  # how many negative prompts kept in one sample
    # mask out fake negatives
    infonce_mask_fake_negative = strtobool(os.environ.get('INFONCE_MASK_FAKE_NEGATIVE', 'False'))
    if hard_negatives is not None:
        hard_negatives = int(hard_negatives)
    from swift.utils import get_dist_setting
    rank, _, world_size, _ = get_dist_setting()
    # repeat of anchor(1)+positive(1)+negatives(n)
    sentences = outputs['last_hidden_state']

    if world_size > 1 and use_batch:
        # gather all the sentences and labels across the gpus when calculate loss across all batches of all gpus
        all_sentences = gather_object(sentences.unsqueeze(0))
        labels = gather_object(labels)
        # override the gathered one
        all_sentences[rank] = sentences
        for idx in range(len(all_sentences)):
            if idx == rank:
                continue
            # we don't calculate grad from other gpus
            all_sentences[idx] = all_sentences[idx].detach().to(sentences.device)
        sentences = torch.cat(all_sentences, dim=0)
        labels = [tensor.to(sentences.device) for tensor in labels]
        labels = torch.stack(labels, dim=0)

    # split tensors into single sample
    # for example: batch_size=2 with tensor anchor(1)+positive(1)+negatives(3) + anchor(1)+positive(1)+negatives(2)
    # labels will be [1,0,0,0,1,0,0], meaning 1 positive, 3 negatives, 1 positive, 2 negatives
    split_tensors = _parse_multi_negative_sentences(sentences, labels, hard_negatives)
    loss = 0
    can_batched = hard_negatives is not None
    if hard_negatives is None and len(set([s.shape[0] for s in split_tensors])) == 1:
        # all tensors have the same batch size
        can_batched = True
    if not use_batch:
        # only calculate loss inside one sample
        if can_batched:
            # negative numbers are equal
            # [B, neg+2, D]
            sentences = torch.stack(split_tensors, dim=0)
            # [B, 1, D] * [B, neg+1, D]
            similarity_matrix = torch.matmul(sentences[:, 0:1], sentences[:, 1:].transpose(1, 2)) / temperature
            # The positive one is the first element
            labels = torch.zeros(len(split_tensors), dtype=torch.int64).to(sentences.device)
            loss = nn.CrossEntropyLoss()(similarity_matrix.squeeze(1), labels)
        else:
            # the negative numbers may be different, use for loop
            for tensor in split_tensors:
                # [D] * [neg+1, D]
                similarity_matrix = torch.matmul(tensor[0], tensor[1:].T) / temperature
                # The positive one is the first element
                labels = torch.tensor(0).to(tensor.device)
                loss += nn.CrossEntropyLoss()(similarity_matrix, labels)
            # avg between all batches in one gpu
            loss /= len(split_tensors)
    else:

        def mask_fake_negative(sim_matrix, sim_labels):
            thresholds = sim_matrix[torch.arange(sim_matrix.size(0)), sim_labels].view(-1, 1) + 0.1
            thresholds = thresholds.detach()
            mask = sim_matrix > thresholds
            sim_matrix[mask] = float('-inf')

        if can_batched:
            # [B, neg+2, D]
            sentences = torch.stack(split_tensors, dim=0)
            # [B, D] * [B*(neg+1), D]
            similarity_matrix = torch.matmul(sentences[:, 0].squeeze(1), sentences[:,
                                                                                   1:].reshape(-1, sentences.size(2)).T)
            labels = torch.tensor(range(0,
                                        sentences.size(0) * (sentences.size(1) - 1),
                                        sentences.size(1) - 1)).view(-1).to(sentences.device)
            if infonce_mask_fake_negative:
                mask_fake_negative(similarity_matrix, labels)
            similarity_matrix = similarity_matrix / temperature
            # every neg+1 is positive start from 0
            loss = nn.CrossEntropyLoss()(similarity_matrix, labels) / world_size  # avoid duplicate
        else:
            all_tensors = []
            for tensor in split_tensors:
                all_tensors.append(tensor[1:])
            # cat all neg+1 tensors
            sentences = torch.cat(all_tensors, dim=0)
            length = 0
            for idx, tensor in enumerate(split_tensors):
                # [D] * [B*(neg+1), D], neg numbers are different
                similarity_matrix = torch.matmul(tensor[0], sentences.T) / temperature
                labels = torch.tensor(length).to(tensor.device)
                loss += nn.CrossEntropyLoss()(similarity_matrix, labels)
                # next positive is neg+1
                length += tensor.size(0) - 1
            loss /= len(split_tensors)
            loss /= world_size  # avoid duplicate
    return loss


def online_contrastive_loss(outputs, labels, loss_scale=None, num_items_in_batch=None, **kwargs) -> torch.Tensor:
    sentence1, sentence2 = _parse_pair_sentence(outputs)
    distance_metric = SiameseDistanceMetric.COSINE_DISTANCE
    distance_matrix = distance_metric(sentence1, sentence2)
    negs = distance_matrix[labels == 0]
    poss = distance_matrix[labels == 1]

    # select hard positive and hard negative pairs
    negative_pairs = negs[negs < (poss.max() if len(poss) > 1 else negs.mean())]
    positive_pairs = poss[poss > (negs.min() if len(negs) > 1 else poss.mean())]

    positive_loss = positive_pairs.pow(2).sum()
    margin = 0.5
    negative_loss = F.relu(margin - negative_pairs).pow(2).sum()
    loss = positive_loss + negative_loss
    return loss


def reranker_loss(outputs, labels, loss_scale=None, num_items_in_batch=None, **kwargs) -> torch.Tensor:
    logits = outputs.logits
    logits = logits.squeeze(1)
    labels = labels.to(logits.dtype)
    loss_fct = nn.BCEWithLogitsLoss()
    loss = loss_fct(logits, labels)
    return loss


def generative_reranker_loss(outputs,
                             labels,
                             loss_scale=None,
                             num_items_in_batch=None,
                             trainer=None,
                             **kwargs) -> torch.Tensor:
    """
    Generative reranker loss function.

    This loss function is designed for generative rerankers that use token probabilities
    (e.g., "yes"/"no") to determine relevance scores. It only computes loss on the
    last token position for specific tokens.

    Args:
        outputs: Model outputs containing logits
        labels: Binary labels (0/1) for irrelevant/relevant pairs
        loss_scale: Not used for generative reranker
        num_items_in_batch: Not used for generative reranker
        trainer: Trainer instance to access tokenizer

    Returns:
        torch.Tensor: Cross entropy loss for yes/no classification
    """
    if trainer is None:
        raise ValueError('trainer is required for generative_reranker_loss to access tokenizer')

    logits = outputs.logits
    tokenizer = trainer.processing_class

    # Get token IDs for positive and negative tokens
    # Default to "yes"/"no", but can be configured via environment variables
    positive_token = os.environ.get('GENERATIVE_RERANKER_POSITIVE_TOKEN', 'yes')
    negative_token = os.environ.get('GENERATIVE_RERANKER_NEGATIVE_TOKEN', 'no')

    try:
        positive_token_id = tokenizer.convert_tokens_to_ids(positive_token)
        negative_token_id = tokenizer.convert_tokens_to_ids(negative_token)
    except Exception as e:
        raise ValueError(f"Failed to convert tokens '{positive_token}'/'{negative_token}' to IDs. "
                         f'Please check if these tokens exist in the tokenizer vocabulary. Error: {e}')

    # Extract logits for positive and negative tokens directly from last position
    # This avoids creating the large intermediate tensor last_logits
    positive_logits = logits[:, -1, positive_token_id]  # [batch_size]
    negative_logits = logits[:, -1, negative_token_id]  # [batch_size]

    # Stack to create binary classification logits
    # Shape: [batch_size, 2] where dim=1 represents [negative, positive]
    binary_logits = torch.stack([negative_logits, positive_logits], dim=1)

    # Convert labels to the correct device and type
    binary_labels = labels.to(binary_logits.device).long()

    # Compute cross entropy loss
    loss_fct = CrossEntropyLoss()
    loss = loss_fct(binary_logits, binary_labels)

    return loss


def listwise_reranker_loss(outputs, labels, loss_scale=None, num_items_in_batch=None, **kwargs) -> torch.Tensor:
    """
    List-wise reranker loss function.

    This loss function groups samples by query based on the pattern where each group
    consists of 1 positive document followed by n negative documents. It treats the
    ranking task as a classification problem within each group, using cross-entropy
    loss to identify the positive document among all candidates.

    Data format expected:
    - labels: [1, 0, 0, 0, 1, 0, 0, ...] where 1 indicates positive, 0 indicates negative
    - Each 1 is followed by its corresponding negative documents until the next 1

    Environment variables for configuration:
    - LISTWISE_RERANKER_TEMPERATURE: Temperature for softmax (default: 1.0)
    - LISTWISE_RERANKER_MIN_GROUP_SIZE: Minimum group size to include (default: 2)

    Args:
        outputs: Model outputs containing logits [batch_size, 1]
        labels: Binary labels (1 for positive, 0 for negative) [batch_size]
        loss_scale: Not used for listwise reranker
        num_items_in_batch: Not used for listwise reranker

    Returns:
        torch.Tensor: Cross entropy loss for ranking classification
    """
    logits = outputs.logits.squeeze(-1)  # [batch_size]
    labels = labels.float()

    # Configuration from environment variables
    temperature = float(os.environ.get('LISTWISE_RERANKER_TEMPERATURE', '1.0'))
    min_group_size = int(os.environ.get('LISTWISE_RERANKER_MIN_GROUP_SIZE', '2'))

    # Find positive sample indices to determine group boundaries
    positive_indices = torch.nonzero(labels == 1, as_tuple=False).squeeze(-1)

    if len(positive_indices) == 0:
        # No positive samples in this batch, return zero loss
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    # Ensure positive_indices is 1D
    if positive_indices.dim() == 0:
        positive_indices = positive_indices.unsqueeze(0)

    total_loss = 0.0
    num_groups = 0

    for i, pos_idx in enumerate(positive_indices):
        # Determine group boundaries
        group_start = pos_idx.item()

        # Find the end of current group (start of next group or end of batch)
        if i + 1 < len(positive_indices):
            group_end = positive_indices[i + 1].item()
        else:
            group_end = len(labels)

        # Extract group logits and labels
        group_logits = logits[group_start:group_end]  # [group_size]
        group_labels = labels[group_start:group_end]  # [group_size]

        # Skip groups that are too small
        if len(group_logits) < min_group_size:
            continue

        # Verify that the first sample in the group is positive
        if group_labels[0] != 1:
            continue  # Skip malformed groups

        # Apply temperature scaling for better training dynamics
        scaled_logits = group_logits / temperature

        # The positive document is always at index 0 within the group
        target = torch.tensor(0, dtype=torch.long, device=logits.device)

        # Apply cross-entropy loss: positive document should have highest score
        loss_fct = CrossEntropyLoss()
        group_loss = loss_fct(scaled_logits.unsqueeze(0), target.unsqueeze(0))

        total_loss += group_loss
        num_groups += 1

    if num_groups == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    # Return average loss across all groups
    return total_loss / num_groups


def listwise_generative_reranker_loss(outputs,
                                      labels,
                                      loss_scale=None,
                                      num_items_in_batch=None,
                                      trainer=None,
                                      **kwargs) -> torch.Tensor:
    """
    List-wise generative reranker loss function.

    This loss function combines the generative reranker approach (using token probabilities)
    with list-wise ranking. It groups samples by query based on the pattern where each group
    consists of 1 positive document followed by n negative documents, then uses the
    probabilities of specific tokens (e.g., "yes"/"no") to perform ranking within each group.

    Data format expected:
    - labels: [1, 0, 0, 0, 1, 0, 0, ...] where 1 indicates positive, 0 indicates negative
    - Each 1 is followed by its corresponding negative documents until the next 1

    Environment variables for configuration:
    - GENERATIVE_RERANKER_POSITIVE_TOKEN: Token for positive relevance (default: "yes")
    - GENERATIVE_RERANKER_NEGATIVE_TOKEN: Token for negative relevance (default: "no")
    - LISTWISE_RERANKER_TEMPERATURE: Temperature for softmax (default: 1.0)
    - LISTWISE_RERANKER_MIN_GROUP_SIZE: Minimum group size to include (default: 2)

    Args:
        outputs: Model outputs containing logits [batch_size, seq_len, vocab_size]
        labels: Binary labels (1 for positive, 0 for negative) [batch_size]
        loss_scale: Not used for listwise generative reranker
        num_items_in_batch: Not used for listwise generative reranker
        trainer: Trainer instance to access tokenizer

    Returns:
        torch.Tensor: Cross entropy loss for ranking classification based on token probabilities
    """
    if trainer is None:
        raise ValueError('trainer is required for listwise_generative_reranker_loss to access tokenizer')

    logits = outputs.logits
    tokenizer = trainer.processing_class
    labels = labels.float()

    # Configuration from environment variables
    positive_token = os.environ.get('GENERATIVE_RERANKER_POSITIVE_TOKEN', 'yes')
    negative_token = os.environ.get('GENERATIVE_RERANKER_NEGATIVE_TOKEN', 'no')
    temperature = float(os.environ.get('LISTWISE_RERANKER_TEMPERATURE', '1.0'))
    min_group_size = int(os.environ.get('LISTWISE_RERANKER_MIN_GROUP_SIZE', '2'))

    # Get token IDs for positive and negative tokens
    try:
        positive_token_id = tokenizer.convert_tokens_to_ids(positive_token)
        negative_token_id = tokenizer.convert_tokens_to_ids(negative_token)
    except Exception as e:
        raise ValueError(f"Failed to convert tokens '{positive_token}'/'{negative_token}' to IDs. "
                         f'Please check if these tokens exist in the tokenizer vocabulary. Error: {e}')

    # Extract logits for positive and negative tokens from last position
    positive_logits = logits[:, -1, positive_token_id]  # [batch_size]
    negative_logits = logits[:, -1, negative_token_id]  # [batch_size]

    # Create binary classification logits for each sample
    # Shape: [batch_size, 2] where dim=1 represents [negative, positive]
    binary_logits = torch.stack([negative_logits, positive_logits], dim=1)

    # Convert to relevance scores using softmax (probability of positive class)
    binary_probs = torch.softmax(binary_logits, dim=1)
    relevance_scores = binary_probs[:, 1]  # Probability of positive class [batch_size]

    # Find positive sample indices to determine group boundaries
    positive_indices = torch.nonzero(labels == 1, as_tuple=False).squeeze(-1)

    if len(positive_indices) == 0:
        # No positive samples in this batch, return zero loss
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    # Ensure positive_indices is 1D
    if positive_indices.dim() == 0:
        positive_indices = positive_indices.unsqueeze(0)

    total_loss = 0.0
    num_groups = 0

    for i, pos_idx in enumerate(positive_indices):
        # Determine group boundaries
        group_start = pos_idx.item()

        # Find the end of current group (start of next group or end of batch)
        if i + 1 < len(positive_indices):
            group_end = positive_indices[i + 1].item()
        else:
            group_end = len(labels)

        # Extract group relevance scores and labels
        group_scores = relevance_scores[group_start:group_end]  # [group_size]
        group_labels = labels[group_start:group_end]  # [group_size]

        # Skip groups that are too small
        if len(group_scores) < min_group_size:
            continue

        # Verify that the first sample in the group is positive
        if group_labels[0] != 1:
            continue  # Skip malformed groups

        # Convert relevance scores to logits for cross-entropy loss
        # We use log to convert probabilities back to logits, then apply temperature
        group_logits = torch.log(group_scores + 1e-8) / temperature  # Add small epsilon for numerical stability

        # The positive document is always at index 0 within the group
        target = torch.tensor(0, dtype=torch.long, device=logits.device)

        # Apply cross-entropy loss: positive document should have highest relevance score
        loss_fct = CrossEntropyLoss()
        group_loss = loss_fct(group_logits.unsqueeze(0), target.unsqueeze(0))

        total_loss += group_loss
        num_groups += 1

    if num_groups == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    # Return average loss across all groups
    return total_loss / num_groups


###################
def seqcls_plus_listwise_rank_loss(outputs,
                                   labels,
                                   loss_scale=None,
                                   num_items_in_batch=None,
                                   **kwargs) -> torch.Tensor:
    """
    Mixed: seq-cls CE + list-wise (positive-first) + adjacent monotonic regularizer (+ optional pairwise hinge).

    分组：
      - 每个组以 label=1 开头，直到下一个 label=1 之前的样本属于同组（数据需已按“折扣从高到低”排好）。

    环境变量：
      MIXED_CE_WEIGHT            默认 0.8
      MIXED_RANK_WEIGHT          默认 0.15
      MIXED_MONO_WEIGHT          默认 0.05
      MIXED_PAIRWISE_WEIGHT      默认 0.0   # >0 开启组内 pairwise hinge
      MIXED_TEMPERATURE          默认 1.0
      MIXED_MIN_GROUP_SIZE       默认 2
      MIXED_EXPECT_GROUP_SIZE    默认 0     # >0 时与 STRICT_GROUPING 配合
      MIXED_STRICT_GROUPING      默认 True  # True: 仅等长组；False: 允许 >= MIN_GROUP_SIZE
      MIXED_POSITIVE_CLASS_INDEX 默认 1
      MIXED_MONO_MARGIN          默认 0.0   # 相邻单调 margin
      MIXED_PAIRWISE_MARGIN      默认 0.2   # pairwise hinge margin
      MIXED_GROUP_WEIGHTING      默认 'none'  # 'none' | 'len' | 'neg'
    """
    logits = outputs.logits
    device = logits.device
    y = labels

    # ---- 超参 ----
    ce_w   = float(os.environ.get('MIXED_CE_WEIGHT',   '0.60'))
    rk_w   = float(os.environ.get('MIXED_RANK_WEIGHT', '0.25'))
    mono_w = float(os.environ.get('MIXED_MONO_WEIGHT', '0.10'))
    pair_w = float(os.environ.get('MIXED_PAIRWISE_WEIGHT', '0.05'))

    temperature    = float(os.environ.get('MIXED_TEMPERATURE', '0.7'))
    min_group_size = int(os.environ.get('MIXED_MIN_GROUP_SIZE', '2'))
    expect_size    = int(os.environ.get('MIXED_EXPECT_GROUP_SIZE', '4'))
    strict_group   = os.environ.get('MIXED_STRICT_GROUPING', 'True').lower() != 'false'
    pos_class_idx  = int(os.environ.get('MIXED_POSITIVE_CLASS_INDEX', '1'))
    mono_margin    = float(os.environ.get('MIXED_MONO_MARGIN', '0.1'))
    pair_margin    = float(os.environ.get('MIXED_PAIRWISE_MARGIN', '0.3'))
    group_weighting = os.environ.get('MIXED_GROUP_WEIGHTING', 'none').lower()

    eps = 1e-6

    # ---- 分类损失 L_ce & 排序分数 s ----
    if logits.dim() == 2 and logits.size(-1) >= 2:
        ce_loss = CrossEntropyLoss()(logits, y.long().to(device))
        s = logits[:, pos_class_idx]   # 正类 logit 作为排序分数
    else:
        # [B] or [B,1]
        logits_bce = logits.squeeze(-1) if (logits.dim()==2 and logits.size(-1)==1) else logits
        ce_loss = nn.BCEWithLogitsLoss()(logits_bce, y.float().to(device))
        s = logits_bce

    # ---- 根据 label==1 找到每个组 ----
    yf = y.float()
    pos_idx = torch.nonzero(yf == 1, as_tuple=False).squeeze(-1)
    if pos_idx.numel() == 0:
        return ce_w * ce_loss
    if pos_idx.dim() == 0:
        pos_idx = pos_idx.unsqueeze(0)

    total_rank = torch.tensor(0.0, device=device)
    total_mono = torch.tensor(0.0, device=device)
    total_pair = torch.tensor(0.0, device=device)
    rank_wsum = mono_wsum = pair_wsum = 0.0  # 用于按组加权的归一

    for i, st in enumerate(pos_idx):
        st = st.item()
        ed = pos_idx[i+1].item() if (i+1) < len(pos_idx) else yf.numel()

        gs = s[st:ed]      # [G]
        gl = yf[st:ed]     # [G]
        G = gs.numel()

        # 组合法校验
        if G < max(2, min_group_size):
            continue
        if gl[0] != 1:
            continue
        if expect_size > 0:
            if strict_group and (G != expect_size):
                continue
            if (not strict_group) and (G < expect_size):
                continue

        # 组权重（可选）：none | len(=G) | neg(=G-1)
        if group_weighting == 'len':
            gw = float(G)
        elif group_weighting == 'neg':
            gw = float(max(1, G-1))
        else:
            gw = 1.0

        # ---- listwise 排序（正样本应排第一）----
        scaled = gs / max(temperature, eps)             # [G]
        # CrossEntropyLoss 期望 [N,C]，我们把一组当 batch=1，C=G
        group_rank = CrossEntropyLoss()(scaled.unsqueeze(0),
                                        torch.tensor([0], device=device))
        total_rank = total_rank + gw * group_rank
        rank_wsum += gw

        # ---- 相邻单调约束（s[0] ≥ s[1] ≥ ...）----
        if G >= 2 and mono_w > 0:
            diffs = gs[:-1] - gs[1:]                    # [G-1]
            viola = torch.relu(mono_margin - diffs)     # 违反单调的“缺口”
            mono_loss = viola.mean()
            total_mono = total_mono + gw * mono_loss
            mono_wsum += gw

        # ---- 组内 pairwise hinge（可选）: s_pos - s_neg ≥ pair_margin ----
        if pair_w > 0 and G >= 2:
            s_pos = gs[0]
            s_negs = gs[1:]
            hinge = torch.relu(pair_margin - (s_pos - s_negs))  # [G-1]
            pair_loss = hinge.mean()
            total_pair = total_pair + gw * pair_loss
            pair_wsum += gw

    rank_loss = total_rank / max(rank_wsum, 1.0) if rank_wsum > 0 else torch.tensor(0.0, device=device)
    mono_loss = total_mono / max(mono_wsum, 1.0) if mono_wsum > 0 else torch.tensor(0.0, device=device)
    opt_pair  = total_pair / max(pair_wsum, 1.0) if pair_wsum > 0 else torch.tensor(0.0, device=device)

    return ce_w * ce_loss + rk_w * rank_loss + mono_w * mono_loss + pair_w * opt_pair

def seqcls_plus_pairwise_rank_loss(outputs, labels, **kwargs):
    """
    CE + Pairwise(组内) + Monotone(相邻)
    分组：label==1 为起点到下一个 1 之前为一组；生成数据应已按折扣从高到低排序。
    环境变量：
      MIXED_CE_WEIGHT, MIXED_PAIR_WEIGHT, MIXED_MONO_WEIGHT
      MIXED_TEMPERATURE (pairwise温度, 默认0.5~1)
      MIXED_MIN_GROUP_SIZE (默认2)
      MIXED_EXPECT_GROUP_SIZE (0不强制；>0时仅长度==该值的组计算排序/单调)
      MIXED_POSITIVE_CLASS_INDEX (多分类正类下标，默认1)
      MIXED_MONO_MARGIN (相邻单调边距 τ, 默认0.0~0.2)
      PAIR_INCLUDE_NEGNEG (是否对负例内部也做 pairwise，默认1)
      PAIR_MARGIN (pairwise hinge 边距，默认0，用 logistic 就忽略)
      PAIR_LOSS (logistic|hinge, 默认 logistic)
    """
    logits = outputs.logits
    device = logits.device
    y = labels

    # weights & hparams
    ce_w = float(os.environ.get('MIXED_CE_WEIGHT', '0.5'))
    pr_w = float(os.environ.get('MIXED_PAIR_WEIGHT', '0.4'))
    mono_w = float(os.environ.get('MIXED_MONO_WEIGHT', '0.1'))
    T = float(os.environ.get('MIXED_TEMPERATURE', '0.6'))
    minG = int(os.environ.get('MIXED_MIN_GROUP_SIZE', '2'))
    expG = int(os.environ.get('MIXED_EXPECT_GROUP_SIZE', '0'))
    pos_ix = int(os.environ.get('MIXED_POSITIVE_CLASS_INDEX', '1'))
    mono_m = float(os.environ.get('MIXED_MONO_MARGIN', '0.1'))
    negneg = os.environ.get('PAIR_INCLUDE_NEGNEG', '1') == '1'
    pair_margin = float(os.environ.get('PAIR_MARGIN', '0.3'))
    pair_mode = os.environ.get('PAIR_LOSS', 'logistic').lower()  # 'logistic' or 'hinge'

    # ce_w   = float(os.environ.get('MIXED_CE_WEIGHT',   '0.3'))
    # pr_w   = float(os.environ.get('MIXED_PAIR_WEIGHT', '0.6'))
    # mono_w = float(os.environ.get('MIXED_MONO_WEIGHT', '0.1'))
    # T      = float(os.environ.get('MIXED_TEMPERATURE', '0.55'))
    # minG   = int(os.environ.get('MIXED_MIN_GROUP_SIZE','2'))
    # expG   = int(os.environ.get('MIXED_EXPECT_GROUP_SIZE','0'))
    # pos_ix = int(os.environ.get('MIXED_POSITIVE_CLASS_INDEX','1'))
    # mono_m = float(os.environ.get('MIXED_MONO_MARGIN','0.12'))
    # negneg = os.environ.get('PAIR_INCLUDE_NEGNEG','1') == '1'
    # pair_margin = float(os.environ.get('PAIR_MARGIN','0.3'))
    # pair_mode = os.environ.get('PAIR_LOSS','logistic').lower()  # 'logistic' or 'hinge'

    # CE + 取排序分数 s
    if logits.dim()==2 and logits.size(-1)>=2:
        ce = CrossEntropyLoss()(logits, y.long().to(device))
        s  = logits[:, pos_ix]
    else:
        z = logits.squeeze(-1) if (logits.dim()==2 and logits.size(-1)==1) else logits
        ce = nn.BCEWithLogitsLoss()(z, y.float().to(device))
        s  = z

    # 分组
    yf = y.float()
    pos_idx = torch.nonzero(yf==1, as_tuple=False).squeeze(-1)
    if pos_idx.numel()==0:
        return ce_w*ce
    if pos_idx.dim()==0:
        pos_idx = pos_idx.unsqueeze(0)

    # helpers
    def pairwise_logistic(x):
        # L = log(1 + exp(-(x)/T))
        return torch.nn.functional.softplus(-(x)/max(T,1e-6))
    def pairwise_hinge(x):
        # L = relu(margin - x)
        return torch.relu(pair_margin - x)

    total_pair = torch.tensor(0.0, device=device)
    total_mono = torch.tensor(0.0, device=device)
    n_pairG = 0
    n_monoG = 0

    for i, st in enumerate(pos_idx.tolist()):
        ed = pos_idx[i+1].item() if (i+1)<len(pos_idx) else yf.numel()
        gs = s[st:ed]   # [G]
        gl = yf[st:ed]  # [G]
        G  = gs.numel()
        if G < minG:
            continue
        if gl[0] != 1:
            continue
        if expG>0 and G != expG:
            continue

        # (1) pairwise：正对负
        pos_score = gs[0]
        neg_scores = gs[1:]
        if neg_scores.numel()>0:
            diff = pos_score - neg_scores  # 应 >= 0
            loss_vec = pairwise_logistic(diff) if pair_mode=='logistic' else pairwise_hinge(diff)
            total_pair = total_pair + loss_vec.mean()
            n_pairG += 1

        # (2) 负例内部也按顺序约束（高折扣 ≥ 低折扣）
        if negneg and neg_scores.numel()>=2:
            # 全对儿或只相邻都行，这里做“全对儿更强”；相邻可改为 diff = neg_scores[:-1]-neg_scores[1:]
            diffs = []
            for a in range(neg_scores.numel()-1):
                for b in range(a+1, neg_scores.numel()):
                    diffs.append(neg_scores[a] - neg_scores[b])  # 应 ≥ 0
            diffs = torch.stack(diffs)
            loss_vec2 = pairwise_logistic(diffs) if pair_mode=='logistic' else pairwise_hinge(diffs)
            total_pair = total_pair + loss_vec2.mean()
            # 不另计 n_pairG，维持尺度稳定

        # (3) 相邻单调（与上一步相比更温和，可一起用）
        if G >= 2:
            dif = gs[:-1] - gs[1:]
            mono = torch.relu(mono_m - dif).mean()
            total_mono = total_mono + mono
            n_monoG += 1

    pair_loss = (total_pair / max(n_pairG,1)) if n_pairG>0 else torch.tensor(0.0, device=device)
    mono_loss = (total_mono / max(n_monoG,1)) if n_monoG>0 else torch.tensor(0.0, device=device)

    return ce_w*ce + pr_w*pair_loss + mono_w*mono_loss


def seqcls_plus_pairwise_rank_loss_threshold(
        outputs,
        labels,
        loss_scale: Optional[float] = None,
        num_items_in_batch: Optional[int] = None,
        **kwargs
) -> torch.Tensor:
    """
    基于阈值分组的软标签Pairwise Ranking损失（Swift框架标准版本）

    核心逻辑：
    1. BCE使用所有样本的软标签（0-1之间的浮点数）
    2. Pairwise分组：通过阈值判断"强正例"和"强负例"
    3. 单调性约束：相邻discount的分数应递减
    4. 自动权重推断：基于标签置信度（接近0或1的权重高）

    参数：
        outputs: 模型输出对象（包含logits属性）
        labels: 软标签，tensor of shape [batch_size], 值在[0, 1]之间
        loss_scale: 损失缩放因子（swift框架保留参数）
        num_items_in_batch: batch中的有效样本数（swift框架保留参数）
        **kwargs: 其他参数（swift框架可能传入的额外信息）

    环境变量配置：
        MIXED_CE_WEIGHT: BCE权重，默认0.5
        MIXED_PAIR_WEIGHT: Pairwise权重，默认0.3
        MIXED_MONO_WEIGHT: 单调性权重，默认0.2
        MIXED_TEMPERATURE: Pairwise温度参数，默认0.6
        MIXED_MONO_MARGIN: 单调性边距，默认0.1
        MIXED_POS_THRESHOLD: 强正例阈值，默认0.7
        MIXED_NEG_THRESHOLD: 强负例阈值，默认0.3
        MIXED_MIN_GROUP_SIZE: 最小组大小，默认2
        PAIR_INCLUDE_NEGNEG: 负例之间是否也做pairwise，默认1
        PAIR_MIN_DIFF: Pairwise最小标签差异阈值，默认0.15
        MIXED_AUTO_WEIGHT: 是否自动计算样本权重，默认1
            - 0: 所有样本权重=1
            - 1: 根据标签置信度计算（label接近0或1 → 权重高）
        MIXED_WEIGHT_MIN: 自动权重的最小值，默认0.3
        MIXED_WEIGHT_MAX: 自动权重的最大值，默认1.0

    返回：
        torch.Tensor: 标量损失值
    """
    # ========== 提取logits和labels ==========
    logits = outputs.logits
    device = logits.device
    y = labels.float().to(device)  # 软标签

    # ========== 超参数读取 ==========
    ce_w = float(os.environ.get('MIXED_CE_WEIGHT', '0.5'))
    pr_w = float(os.environ.get('MIXED_PAIR_WEIGHT', '0.3'))
    mono_w = float(os.environ.get('MIXED_MONO_WEIGHT', '0.2'))
    T = float(os.environ.get('MIXED_TEMPERATURE', '0.6'))
    mono_margin = float(os.environ.get('MIXED_MONO_MARGIN', '0.1'))

    # 阈值参数
    pos_threshold = float(os.environ.get('MIXED_POS_THRESHOLD', '0.7'))
    neg_threshold = float(os.environ.get('MIXED_NEG_THRESHOLD', '0.3'))
    min_group_size = int(os.environ.get('MIXED_MIN_GROUP_SIZE', '2'))
    negneg = os.environ.get('PAIR_INCLUDE_NEGNEG', '1') == '1'
    min_diff = float(os.environ.get('PAIR_MIN_DIFF', '0.15'))

    # 权重参数
    auto_weight = os.environ.get('MIXED_AUTO_WEIGHT', '1') == '1'
    weight_min = float(os.environ.get('MIXED_WEIGHT_MIN', '0.3'))
    weight_max = float(os.environ.get('MIXED_WEIGHT_MAX', '1.0'))

    # ========== 自动权重计算 ==========
    if auto_weight:
        # 基于标签置信度计算权重
        # label接近0或1（确定性高） → 权重高
        # label接近0.5（不确定性高） → 权重低

        # 计算距离0.5的距离（归一化到[0,1]）
        confidence = torch.abs(y - 0.5) * 2  # [0, 1]范围

        # 映射到[weight_min, weight_max]
        weights = weight_min + (weight_max - weight_min) * confidence

        # 示例：
        # label=1.0 → confidence=1.0 → weight=1.0
        # label=0.0 → confidence=1.0 → weight=1.0
        # label=0.85 → confidence=0.7 → weight=0.3+0.7*0.7=0.79
        # label=0.50 → confidence=0.0 → weight=0.3
    else:
        # 所有样本权重相同
        weights = torch.ones_like(y)

    # ========== 1. 加权BCE损失（支持软标签）==========
    if logits.dim() == 2 and logits.size(-1) >= 2:
        # 多分类情况
        pos_class_idx = int(os.environ.get('MIXED_POSITIVE_CLASS_INDEX', '1'))
        s = logits[:, pos_class_idx]  # 排序分数

        # 计算概率
        probs = torch.softmax(logits, dim=-1)[:, pos_class_idx]

        # BCE loss
        ce = F.binary_cross_entropy(probs, y, reduction='none')

    else:
        # 单输出情况
        z = logits.squeeze(-1) if (logits.dim() == 2 and logits.size(-1) == 1) else logits
        s = z  # logits作为排序分数

        # BCE with logits
        ce = F.binary_cross_entropy_with_logits(z, y, reduction='none')

    # 加权平均
    ce_weighted = (ce * weights).sum()

    # 归一化
    if num_items_in_batch is not None and num_items_in_batch > 0:
        ce_loss = ce_weighted / num_items_in_batch
    else:
        ce_loss = ce_weighted / max(weights.sum(), 1e-6)

    # ========== 2. 基于阈值的Pairwise损失 ==========
    strong_pos_mask = y >= pos_threshold
    strong_pos_idx = torch.nonzero(strong_pos_mask, as_tuple=False).squeeze(-1)

    if strong_pos_idx.numel() == 0:
        pair_loss = torch.tensor(0.0, device=device)
    else:
        if strong_pos_idx.dim() == 0:
            strong_pos_idx = strong_pos_idx.unsqueeze(0)

        total_pair = torch.tensor(0.0, device=device)
        pair_count = 0

        for i, st in enumerate(strong_pos_idx.tolist()):
            ed = strong_pos_idx[i + 1].item() if (i + 1) < len(strong_pos_idx) else y.numel()

            gs = s[st:ed]
            gy = y[st:ed]
            gw = weights[st:ed]

            G = gs.numel()

            if G < min_group_size:
                continue

            if gy[0] < pos_threshold:
                continue

            # === Pairwise: anchor vs 其他样本 ===
            pos_score = gs[0]
            pos_label = gy[0]

            for j in range(1, G):
                other_score = gs[j]
                other_label = gy[j]
                other_weight = gw[j]

                y_diff = pos_label - other_label

                if y_diff > min_diff:
                    score_diff = pos_score - other_score
                    loss_pair = F.softplus(-score_diff / max(T, 1e-6))

                    # 权重 = 标签差异 × 样本权重
                    weight_pair = y_diff * other_weight

                    total_pair += loss_pair * weight_pair
                    pair_count += 1

            # === 负例之间的pairwise ===
            if negneg and G >= 3:
                neg_indices = [j for j in range(1, G) if gy[j] <= neg_threshold]

                for idx1 in range(len(neg_indices) - 1):
                    for idx2 in range(idx1 + 1, len(neg_indices)):
                        j1 = neg_indices[idx1]
                        j2 = neg_indices[idx2]

                        y_diff_neg = gy[j1] - gy[j2]

                        if abs(y_diff_neg) > 0.1:
                            score_diff_neg = gs[j1] - gs[j2]

                            if y_diff_neg > 0:
                                loss_neg = F.softplus(-score_diff_neg / max(T, 1e-6))
                            else:
                                loss_neg = F.softplus(score_diff_neg / max(T, 1e-6))

                            weight_neg = abs(y_diff_neg) * (gw[j1] + gw[j2]) / 2 * 0.5

                            total_pair += loss_neg * weight_neg
                            pair_count += 1

        pair_loss = total_pair / max(pair_count, 1) if pair_count > 0 else torch.tensor(0.0, device=device)

    # ========== 3. 单调性约束 ==========
    total_mono = torch.tensor(0.0, device=device)
    mono_count = 0

    if strong_pos_idx.numel() > 0:
        for i, st in enumerate(strong_pos_idx.tolist()):
            ed = strong_pos_idx[i + 1].item() if (i + 1) < len(strong_pos_idx) else y.numel()

            gs = s[st:ed]
            gw = weights[st:ed]
            G = gs.numel()

            if G >= 2:
                diffs = gs[:-1] - gs[1:]
                mono_violations = torch.relu(mono_margin - diffs)

                # 使用相邻样本的平均权重
                pair_weights = (gw[:-1] + gw[1:]) / 2

                total_mono += (mono_violations * pair_weights).sum()
                mono_count += len(diffs)

    mono_loss = total_mono / max(mono_count, 1) if mono_count > 0 else torch.tensor(0.0, device=device)

    # ========== 4. 组合损失 ==========
    total_loss = ce_w * ce_loss + pr_w * pair_loss + mono_w * mono_loss

    # 如果提供了loss_scale，应用缩放
    if loss_scale is not None and loss_scale != 1.0:
        total_loss = total_loss * loss_scale

    return total_loss


###################
def pricing_pairwise_loss(outputs, labels, loss_scale=None, num_items_in_batch=None, **kwargs) -> torch.Tensor:
    """
    营销定价损失函数：50% CE + 50% Pairwise

    数据格式要求：
    - 相邻两条样本形成正负对，具有相同的用户画像但不同的discount
    - batch_size必须是偶数
    - 样本组织（由数据合成脚本生成）：
      * 偶数索引（0,2,4...）：discount较大的样本（相对"正"）
      * 奇数索引（1,3,5...）：discount较小的样本（相对"负"）

    损失计算：
    1. CE Loss (50%): 标准的二分类交叉熵，使用真实label
    2. Pairwise Loss (50%): 确保discount大的样本得分高于discount小的样本

    环境变量配置：
    - PRICING_PAIRWISE_MARGIN: Pairwise loss的margin (默认0.5)
    - PRICING_CE_WEIGHT: CE loss的权重 (默认0.5)
    - PRICING_PAIRWISE_WEIGHT: Pairwise loss的权重 (默认0.5)
    """
    logits = outputs.logits  # 可能是 [batch_size] 或 [batch_size, num_classes]

    # 处理不同的logits形状
    if len(logits.shape) == 1:
        # 形状是 [batch_size]，二分类单输出
        logits = logits
        use_bce = True
    elif len(logits.shape) == 2:
        if logits.shape[1] == 1:
            # 形状是 [batch_size, 1]，需要squeeze
            logits = logits.squeeze(-1)
            use_bce = True
        elif logits.shape[1] == 2:
            # 形状是 [batch_size, 2]，二分类双输出
            # 对于Pairwise，我们使用正类（class 1）的logit
            use_bce = False
        else:
            raise ValueError(f"Unexpected logits shape: {logits.shape}")
    else:
        raise ValueError(f"Unexpected logits shape: {logits.shape}")

    labels = labels.long() if not use_bce else labels.float()
    batch_size = logits.size(0)

    # 确保batch_size是偶数
    if batch_size % 2 != 0:
        raise ValueError(
            f"Batch size must be even for pairwise loss, got {batch_size}. "
            f"Please set batch_size to an even number (e.g., 2, 4, 8, 16)."
        )

    # 从环境变量获取配置
    margin = float(os.environ.get('PRICING_PAIRWISE_MARGIN', '0.5'))
    ce_weight = float(os.environ.get('PRICING_CE_WEIGHT', '0.5'))
    pairwise_weight = float(os.environ.get('PRICING_PAIRWISE_WEIGHT', '0.5'))

    # 1. 计算交叉熵损失
    if use_bce:
        # 使用BCEWithLogitsLoss (logits是 [batch_size])
        ce_loss_fct = BCEWithLogitsLoss()
        ce_loss = ce_loss_fct(logits, labels)
        # 用于pairwise的得分就是logits本身
        scores = logits
    else:
        # 使用CrossEntropyLoss (logits是 [batch_size, 2])
        ce_loss_fct = CrossEntropyLoss()
        ce_loss = ce_loss_fct(logits, labels)
        # 用于pairwise的得分是正类的logit（索引为1）
        scores = logits[:, 1]

    # 2. 计算Pairwise损失
    # 数据组织：[discount大, discount小, discount大, discount小, ...]
    # 索引：     [0,         1,         2,         3,         ...]

    pos_indices = torch.arange(0, batch_size, 2, device=scores.device)  # [0, 2, 4, ...]
    neg_indices = torch.arange(1, batch_size, 2, device=scores.device)  # [1, 3, 5, ...]

    pos_scores = scores[pos_indices]  # discount较大的样本的得分
    neg_scores = scores[neg_indices]  # discount较小的样本的得分

    # 使用Margin Ranking Loss
    # target=1 表示期望第一个输入 > 第二个输入（即pos_scores > neg_scores）
    target = torch.ones_like(pos_scores)
    pairwise_loss_fct = MarginRankingLoss(margin=margin)
    pairwise_loss = pairwise_loss_fct(pos_scores, neg_scores, target)

    # 3. 组合损失
    total_loss = ce_weight * ce_loss + pairwise_weight * pairwise_loss

    # 4. 记录额外信息（用于监控，不影响反向传播）
    with torch.no_grad():
        # 计算pairwise准确率
        pairwise_correct = (pos_scores > neg_scores).float().mean()
        avg_margin = (pos_scores - neg_scores).mean()

        # 可以通过outputs存储额外信息供后续使用
        if hasattr(outputs, 'loss_info'):
            outputs.loss_info = {
                'ce_loss': ce_loss.item(),
                'pairwise_loss': pairwise_loss.item(),
                'pairwise_accuracy': pairwise_correct.item(),
                'avg_margin': avg_margin.item(),
            }

    return total_loss


def pricing_listwise_loss_old(outputs, labels, loss_scale=None, num_items_in_batch=None, **kwargs) -> torch.Tensor:
    """
    营销定价Listwise损失函数（改进版）：CE + Listwise Ranking + Monotonic Constraint

    三个目标：
    1. CE Loss: 二分类准确性
    2. Listwise Loss: 整体排序分布优化
    3. Monotonic Loss: 显式单调性约束

    环境变量：
    - PRICING_CHAIN_LENGTH: 序列长度（默认8）
    - PRICING_CE_WEIGHT: CE权重（默认0.2）
    - PRICING_LISTWISE_WEIGHT: Listwise权重（默认0.5）
    - PRICING_MONOTONIC_WEIGHT: 单调性权重（默认0.3）
    - PRICING_LISTWISE_TEMP: Softmax温度（默认1.0）
    - PRICING_MONOTONIC_MARGIN: 单调性margin（默认0.1）
    """
    logits = outputs.logits

    # 处理logits形状
    if len(logits.shape) == 2:
        if logits.shape[1] == 1:
            scores = logits.squeeze(-1)
            use_bce = True
        elif logits.shape[1] == 2:
            scores = logits[:, 1]  # 正类的logit
            use_bce = False
        else:
            raise ValueError(f"Unexpected logits shape: {logits.shape}")
    else:
        scores = logits
        use_bce = True

    labels_for_ce = labels.float() if use_bce else labels.long()

    # 配置参数
    chain_length = int(os.environ.get('PRICING_CHAIN_LENGTH', '8'))
    ce_weight = float(os.environ.get('PRICING_CE_WEIGHT', '0.2'))
    listwise_weight = float(os.environ.get('PRICING_LISTWISE_WEIGHT', '0.5'))
    monotonic_weight = float(os.environ.get('PRICING_MONOTONIC_WEIGHT', '0.3'))
    temperature = float(os.environ.get('PRICING_LISTWISE_TEMP', '1.0'))
    margin = float(os.environ.get('PRICING_MONOTONIC_MARGIN', '0.1'))

    batch_size = scores.size(0)

    if batch_size % chain_length != 0:
        raise ValueError(
            f"Batch size ({batch_size}) must be divisible by chain_length ({chain_length}). "
            f"Current batch_size={batch_size}, chain_length={chain_length}"
        )

    # 1. CE损失（标准分类loss）
    if use_bce:
        ce_loss = nn.BCEWithLogitsLoss()(
            logits.squeeze(-1) if len(logits.shape) == 2 else logits,
            labels_for_ce
        )
    else:
        ce_loss = nn.CrossEntropyLoss()(logits, labels_for_ce)

    # 重塑scores: [num_sequences, chain_length]
    num_sequences = batch_size // chain_length
    scores_reshaped = scores.view(num_sequences, chain_length)
    labels_reshaped = labels.view(num_sequences, chain_length)

    # 2. Listwise Ranking损失（改进版）
    # 构建更aggressive的理想分布
    # 使用指数增长而不是线性增长
    ideal_scores = torch.arange(chain_length, dtype=scores.dtype, device=scores.device)
    # 指数缩放：让后面位置的优势更明显
    ideal_scores = torch.exp(ideal_scores * 0.5)  # 指数增长
    ideal_scores = ideal_scores.unsqueeze(0).expand(num_sequences, -1)

    # 计算softmax分布
    pred_probs = torch.softmax(scores_reshaped / temperature, dim=1)
    ideal_probs = torch.softmax(ideal_scores / temperature, dim=1)

    # KL散度：pred尽量接近ideal
    listwise_loss = nn.KLDivLoss(reduction='batchmean')(
        torch.log(pred_probs + 1e-10),
        ideal_probs
    )

    # 3. 显式单调性约束
    # 确保 score[i+1] > score[i] + margin
    # 计算相邻位置的得分差
    score_diffs = scores_reshaped[:, 1:] - scores_reshaped[:, :-1]  # [num_seq, chain_length-1]

    # Hinge loss: max(0, margin - diff)
    # 如果 diff >= margin，loss=0；否则loss=margin-diff
    monotonic_loss = torch.clamp(margin - score_diffs, min=0).mean()

    # 4. 组合损失
    total_loss = (
            ce_weight * ce_loss +
            listwise_weight * listwise_loss +
            monotonic_weight * monotonic_loss
    )

    # 5. 记录监控指标（不影响梯度）
    # with torch.no_grad():
    #     # 单调性违反率：有多少相邻对不满足 score[i+1] > score[i]
    #     monotonic_violations = (score_diffs <= 0).float().mean()
    #
    #     # 平均得分增量
    #     avg_score_increment = score_diffs.mean()
    #
    #     # 首尾得分差
    #     score_range = (scores_reshaped[:, -1] - scores_reshaped[:, 0]).mean()
    #
    #     # Kendall's tau相关系数（理想排序 vs 实际排序）
    #     # 简化版：计算逆序对比例
    #     n_pairs = chain_length * (chain_length - 1) / 2
    #     inversions = 0
    #     for i in range(chain_length):
    #         for j in range(i + 1, chain_length):
    #             inversions += (scores_reshaped[:, i] > scores_reshaped[:, j]).sum().item()
    #     inversion_rate = inversions / (num_sequences * n_pairs)
    #
    #     # 存储到outputs（可选）
    #     if hasattr(outputs, 'loss_info'):
    #         outputs.loss_info = {
    #             'ce_loss': ce_loss.item(),
    #             'listwise_loss': listwise_loss.item(),
    #             'monotonic_loss': monotonic_loss.item(),
    #             'monotonic_violation_rate': monotonic_violations.item(),
    #             'avg_score_increment': avg_score_increment.item(),
    #             'score_range': score_range.item(),
    #             'inversion_rate': inversion_rate,
    #         }

    return total_loss


def pricing_listwise_loss(outputs, labels, loss_scale=None, num_items_in_batch=None, **kwargs) -> torch.Tensor:
    """
    营销定价Listwise损失函数（简洁版）：CE + Listwise Ranking

    环境变量：
    - PRICING_CHAIN_LENGTH: 序列长度（默认8）
    - PRICING_CE_WEIGHT: CE权重（默认0.3）
    - PRICING_LISTWISE_WEIGHT: Listwise权重（默认0.7）
    - PRICING_LISTWISE_TEMP: Softmax温度（默认1.0）
    """
    logits = outputs.logits

    # 处理logits形状
    if len(logits.shape) == 2:
        if logits.shape[1] == 1:
            scores = logits.squeeze(-1)
            use_bce = True
        elif logits.shape[1] == 2:
            scores = logits[:, 1]  # 正类的logit
            use_bce = False
        else:
            raise ValueError(f"Unexpected logits shape: {logits.shape}")
    else:
        scores = logits
        use_bce = True

    labels_for_ce = labels.float() if use_bce else labels.long()

    # 配置参数
    # chain_length = int(os.environ.get('PRICING_CHAIN_LENGTH', '8'))
    # ce_weight = float(os.environ.get('PRICING_CE_WEIGHT', '0.1'))
    # listwise_weight = float(os.environ.get('PRICING_LISTWISE_WEIGHT', '0.9'))
    # temperature = float(os.environ.get('PRICING_LISTWISE_TEMP', '0.5'))
    chain_length = int(os.environ.get('PRICING_CHAIN_LENGTH', '8'))
    ce_weight = float(os.environ.get('PRICING_CE_WEIGHT', '0.022'))
    listwise_weight = float(os.environ.get('PRICING_LISTWISE_WEIGHT', '0.978'))
    temperature = float(os.environ.get('PRICING_LISTWISE_TEMP', '0.5'))

    batch_size = scores.size(0)

    if batch_size % chain_length != 0:
        raise ValueError(
            f"Batch size ({batch_size}) must be divisible by chain_length ({chain_length})"
        )

    # 1. CE损失
    if use_bce:
        ce_loss = nn.BCEWithLogitsLoss()(
            logits.squeeze(-1) if len(logits.shape) == 2 else logits,
            labels_for_ce
        )
    else:
        ce_loss = nn.CrossEntropyLoss()(logits, labels_for_ce)

    # 2. Listwise Ranking损失
    num_sequences = batch_size // chain_length
    scores_reshaped = scores.view(num_sequences, chain_length)

    # 理想分布：位置越靠后，概率越高（指数增长）
    ideal_scores = torch.arange(chain_length, dtype=scores.dtype, device=scores.device)
    ideal_scores = torch.exp(ideal_scores * 0.5)
    ideal_scores = ideal_scores.unsqueeze(0).expand(num_sequences, -1)

    # 计算softmax分布
    pred_probs = torch.softmax(scores_reshaped / temperature, dim=1)
    ideal_probs = torch.softmax(ideal_scores / temperature, dim=1)

    # KL散度
    listwise_loss = nn.KLDivLoss(reduction='batchmean')(
        torch.log(pred_probs + 1e-10),
        ideal_probs
    )

    # 组合损失
    total_loss = ce_weight * ce_loss + listwise_weight * listwise_loss

    return total_loss


import swanlab


def should_log_to_swanlab():
    """
    Check if current process should log to SwanLab.
    Only main process (rank 0) should log in distributed training.
    """
    # Check if SwanLab run exists (only initialized on main process)
    if swanlab.get_run() is None:
        return False

    # Additional safety: verify we're on main process in distributed training
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0

    return True


def to_scalar(value):
    """
    将输入值转换为标量
    - 如果是 PyTorch Tensor 类型，调用 .item() 转为标量
    - 如果是 float/int 等数值类型，直接返回原值
    - 其他类型会尝试转换为 float，转换失败则抛出异常
    """
    if isinstance(value, torch.Tensor):
        return value.item()
    else:
        return float(value)

def pricing_listmle_loss(
        outputs,
        labels,
        loss_scale=None,
        num_items_in_batch=None,
        **kwargs
) -> torch.Tensor:
    """
    Pricing ListMLE Loss（反转排序版本）

    数据格式：
    - batch_size是8的倍数（如8, 16, 24...）
    - 每8个样本为一组，对应同一用户
    - 在每组内，位置0→7，接受概率递增
      位置0: 低折扣(d0)  → 接受概率低 → score应该低
      位置7: 高折扣(d140) → 接受概率高 → score应该高
    - label用于CE loss，ListMLE部分不使用label

    优化目标：
    - ListMLE：最大化从高到低排序 [7,6,5,4,3,2,1,0] 的概率
      即：s_7 > s_6 > ... > s_1 > s_0
      等价于：s_0 < s_1 < ... < s_7 ✅
    - CE：基础分类能力（使用label）

    环境变量：
    - PRICING_CHAIN_LENGTH: 每组样本数（默认8）
    - PRICING_CE_WEIGHT: CE权重（默认0.1）
    - PRICING_LISTMLE_WEIGHT: ListMLE权重（默认0.9）
    - PRICING_LISTMLE_TEMP: 温度参数（默认1.0）

    ListMLE公式：
        理想排序 π = [7, 6, 5, 4, 3, 2, 1, 0]（反向位置顺序）
        P(π) = ∏_{i=0}^{n-1} exp(s_{π_i}) / Σ_{j=i}^{n-1} exp(s_{π_j})
        Loss = -log P(π)
    """
    logits = outputs.logits

    # ========== 1. 处理logits形状 ==========
    if len(logits.shape) == 2:
        if logits.shape[1] == 1:
            scores = logits.squeeze(-1)
            use_bce = True
            use_3class = False
        elif logits.shape[1] == 2:
            scores = logits[:, 1]  # 正类的logit
            use_bce = False
            use_3class = False
        elif logits.shape[1] == 3:
            # 三分类支持 - 使用对应类别的logit作为ranking score
            # 原因：训练序列内label都相同，应该比较同类内的置信度差异
            # 而非跨类别的加权组合（会导致label=0序列score全为0）
            # 提取每个样本对应label的logit: scores[i] = logits[i, labels[i]]
            scores = torch.gather(logits, dim=1, index=labels.unsqueeze(1)).squeeze(1)
            use_bce = False
            use_3class = True
        else:
            raise ValueError(f"Unexpected logits shape: {logits.shape}")
    else:
        scores = logits
        use_bce = True
        use_3class = False

    labels_for_ce = labels.float() if use_bce else labels.long()

    # ========== 2. 读取配置 ==========
    # chain_length = int(os.environ.get('PRICING_CHAIN_LENGTH', '8'))
    # ce_weight = float(os.environ.get('PRICING_CE_WEIGHT', '0.1'))
    # listmle_weight = float(os.environ.get('PRICING_LISTMLE_WEIGHT', '0.9'))
    # temperature = float(os.environ.get('PRICING_LISTMLE_TEMP', '1.0'))
    chain_length = int(os.environ.get('PRICING_CHAIN_LENGTH', '8'))
    ce_weight = float(os.environ.get('PRICING_CE_WEIGHT', '0.85'))
    listmle_weight = float(os.environ.get('PRICING_LISTMLE_WEIGHT', '0.15'))
    temperature = float(os.environ.get('PRICING_LISTMLE_TEMP', '1.0'))

    batch_size = scores.size(0)

    if batch_size % chain_length != 0:
        raise ValueError(
            f"Batch size ({batch_size}) must be divisible by chain_length ({chain_length})"
        )

    # ========== 3. CE损失（使用label） ==========
    if use_3class:
        ce_loss = nn.CrossEntropyLoss()(logits, labels_for_ce)
    elif use_bce:
        ce_loss = nn.BCEWithLogitsLoss()(
            logits.squeeze(-1) if len(logits.shape) == 2 else logits,
            labels_for_ce
        )
    else:
        ce_loss = nn.CrossEntropyLoss()(logits, labels_for_ce)

    # ========== 3.5. 计算预测标签和统计信息 ==========
    with torch.no_grad():  # 不需要梯度
        if use_3class:
            # 三分类：获取概率最大的类别
            predicted_labels = torch.argmax(logits, dim=-1)  # [batch_size]
        elif use_bce:
            # 二分类 BCE：概率 > 0.5 为正类
            predicted_labels = (torch.sigmoid(logits.squeeze(-1) if len(logits.shape) == 2 else logits) > 0.5).long()
        else:
            # 二分类 CE：获取概率最大的类别
            predicted_labels = torch.argmax(logits, dim=-1)

        # 真实标签
        true_labels = labels.long()

        # 计算批次准确率
        correct = (predicted_labels == true_labels).float()
        batch_accuracy = correct.mean().item()

        # 计算每个类别的准确率和样本数
        label_stats = {}
        num_classes = 3 if use_3class else 2

        for label_idx in range(num_classes):
            # 找到该类别的所有样本
            label_mask = (true_labels == label_idx)
            label_count = label_mask.sum().item()

            if label_count > 0:
                # 计算该类别的准确率
                label_correct = correct[label_mask].sum().item()
                label_accuracy = label_correct / label_count
            else:
                label_accuracy = 0.0

            label_stats[f'label_{label_idx}_accuracy'] = label_accuracy
            label_stats[f'label_{label_idx}_count'] = label_count

    # ========== 4. ListMLE损失（不使用label，只用位置） ==========
    num_sequences = batch_size // chain_length
    total_listmle = 0.0

    for seq_idx in range(num_sequences):
        start = seq_idx * chain_length
        end = start + chain_length

        seq_scores = scores[start:end]  # [chain_length]

        # 温度缩放
        if temperature != 1.0:
            seq_scores = seq_scores / temperature

        # 🔄 关键：反转scores，使得理想排序变为 [7,6,5,4,3,2,1,0]
        # 原始：seq_scores[0] 对应位置0 (低折扣)
        #      seq_scores[7] 对应位置7 (高折扣)
        # 反转后：reversed_scores[0] 对应位置7 (高折扣) ← 应该最先被选
        #        reversed_scores[7] 对应位置0 (低折扣) ← 应该最后被选
        reversed_scores = torch.flip(seq_scores, dims=[0])

        # 计算ListMLE损失
        # 理想排序：[位置7, 位置6, ..., 位置1, 位置0]
        # Loss = -Σ_{i=0}^{n-1} [reversed_scores[i] - log_sum_exp(reversed_scores[i:])]
        listmle = 0.0
        for i in range(chain_length):
            numerator = reversed_scores[i]
            remaining_scores = reversed_scores[i:]
            denominator = torch.logsumexp(remaining_scores, dim=0)
            listmle -= (numerator - denominator)

        total_listmle += listmle

    # 平均ListMLE损失
    avg_listmle = total_listmle / num_sequences

    # ========== 5. 组合损失与日志记录 ==========
    total_loss = ce_weight * ce_loss + listmle_weight * avg_listmle

    # 构建日志字典
    log_dict = {
        "ce_loss": to_scalar(ce_loss),
        "listmle_loss": to_scalar(avg_listmle),
        "total_loss": to_scalar(total_loss),
        "batch_accuracy": batch_accuracy,
    }

    # 添加每个类别的统计信息
    log_dict.update(label_stats)

    # 记录到 swanlab
    if should_log_to_swanlab():
        swanlab.log(log_dict)

    return total_loss


def bank_rank_listwise_loss(outputs, labels, loss_scale=None, num_items_in_batch=None,
                            **kwargs) -> torch.Tensor:
    """
    多城市银行推荐Listwise损失（修复版）

    核心思路：
    1. CE Loss: 基础分类损失
    2. Ranking Loss: 使用KL散度优化排序分布
       - 理想分布：第1>第2>其他，且呈指数递减

    数据格式：
    - 每个chain由chain_length个样本组成
    - 第1个: Ground Truth (label=1)
    - 第2-N个: 按流行度排序的负样本 (label=0)

    环境变量：
    - BANK_CHAIN_LENGTH: 序列长度（默认8）
    - BANK_CE_WEIGHT: CE权重（默认0.3）
    - BANK_RANKING_WEIGHT: Ranking权重（默认0.7）
    - BANK_RANKING_TEMP: Softmax温度（默认0.5）
    - BANK_TOP2_FACTOR: Top-2权重因子（默认0.7，范围0-1）
    - BANK_RANKING_MODE: 排序模式（默认'gt_only'）

    Args:
        outputs: 模型输出
        labels: 二值标签

    Returns:
        torch.Tensor: 组合损失
    """

    logits = outputs.logits  # [batch_size, num_classes]

    # ========== 配置参数 ==========
    chain_length = int(os.environ.get('BANK_CHAIN_LENGTH', '8'))
    ce_weight = float(os.environ.get('BANK_CE_WEIGHT', '0.3'))
    ranking_weight = float(os.environ.get('BANK_RANKING_WEIGHT', '0.7'))
    temperature = float(os.environ.get('BANK_RANKING_TEMP', '0.5'))
    top2_factor = float(os.environ.get('BANK_TOP2_FACTOR', '0.7'))
    ranking_mode = os.environ.get('BANK_RANKING_MODE', 'gt_only')

    batch_size = logits.size(0)
    num_classes = logits.size(1)

    # ========== 验证batch size ==========
    if batch_size % chain_length != 0:
        raise ValueError(
            f"Batch size ({batch_size}) must be divisible by chain_length ({chain_length})"
        )

    num_sequences = batch_size // chain_length

    # ========== Part 1: 提取GT样本（修复） ==========
    # 直接使用每个sequence的第一个样本作为GT
    # 而不是依赖label==1来查找

    # Reshape logits: [num_sequences, chain_length, num_classes]
    logits_reshaped = logits.view(num_sequences, chain_length, num_classes)

    # 提取第一个位置的logits作为GT样本
    gt_logits = logits_reshaped[:, 0, :]  # [num_sequences, num_classes]

    # 获取GT类别
    gt_classes = torch.argmax(gt_logits, dim=-1)  # [num_sequences]

    # ========== Part 2: CE损失 ==========
    ce_loss = nn.CrossEntropyLoss()(gt_logits, gt_classes)

    # ========== Part 3: 提取Ranking Scores ==========
    if ranking_mode == 'gt_only':
        # 使用GT类别的logit作为ranking score
        gt_classes_expanded = gt_classes.unsqueeze(1).unsqueeze(2)  # [num_sequences, 1, 1]
        gt_classes_expanded = gt_classes_expanded.expand(-1, chain_length, -1)  # [num_sequences, chain_length, 1]
        ranking_scores = torch.gather(logits_reshaped, 2, gt_classes_expanded).squeeze(-1)
        # ranking_scores: [num_sequences, chain_length]

    elif ranking_mode == 'top_k':
        top_k = min(5, num_classes)
        top_k_logits, _ = torch.topk(logits_reshaped, k=top_k, dim=2)
        ranking_scores = top_k_logits.mean(dim=2)

    elif ranking_mode == 'weighted':
        probs = torch.softmax(logits_reshaped, dim=2)
        ranking_scores = (probs * logits_reshaped).sum(dim=2)

    else:
        raise ValueError(f"Unknown ranking_mode: {ranking_mode}")

    # ========== Part 4: 构建理想分布 ==========
    ideal_scores = torch.zeros(chain_length, dtype=ranking_scores.dtype, device=ranking_scores.device)

    # 第1个位置：权重1.0
    ideal_scores[0] = 1.0

    # 第2个位置：权重top2_factor（默认0.7）
    ideal_scores[1] = top2_factor

    # 其他位置：指数递减
    if chain_length > 2:
        decay_positions = torch.arange(2, chain_length, dtype=ranking_scores.dtype, device=ranking_scores.device)
        ideal_scores[2:] = top2_factor * torch.exp(-(decay_positions - 2) * 0.5)

    # 扩展到所有序列
    ideal_scores = ideal_scores.unsqueeze(0).expand(num_sequences, -1)

    # ========== Part 5: KL散度（Ranking Loss） ==========
    pred_probs = torch.softmax(ranking_scores / temperature, dim=1)
    ideal_probs = torch.softmax(ideal_scores / temperature, dim=1)

    # KL散度：D_KL(ideal || pred)
    ranking_loss = nn.KLDivLoss(reduction='batchmean')(
        torch.log(pred_probs + 1e-10),
        ideal_probs
    )

    # ========== 组合损失 ==========
    total_loss = ce_weight * ce_loss + ranking_weight * ranking_loss

    return total_loss

def pricing_focal_loss(
        outputs,
        labels,
        loss_scale=None,
        num_items_in_batch=None,
        **kwargs
) -> torch.Tensor:
    """
    Pricing Focal Loss + ListMLE Loss（二分类使用Focal Loss处理类别不平衡）

    数据格式：
    - batch_size是8的倍数（如8, 16, 24...）
    - 每8个样本为一组，对应同一用户
    - 在每组内，位置0→7，接受概率递增
      位置0: 低折扣(d0)  → 接受概率低 → score应该低
      位置7: 高折扣(d140) → 接受概率高 → score应该高
    - label用于Focal loss，ListMLE部分不使用label

    类别分布：
    - 正类(1) : 负类(0) ≈ 1:10
    - 使用Focal Loss来更关注正类（少数类）

    优化目标：
    - Focal Loss：处理类别不平衡，更关注难分类样本和正类
    - ListMLE：最大化从高到低排序 [7,6,5,4,3,2,1,0] 的概率

    环境变量：
    - PRICING_CHAIN_LENGTH: 每组样本数（默认8）
    - PRICING_FOCAL_WEIGHT: Focal Loss权重（默认0.85）
    - PRICING_LISTMLE_WEIGHT: ListMLE权重（默认0.15）
    - PRICING_LISTMLE_TEMP: 温度参数（默认1.0）
    - PRICING_FOCAL_ALPHA: Focal Loss的alpha参数，正类权重（默认0.75）
    - PRICING_FOCAL_GAMMA: Focal Loss的gamma参数，聚焦参数（默认2.0）

    Focal Loss公式：
        FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
        其中 p_t 是正确类别的预测概率
    """
    logits = outputs.logits

    # ========== 1. 处理logits形状 ==========
    if len(logits.shape) == 2:
        if logits.shape[1] == 1:
            scores = logits.squeeze(-1)
            use_bce = True
            use_3class = False
        elif logits.shape[1] == 2:
            scores = logits[:, 1]  # 正类的logit
            use_bce = False
            use_3class = False
        elif logits.shape[1] == 3:
            # 三分类支持 - 使用对应类别的logit作为ranking score
            scores = torch.gather(logits, dim=1, index=labels.unsqueeze(1)).squeeze(1)
            use_bce = False
            use_3class = True
        else:
            raise ValueError(f"Unexpected logits shape: {logits.shape}")
    else:
        scores = logits
        use_bce = True
        use_3class = False

    labels_for_focal = labels.float()

    # ========== 2. 读取配置 ==========
    chain_length = int(os.environ.get('PRICING_CHAIN_LENGTH', '8'))
    focal_weight = float(os.environ.get('PRICING_FOCAL_WEIGHT', '0.85'))
    listmle_weight = float(os.environ.get('PRICING_LISTMLE_WEIGHT', '0.15'))
    temperature = float(os.environ.get('PRICING_LISTMLE_TEMP', '1.0'))

    # Focal Loss参数
    focal_alpha = float(os.environ.get('PRICING_FOCAL_ALPHA', '0.85'))  # 正类权重，范围[0,1]
    focal_gamma = float(os.environ.get('PRICING_FOCAL_GAMMA', '2.0'))  # 聚焦参数，通常为2

    batch_size = scores.size(0)

    if batch_size % chain_length != 0:
        raise ValueError(
            f"Batch size ({batch_size}) must be divisible by chain_length ({chain_length})"
        )

    # ========== 3. Focal Loss（针对二分类） ==========
    if use_3class:
        # 三分类暂时使用标准CE，可以后续扩展为多分类Focal Loss
        focal_loss = nn.CrossEntropyLoss()(logits, labels.long())
    else:
        # 二分类使用Focal Loss
        if use_bce or logits.shape[1] == 1:
            # 单输出logit的情况
            logits_for_focal = logits.squeeze(-1) if len(logits.shape) == 2 else logits
        else:
            # 双输出logit的情况，提取正类logit
            logits_for_focal = logits[:, 1] - logits[:, 0]  # log(p1/p0)

        # 计算概率 p
        probs = torch.sigmoid(logits_for_focal)

        # 计算 p_t：正确类别的预测概率
        # 当label=1时，p_t=p；当label=0时，p_t=1-p
        p_t = probs * labels_for_focal + (1 - probs) * (1 - labels_for_focal)

        # 计算 α_t：正类使用alpha，负类使用1-alpha
        # alpha越大，对正类的关注越多（适合正类样本少的情况）
        alpha_t = focal_alpha * labels_for_focal + (1 - focal_alpha) * (1 - labels_for_focal)

        # 计算 Focal Loss
        # FL = -α_t * (1 - p_t)^γ * log(p_t)
        focal_weight_factor = (1 - p_t) ** focal_gamma
        focal_loss = -alpha_t * focal_weight_factor * torch.log(p_t + 1e-8)
        focal_loss = focal_loss.mean()

    # ========== 3.5. 计算预测标签和统计信息 ==========
    with torch.no_grad():  # 不需要梯度
        if use_3class:
            # 三分类：获取概率最大的类别
            predicted_labels = torch.argmax(logits, dim=-1)  # [batch_size]
        elif use_bce or logits.shape[1] == 1:
            # 二分类 BCE：概率 > 0.5 为正类
            predicted_labels = (torch.sigmoid(logits.squeeze(-1) if len(logits.shape) == 2 else logits) > 0.5).long()
        else:
            # 二分类双输出：获取概率最大的类别
            predicted_labels = torch.argmax(logits, dim=-1)

        # 真实标签
        true_labels = labels.long()

        # 计算批次准确率
        correct = (predicted_labels == true_labels).float()
        batch_accuracy = correct.mean().item()

        # 计算每个类别的准确率、样本数和召回率
        label_stats = {}
        num_classes = 3 if use_3class else 2

        for label_idx in range(num_classes):
            # 找到该类别的所有样本（真实标签）
            label_mask = (true_labels == label_idx)
            label_count = label_mask.sum().item()

            if label_count > 0:
                # 计算该类别的准确率（在该类别中预测正确的比例）
                label_correct = correct[label_mask].sum().item()
                label_accuracy = label_correct / label_count

                # 计算该类别的召回率（有多少该类别的样本被正确预测）
                predicted_as_label = (predicted_labels == label_idx)
                true_positive = (label_mask & predicted_as_label).sum().item()
                recall = true_positive / label_count
            else:
                label_accuracy = 0.0
                recall = 0.0

            label_stats[f'label_{label_idx}_accuracy'] = label_accuracy
            label_stats[f'label_{label_idx}_recall'] = recall
            label_stats[f'label_{label_idx}_count'] = label_count

    # ========== 4. ListMLE损失（不使用label，只用位置） ==========
    # num_sequences = batch_size // chain_length
    # total_listmle = 0.0
    #
    # for seq_idx in range(num_sequences):
    #     start = seq_idx * chain_length
    #     end = start + chain_length
    #
    #     seq_scores = scores[start:end]  # [chain_length]
    #
    #     # 温度缩放
    #     if temperature != 1.0:
    #         seq_scores = seq_scores / temperature
    #
    #     # 🔄 关键：反转scores，使得理想排序变为 [7,6,5,4,3,2,1,0]
    #     # 原始：seq_scores[0] 对应位置0 (低折扣)
    #     #      seq_scores[7] 对应位置7 (高折扣)
    #     # 反转后：reversed_scores[0] 对应位置7 (高折扣) ← 应该最先被选
    #     #        reversed_scores[7] 对应位置0 (低折扣) ← 应该最后被选
    #     reversed_scores = torch.flip(seq_scores, dims=[0])
    #
    #     # 计算ListMLE损失
    #     # 理想排序：[位置7, 位置6, ..., 位置1, 位置0]
    #     # Loss = -Σ_{i=0}^{n-1} [reversed_scores[i] - log_sum_exp(reversed_scores[i:])]
    #     listmle = 0.0
    #     for i in range(chain_length):
    #         numerator = reversed_scores[i]
    #         remaining_scores = reversed_scores[i:]
    #         denominator = torch.logsumexp(remaining_scores, dim=0)
    #         listmle -= (numerator - denominator)
    #
    #     total_listmle += listmle
    #
    # # 平均ListMLE损失
    # avg_listmle = total_listmle / num_sequences

    # ========== 5. 组合损失与日志记录 ==========
    total_loss = focal_weight * focal_loss# + listmle_weight * avg_listmle

    # 构建日志字典
    log_dict = {
        "focal_loss": to_scalar(focal_loss),
        #"listmle_loss": to_scalar(avg_listmle),
        "total_loss": to_scalar(total_loss),
        "batch_accuracy": batch_accuracy,
    }

    # 添加每个类别的统计信息
    log_dict.update(label_stats)

    # 记录到 swanlab
    if should_log_to_swanlab():
        swanlab.log(log_dict)

    return total_loss


def ce_loss_func(outputs, labels, **kwargs):
    """Cross entropy loss function that returns per-token losses and masks.

    Args:
        outputs: The model outputs containing logits
        labels: The labels tensor [batch_size, seq_len]

    Returns:
        loss: Per-token cross entropy loss (only for valid tokens where label != -100)
        masks: Boolean mask indicating valid positions (label != -100)
    """
    logits = outputs.logits
    # Upcast to float if we need to compute the loss to avoid potential precision issues
    logits = logits.float()

    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Flatten the tokens
    shift_logits = shift_logits.view(-1, shift_logits.shape[-1])
    shift_labels = shift_labels.view(-1)

    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)

    # Compute per-token loss without reduction
    loss_fct = CrossEntropyLoss(reduction='none', ignore_index=-100)
    loss = loss_fct(shift_logits, shift_labels)

    # Create mask for valid tokens (label != -100)
    masks = shift_labels != -100

    # Only keep losses for valid tokens
    loss = loss[masks]

    return loss, masks


def scale_loss_func(outputs, labels, loss_scale=None, num_items_in_batch=None, **kwargs) -> torch.Tensor:
    """Loss func with token-level scaling based on configuration.

    This function applies different loss weights to different parts of the response
    based on the loss_scale configuration (e.g., ignore_empty_think.json).
    It protects model's thinking ability by setting zero weights for empty think tags.

    Args:
        outputs: The model outputs containing logits
        labels: The labels tensor [batch_size, seq_len]
        loss_scale: Loss scale tensor [batch_size, seq_len] with weights for each token
        num_items_in_batch: Number of tokens in the labels of gradient accumulation round that are not -100.

    Returns:
        Scalar loss value
    """
    loss, masks = ce_loss_func(outputs, labels)

    if loss_scale is not None:
        # Shift loss_scale to match the shifted labels (skip first token)
        shift_scale = loss_scale[..., 1:].contiguous().to(masks.device)

        # Flatten
        shift_scale = shift_scale.view(-1)

        # ===== DEBUG: Check for shape mismatch =====
        if shift_scale.shape[0] != masks.shape[0]:
            import logging
            logger = logging.getLogger(__name__)

            logger.warning("=" * 70)
            logger.warning(f"[DEBUG] Shape mismatch detected in scale_loss_func!")
            logger.warning(f"[DEBUG] Original shapes:")
            logger.warning(f"  labels.shape: {labels.shape}")
            logger.warning(f"  loss_scale.shape: {loss_scale.shape}")
            logger.warning(f"  outputs.logits.shape: {outputs.logits.shape}")

            logger.warning(f"[DEBUG] After shift and flatten:")
            logger.warning(f"  shift_scale.shape: {shift_scale.shape}")
            logger.warning(f"  masks.shape: {masks.shape}")
            logger.warning(f"  Difference: {shift_scale.shape[0] - masks.shape[0]} tokens")

            # Per-sample analysis
            if labels.dim() == 2:
                batch_size = labels.shape[0]
                logger.warning(f"[DEBUG] Per-sample analysis (batch_size={batch_size}):")
                for i in range(min(batch_size, 3)):  # Only print first 3 samples
                    label_len = labels[i].shape[0]
                    scale_len = loss_scale[i].shape[0]
                    valid_labels = (labels[i] != -100).sum().item()
                    logger.warning(f"  Sample {i}: label_len={label_len}, scale_len={scale_len}, "
                                 f"valid_labels={valid_labels}, diff={scale_len - label_len}")
            logger.warning("=" * 70)
        # ===========================================

        # Check and align length if mismatch
        if shift_scale.shape[0] != masks.shape[0]:
            if shift_scale.shape[0] > masks.shape[0]:
                # Truncate
                shift_scale = shift_scale[:masks.shape[0]]
            else:
                # Pad with default weight 1.0
                padding_size = masks.shape[0] - shift_scale.shape[0]
                padding = torch.ones(
                    padding_size,
                    dtype=shift_scale.dtype,
                    device=shift_scale.device
                )
                shift_scale = torch.cat([shift_scale, padding])

        # Apply mask to get scales for valid tokens only
        shift_scale = shift_scale[masks]

        # ========== Visualize loss_scale distribution ==========
        with torch.no_grad():
            # Get unique scale values and their counts
            unique_scales = shift_scale.unique()
            scale_distribution = {}

            for scale_value in unique_scales:
                count = (shift_scale == scale_value).sum().item()
                scale_key = f"scale_{float(scale_value):.1f}"
                scale_distribution[f"loss_scale/{scale_key}_count"] = count

                # Calculate percentage
                total_tokens = len(shift_scale)
                scale_distribution[f"loss_scale/{scale_key}_pct"] = 100.0 * count / total_tokens if total_tokens > 0 else 0.0

                # Calculate weighted loss contribution for this scale
                scale_mask = shift_scale == scale_value
                if scale_mask.any():
                    scale_loss = (loss[scale_mask] * scale_value).sum().item()
                    scale_distribution[f"loss_scale/{scale_key}_loss_contrib"] = scale_loss

            # Add total token count
            scale_distribution["loss_scale/total_valid_tokens"] = len(shift_scale)

            # Log to swanlab
            if should_log_to_swanlab():
                swanlab.log(scale_distribution)
        # =======================================================

        # Apply loss scaling: zero weight for empty think, higher weight for answer
        loss = shift_scale * loss

    if num_items_in_batch is None:
        loss = loss.mean()
    else:
        # compat transformers>=4.46
        loss = loss.sum() / num_items_in_batch
    return loss



def scale_per_token_loss_func(outputs, labels, loss_scale=None, enable_dft_loss: bool = False, **kwargs):
    logits = outputs.logits
    # Upcast to float if we need to compute the loss to avoid potential precision issues
    logits = logits.float()
    labels = torch.roll(labels, shifts=-1, dims=-1).view(-1)

    # Flatten the tokens
    logits = logits.view(-1, logits.shape[-1])
    # Enable model parallelism
    labels = labels.to(logits.device)
    loss = F.cross_entropy(logits, labels, ignore_index=-100, reduction='none')
    if enable_dft_loss:
        with torch.no_grad():
            target_probs = torch.exp(-loss)
        loss *= target_probs

    # Apply loss scaling if provided
    if loss_scale is not None:
        # Shift loss_scale to align with rolled labels
        shift_scale = torch.roll(loss_scale, shifts=-1, dims=-1).view(-1).to(loss.device)

        # Handle shape mismatch (edge case)
        if shift_scale.shape[0] != loss.shape[0]:
            if shift_scale.shape[0] > loss.shape[0]:
                # Truncate extra positions
                shift_scale = shift_scale[:loss.shape[0]]
            else:
                # Pad with default weight 1.0
                padding_size = loss.shape[0] - shift_scale.shape[0]
                padding = torch.ones(padding_size, dtype=shift_scale.dtype, device=shift_scale.device)
                shift_scale = torch.cat([shift_scale, padding])

        # Apply scaling: zero weight for empty think, higher weight for answer
        loss = loss * shift_scale

    return loss

def scale_cross_entropy_loss_func(outputs, labels, loss_scale=None, num_items_in_batch=None, **kwargs):
    """Cross entropy loss function with token-level scaling.

    This function applies different loss weights to different parts of the response
    based on the loss_scale configuration (e.g., ignore_empty_think.json).

    Args:
        outputs: The model outputs containing logits
        labels: The labels tensor [batch_size, seq_len]
        loss_scale: Loss scale tensor [batch_size, seq_len] with weights for each token
        num_items_in_batch: Number of tokens that are not -100 for normalization

    Returns:
        Scalar loss value
    """
    # ========== Log loss_scale distribution to swanlab (BEFORE scaling) ==========
    if loss_scale is not None:
        # Get unscaled per-token loss for logging
        token_loss_unscaled = scale_per_token_loss_func(outputs, labels, loss_scale=None, **kwargs)

        with torch.no_grad():
            # Shift loss_scale to align with rolled labels
            shift_scale = torch.roll(loss_scale, shifts=-1, dims=-1).view(-1)

            # Filter out positions with -100 labels (padding/input tokens)
            rolled_labels = torch.roll(labels, shifts=-1, dims=-1).view(-1)
            valid_mask = rolled_labels != -100

            # Also filter token_loss_unscaled to only valid positions
            # token_loss_unscaled has shape [batch*seq_len], with 0 loss at -100 positions
            # But we need to filter it to match shift_scale_valid
            token_loss_unscaled_valid = token_loss_unscaled[valid_mask]
            shift_scale_valid = shift_scale[valid_mask]

            # Get unique scale values and their counts
            unique_scales = shift_scale_valid.unique()
            scale_distribution = {}

            for scale_value in unique_scales:
                count = (shift_scale_valid == scale_value).sum().item()
                scale_key = f"scale_{float(scale_value):.1f}"

                # Count of tokens with this scale
                scale_distribution[f"loss_scale/{scale_key}_count"] = count

                # Percentage of tokens with this scale
                total_tokens = len(shift_scale_valid)
                scale_distribution[f"loss_scale/{scale_key}_pct"] = 100.0 * count / total_tokens if total_tokens > 0 else 0.0

                # Calculate weighted loss contribution for this scale
                # Use unscaled loss and multiply by scale_value
                scale_mask = shift_scale_valid == scale_value
                if scale_mask.any():
                    scale_loss = (token_loss_unscaled_valid[scale_mask] * scale_value).sum().item()
                    scale_distribution[f"loss_scale/{scale_key}_loss_contrib"] = scale_loss

            # Add total valid token count
            scale_distribution["loss_scale/total_valid_tokens"] = len(shift_scale_valid)

            # Log to swanlab
            if should_log_to_swanlab():
                swanlab.log(scale_distribution)
    # ==========================================================

    # Get per-token loss with scaling applied
    token_loss = scale_per_token_loss_func(outputs, labels, loss_scale=loss_scale, **kwargs)

    # Aggregate loss
    if num_items_in_batch is None:
        num_items_in_batch = (labels[:, 1:] != -100).sum()
    return token_loss.sum() / num_items_in_batch

def detect_category_from_loss_scale(loss_scale_tensor):
    """从 loss_scale 张量检测样本类别

    根据 ignore_empty_think.json 中的配置：
    - 曝光：<answer> 内容权重为 100.0
    - 领取：<answer> 内容权重为 300.0
    - 核销：<answer> 内容权重为 400.0

    参数：
        loss_scale_tensor: 单个样本的 loss_scale 张量 [seq_len]

    返回：
        类别名称（'曝光'/'领取'/'核销'）或 None
    """

    # 将 loss_scale_tensor 移到 CPU 并转换为 float（避免设备和 dtype 问题）
    scale_values = loss_scale_tensor.cpu().float()
    debug_dict = {"scale_values": scale_values}

    # 检查是否包含特定的权重值（使用容差处理浮点数精度）
    has_400 = torch.any(torch.isclose(scale_values, torch.tensor(4000.0), rtol=1e-5))
    has_300 = torch.any(torch.isclose(scale_values, torch.tensor(2000.0), rtol=1e-5))
    has_100 = torch.any(torch.isclose(scale_values, torch.tensor(1000.0), rtol=1e-5))

    # 优先级：核销 > 领取 > 曝光（因为权重值更大更特异）
    if has_400:
        category = '核销'
    elif has_300:
        category = '领取'
    elif has_100:
        category = '曝光'
    else:
        category = None

    return category, debug_dict

def category_weighted_loss_func(outputs, labels, loss_scale=None, num_items_in_batch=None, trainer=None, **kwargs):
    """带有 token 级缩放和样本级类别加权的交叉熵损失函数。

    此函数应用：
    1. 来自 loss_scale 配置的 token 级权重
    2. 基于 <answer> 内容的样本级类别权重

    类别权重通过环境变量配置：
    - CATEGORY_WEIGHT_曝光（默认：1.0）
    - CATEGORY_WEIGHT_领取（默认：10.0）
    - CATEGORY_WEIGHT_核销（默认：100.0）

    参数：
        outputs: 包含 logits 的模型输出
        labels: Labels 张量 [batch_size, seq_len]
        loss_scale: Loss scale 张量 [batch_size, seq_len]，每个 token 的权重
        num_items_in_batch: 用于归一化的非 -100 token 数量
        trainer: Trainer 实例，用于访问 tokenizer

    返回：
        标量 loss 值
    """
    import os

    # 从环境变量读取类别权重
    category_weights = {
        '曝光': float(os.environ.get('CATEGORY_WEIGHT_曝光', '1.0')),
        '领取': float(os.environ.get('CATEGORY_WEIGHT_领取', '10.0')),
        '核销': float(os.environ.get('CATEGORY_WEIGHT_核销', '100.0')),
    }

    # ========== 记录 loss_scale 分布到 swanlab（缩放前）==========
    if loss_scale is not None:
        # 获取未缩放的 per-token loss 用于日志记录
        token_loss_unscaled = scale_per_token_loss_func(outputs, labels, loss_scale=None, **kwargs)

        with torch.no_grad():
            # 移位 loss_scale 以对齐 rolled labels
            shift_scale = torch.roll(loss_scale, shifts=-1, dims=-1).view(-1)

            # 过滤掉 -100 labels 的位置（填充/输入 token）
            rolled_labels = torch.roll(labels, shifts=-1, dims=-1).view(-1)
            valid_mask = rolled_labels != -100

            token_loss_unscaled_valid = token_loss_unscaled[valid_mask]
            shift_scale_valid = shift_scale[valid_mask]

            # 获取唯一的 scale 值及其计数
            unique_scales = shift_scale_valid.unique()
            scale_distribution = {}

            for scale_value in unique_scales:
                count = (shift_scale_valid == scale_value).sum().item()
                scale_key = f"scale_{float(scale_value):.1f}"

                # 具有此 scale 的 token 计数
                scale_distribution[f"loss_scale/{scale_key}_count"] = count

                # 具有此 scale 的 token 百分比
                total_tokens = len(shift_scale_valid)
                scale_distribution[f"loss_scale/{scale_key}_pct"] = 100.0 * count / total_tokens if total_tokens > 0 else 0.0

                # 计算此 scale 的加权 loss 贡献
                scale_mask = shift_scale_valid == scale_value
                if scale_mask.any():
                    scale_loss = (token_loss_unscaled_valid[scale_mask] * scale_value).sum().item()
                    scale_distribution[f"loss_scale/{scale_key}_loss_contrib"] = scale_loss

            # 添加总有效 token 计数
            scale_distribution["loss_scale/total_valid_tokens"] = len(shift_scale_valid)

            # 记录到 swanlab
            if should_log_to_swanlab():
                swanlab.log(scale_distribution)
    # ==========================================================

    # 获取应用缩放后的 per-token loss
    token_loss = scale_per_token_loss_func(outputs, labels, loss_scale=loss_scale, **kwargs)

    # ========== 应用样本级类别加权 ==========
    if loss_scale is not None:
        batch_size = labels.shape[0]

        # 将 token_loss 重塑为 [batch_size, seq_len-1]
        token_loss_2d = token_loss.view(batch_size, -1)

        # 将 loss_scale 重塑为 2D，确保与 token_loss_2d 形状一致
        if loss_scale.dim() == 1:
            # loss_scale 是 1D [total_tokens]，重塑为 [batch_size, seq_len]
            loss_scale_2d = loss_scale.view(batch_size, -1)
        elif loss_scale.dim() == 2:
            # loss_scale 已经是 2D
            loss_scale_2d = loss_scale
        else:
            # 意外的形状，尝试重塑
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Unexpected loss_scale shape: {loss_scale.shape}, dim={loss_scale.dim()}")
            loss_scale_2d = loss_scale.view(batch_size, -1)

        # 对每个样本检测类别并应用权重
        category_stats = {'曝光': 0, '领取': 0, '核销': 0, 'unknown': 0}
        weighted_losses = []

        for i in range(batch_size):
            # 从 loss_scale 检测类别（新方法）
            # 重要：对 loss_scale 进行 roll 对齐，与前面的 loss_scale 分布日志保持一致
            # 使用 loss_scale_2d[i] 获取单个样本的 loss_scale (1D tensor)
            sample_loss_scale = loss_scale_2d[i]  # [seq_len]

            # Roll 对齐（对 1D tensor 使用 dims=0）
            sample_loss_scale = torch.roll(sample_loss_scale, shifts=-1, dims=0)

            category, _ = detect_category_from_loss_scale(sample_loss_scale)

            sample_loss = token_loss_2d[i]

            if category in category_weights:
                weight = category_weights[category]
                weighted_sample_loss = sample_loss * weight
                category_stats[category] += 1
            else:
                # 未检测到类别，使用默认权重 1.0
                weighted_sample_loss = sample_loss
                category_stats['unknown'] += 1

            weighted_losses.append(weighted_sample_loss)

        # 连接回 1D
        token_loss = torch.cat(weighted_losses, dim=0)

        # 记录类别分布到 swanlab
        if should_log_to_swanlab():
            with torch.no_grad():
                category_log = {
                    'category/曝光_count': category_stats['曝光'],
                    'category/领取_count': category_stats['领取'],
                    'category/核销_count': category_stats['核销'],
                    'category/unknown_count': category_stats['unknown'],
                    'category/曝光_weight': category_weights['曝光'],
                    'category/领取_weight': category_weights['领取'],
                    'category/核销_weight': category_weights['核销'],
                }
                swanlab.log(category_log)
    else:
        # 如果没有 loss_scale，无法检测类别
        pass
    # ==========================================================

    # 聚合 loss
    if num_items_in_batch is None:
        num_items_in_batch = (labels[:, 1:] != -100).sum()
    return token_loss.sum() / num_items_in_batch

def extract_category_from_tokens(token_ids, tokenizer):
    """使用正则表达式从 <answer> 标签中提取类别

    参数：
        token_ids: Token IDs 张量或列表
        tokenizer: Tokenizer 实例

    返回：
        类别名称（'曝光'/'领取'/'核销'）或 None
    """
    try:
        # 解码 token IDs 为文本
        if isinstance(token_ids, torch.Tensor):
            token_ids_list = token_ids.cpu().tolist()
        else:
            token_ids_list = token_ids

        text = tokenizer.decode(token_ids_list, skip_special_tokens=False)

        # 使用正则表达式从 <answer> 标签中提取类别
        # 优先级：核销 > 领取 > 曝光（因为可能包含多个词）
        patterns = [
            (r'<answer>([\s\S]*?核销[\s\S]*?)</answer>', '核销'),
            (r'<answer>([\s\S]*?领取[\s\S]*?)</answer>', '领取'),
            (r'<answer>([\s\S]*?曝光[\s\S]*?)</answer>', '曝光'),
        ]

        # Debug: 保存 token IDs 信息
        debug_token_info = {
            'token_ids_type': str(type(token_ids)),
            'token_ids_shape': str(token_ids.shape) if isinstance(token_ids,
                                                                  torch.Tensor) else f'list_len_{len(token_ids_list)}',
            'token_ids_sample': token_ids_list[:50] if len(token_ids_list) > 50 else token_ids_list,
            'token_ids_min': min(token_ids_list) if len(token_ids_list) > 0 else None,
            'token_ids_max': max(token_ids_list) if len(token_ids_list) > 0 else None,
            'vocab_size': tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else 'unknown',
            'text': text
        }

        for pattern, category in patterns:
            match = re.search(pattern, text)
            if match:
                return category, debug_token_info

        # 提取失败
        return None, debug_token_info
    except Exception as e:
        import traceback
        return None, {'error_extract_category_from_tokens': traceback.format_exc()}

def category_weighted_loss_with_aux_classification(outputs, labels, loss_scale=None, num_items_in_batch=None, trainer=None, **kwargs):
    """带有辅助分类Loss的类别加权损失函数

    结合token级别loss和分类级别loss，直接优化分类准确率。

    参数：
        outputs: 包含 logits 的模型输出
        labels: Labels 张量 [batch_size, seq_len]
        loss_scale: Loss scale 张量 [batch_size, seq_len]
        num_items_in_batch: 用于归一化的非 -100 token 数量
        trainer: Trainer 实例，用于访问 tokenizer

    返回：
        total_loss = token_loss + alpha * classification_loss
    """
    import os

    if trainer is None:
        raise ValueError('trainer is required for category_weighted_loss_with_aux_classification')

    tokenizer = trainer.processing_class

    # 读取配置
    alpha = float(os.environ.get('AUX_CLASSIFICATION_ALPHA', '0.5'))
    category_weights = {
        '曝光': float(os.environ.get('CATEGORY_WEIGHT_曝光', '1.0')),
        '领取': float(os.environ.get('CATEGORY_WEIGHT_领取', '10.0')),
        '核销': float(os.environ.get('CATEGORY_WEIGHT_核销', '100.0')),
    }

    # ========== 1. 计算 token 级别 loss（已包含 sample-level weight）==========
    token_loss = category_weighted_loss_func(outputs, labels, loss_scale, num_items_in_batch, trainer, **kwargs)

    # ========== 2. 提取预测类别和真实类别 ==========
    if loss_scale is None:
        # 没有 loss_scale，无法进行分类
        return token_loss

    batch_size = labels.shape[0]
    logits = outputs.logits

    # 获取预测的 token IDs（argmax）
    pred_token_ids = torch.argmax(logits, dim=-1)  # [batch_size, seq_len]

    true_categories = []
    pred_categories = []
    sample_weights = []
    cate_debugs = []
    pred_dicts = []

    # 处理 loss_scale 形状
    if loss_scale.dim() == 1:
        loss_scale_2d = loss_scale.view(batch_size, -1)
    elif loss_scale.dim() == 2:
        loss_scale_2d = loss_scale
    else:
        loss_scale_2d = loss_scale.view(batch_size, -1)

    for i in range(batch_size):
        # 提取真实类别（从 loss_scale）
        sample_loss_scale = loss_scale_2d[i]
        sample_loss_scale_rolled = torch.roll(sample_loss_scale, shifts=-1, dims=0)
        true_cat, cate_debug_dict = detect_category_from_loss_scale(sample_loss_scale_rolled)
        cate_debugs.append(cate_debug_dict)

        # 提取预测类别（从 pred_token_ids）
        # 只使用有效的 token（labels != -100）
        sample_labels = labels[i]
        valid_mask = sample_labels != -100
        valid_pred_tokens = pred_token_ids[i][valid_mask]
        pred_cat, pred_dict = extract_category_from_tokens(valid_pred_tokens, tokenizer)
        pred_dicts.append(pred_dict)

        true_categories.append(true_cat)
        pred_categories.append(pred_cat)

        # 获取 sample weight
        if true_cat in category_weights:
            sample_weights.append(category_weights[true_cat])
        else:
            sample_weights.append(1.0)

    # ========== 3. 计算分类 loss ==========
    classification_losses = []
    correct_count = 0
    total_count = 0

    for i in range(batch_size):
        true_cat = true_categories[i]
        pred_cat = pred_categories[i]

        if true_cat is None or pred_cat is None:
            # 提取失败，跳过
            continue

        total_count += 1

        # 简化的 0/1 loss
        if pred_cat == true_cat:
            loss_value = 0.0
            correct_count += 1
        else:
            loss_value = 1.0

        # 应用 sample-level weight
        weighted_loss = loss_value * sample_weights[i]
        classification_losses.append(weighted_loss)

    # 聚合分类 loss
    if len(classification_losses) > 0:
        classification_loss = sum(classification_losses) / len(classification_losses)
        classification_loss = torch.tensor(classification_loss, device=token_loss.device, dtype=token_loss.dtype)
    else:
        classification_loss = torch.tensor(0.0, device=token_loss.device, dtype=token_loss.dtype)

    # ========== 4. 组合 loss ==========
    total_loss = token_loss + alpha * classification_loss

    # ========== 5. 记录到 swanlab ==========
    if should_log_to_swanlab():
        with torch.no_grad():
            swanlab.log({
                'loss/token_loss': to_scalar(token_loss),
                'loss/classification_loss': to_scalar(classification_loss),
                'loss/total_loss': to_scalar(total_loss),
                'loss/alpha': alpha,
                'category/correct_count': correct_count,
                'category/total_count': total_count,
                'category/accuracy': correct_count / total_count if total_count > 0 else 0.0,
            })

    # ========== 6. 保存 debug 信息 ==========
    try:
        debug_info = {
            'true_categories': true_categories,
            'pred_categories': pred_categories,
            'token_loss': to_scalar(token_loss),
            'classification_loss': to_scalar(classification_loss),
            'total_loss': to_scalar(total_loss),
            'sample_weights': sample_weights,
            'correct_count': correct_count,
            'total_count': total_count,
            'pred_dicts': pred_dicts,
            'cate_debugs': cate_debugs,
        }

        with open('/tmp/aux_classification_debug.jsonl', 'a') as f:
            f.write(json.dumps(debug_info, ensure_ascii=False) + '\n')
    except Exception as e:
        pass  # 忽略 debug 文件写入错误

    return total_loss

##################

loss_mapping = {
    'cross_entropy': cross_entropy_loss_func,  # examples
    # embedding
    'cosine_similarity': cosine_similarity_func,
    'contrastive': contrastive_loss,
    'online_contrastive': online_contrastive_loss,
    'infonce': infonce_loss,
    # reranker
    'reranker': reranker_loss,
    'generative_reranker': generative_reranker_loss,
    'listwise_reranker': listwise_reranker_loss,
    'listwise_generative_reranker': listwise_generative_reranker_loss,
    # seq-cls + listwise
    'seqcls_plus_listwise_rank': seqcls_plus_listwise_rank_loss,
    # seq-cls + pairwise
    'seqcls_plus_pairwise_rank': seqcls_plus_pairwise_rank_loss,
    'seqcls_plus_pairwise_rank_loss_threshold': seqcls_plus_pairwise_rank_loss_threshold,
    'pricing_pairwise_loss': pricing_pairwise_loss,
    'pricing_listwise_loss': pricing_listwise_loss,
    'bank_rank_listwise_loss': bank_rank_listwise_loss,
    'pricing_listmle_loss': pricing_listmle_loss,
    'pricing_focal_loss': pricing_focal_loss,
    'scale_loss_func': scale_loss_func,
    'scale_cross_entropy_loss_func': scale_cross_entropy_loss_func,
    'category_weighted': category_weighted_loss_func,
    'category_weighted_aux': category_weighted_loss_with_aux_classification,  # 辅助分类Loss
}



def get_loss_func(loss_type: Optional[str]) -> Optional[Callable]:
    if loss_type is None:
        return None
    return loss_mapping[loss_type]
