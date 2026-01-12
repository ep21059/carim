# evaluation/metrics.py
import torch
import torch.nn.functional as F

def compute_similarity(text_embs, image_embs):
    text_embs = F.normalize(text_embs, dim=1)
    image_embs = F.normalize(image_embs, dim=1)
    return text_embs @ image_embs.T

def recall_at_k(sim_matrix, gt_indices, k):
    topk = sim_matrix.topk(k, dim=1).indices
    gt_indices = gt_indices.view(-1, 1)
    correct = (topk == gt_indices).any(dim=1)
    return correct.float().mean().item()

def evaluate_recall(sim_matrix, gt_indices, ks=(1, 5, 10)):
    return {
        f"R@{k}": recall_at_k(sim_matrix, gt_indices, k)
        for k in ks
    }
