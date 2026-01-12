# losses/itc_loss.py
import torch
import torch.nn as nn

class ITCLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, image_emb, text_emb):
        """
        image_emb: [B, D]
        text_emb:  [B, D]
        """
        # 類似度行列 [B, B]
        sim = torch.matmul(image_emb, text_emb.T) / self.temperature

        labels = torch.arange(sim.size(0), device=sim.device)
        loss_i2t = self.criterion(sim, labels)  # image -> text
        loss_t2i = self.criterion(sim.T, labels)  # text -> image

        loss = (loss_i2t + loss_t2i) / 2
        return loss, sim

def recall_at_k(sim, k=1):
    """
    sim: [B, B] 類似度行列
    """
    B = sim.size(0)
    topk = sim.topk(k, dim=1).indices  # 各行の top-k インデックス
    labels = torch.arange(B, device=sim.device).unsqueeze(1)
    hits = (topk == labels).any(dim=1).float()
    return hits.mean().item()
