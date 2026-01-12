import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

# from models.image_encoder import ImageEncoder


class CARIMScorer(nn.Module):
    """
    CARIMのスコアリングモジュール (Text-to-Text)。
    クエリテキストと、キャプションから抽出された要素(Elements)の埋め込みベクトルの類似度を計算します。
    """
    def __init__(
        self,
        text_encoder_name="Qwen/Qwen2-1.5B-Instruct",
        embed_dim=256,
        use_projection=True
    ):
        super().__init__()

        # テキストエンコーダ (Qwen) の初期化
        self.text_encoder = AutoModel.from_pretrained(text_encoder_name, trust_remote_code=True)
        self.use_projection = use_projection
        
        # 学習時は凍結、今回は推論のみ想定だが一応
        for p in self.text_encoder.parameters():
            p.requires_grad = False
            
        # 画像エンコーダは削除 (Text-to-Text化)

        if self.use_projection:
            hidden_dim = self.text_encoder.config.hidden_size
            self.text_proj = nn.Linear(hidden_dim, embed_dim)
        else:
            self.text_proj = nn.Identity()

    def encode_text(self, input_ids, attention_mask):
        """
        クエリまたは要素テキストをエンコードします。
        """
        outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        token_emb = outputs.last_hidden_state # (B, L, H)
        
        if self.use_projection:
             token_emb = self.text_proj(token_emb) # (B, L, D)
             
        token_emb = F.normalize(token_emb, dim=-1)
        return token_emb

    def compute_similarity(self, query_emb, element_embs, query_mask, element_mask=None):
        """
        クエリと要素セット間の類似度を計算 (Inclusive Text Matching)。
        
        Args:
            query_emb: (B, Lq, D) - クエリのトークン埋め込み
            element_embs: (B, M, D) - 要素の埋め込み (M個の要素、各要素は1つのベクトル等)
                          or (B, M, Le, D) if maintaining tokens per element?
                          通常は各要素を1ベクトルに集約済みと仮定 (Mean Pooling)
            query_mask: (B, Lq)
            element_mask: (B, M) - 有効な要素のマスク
            
        Returns:
            scores: (B,)
        """
        # (B, Lq, D) x (B, D, M) -> (B, Lq, M)
        # 各クエリトークンと、各要素ベクトルの類似度
        sim = torch.bmm(query_emb, element_embs.transpose(1, 2))
        
        # マスク処理
        if element_mask is not None:
             # (B, 1, M)
             e_mask = element_mask.unsqueeze(1).bool()
             sim = sim.masked_fill(~e_mask, -1e9)
             
        # Inclusive Matching:
        # クエリの各トークンについて、最もマッチする要素(Max Similarity)を探す (ArgMax)
        
        # Max over Elements (M)
        # (B, Lq)
        # Masked elements are -1e9, so max will pick valid ones.
        matched_scores, _ = sim.max(dim=-1)
        
        # Query Mask Apply
        q_mask = query_mask.bool()
        matched_scores = matched_scores.masked_fill(~q_mask, 0.0)
        
        # Average over valid query tokens (Simple Average)
        # sum / count
        valid_counts = q_mask.sum(dim=-1).clamp(min=1.0)
        score = matched_scores.sum(dim=-1) / valid_counts
        
        return score

    def forward(self, input_ids, attention_mask):
        """
        DataParallel対応のため、forwardはencode_textへのエイリアスとします。
        学習時は model(input_ids, attention_mask) として呼び出し、並列化されたエンコーディングを行います。
        """
        return self.encode_text(input_ids, attention_mask)


