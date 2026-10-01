import torch

def mixup_tokens(tokens_s, neighbor_tokens, alpha=0.7):
    """
    tokens_s: [B, N, D]  strong tokens
    neighbor_tokens: [B, K, N, D]  neighbor tokens
    alpha: mixup 系数 (0~1)
    """
    B, K, N, D = neighbor_tokens.shape
    tokens_s_exp = tokens_s.unsqueeze(1).expand(-1, K, -1, -1)
    mixed_tokens = alpha * tokens_s_exp + (1 - alpha) * neighbor_tokens
    mixed_tokens = mixed_tokens.view(B*K, N, D)
    return mixed_tokens
