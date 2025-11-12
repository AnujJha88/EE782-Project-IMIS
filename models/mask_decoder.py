import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskDecoder(nn.Module):
    """
    Transformer-based mask decoder with iterative refinement
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer_dim = config.transformer_dim
        self.inter_num = config.inter_num

        self.transformer = nn.ModuleList([
            TwoWayTransformerBlock(
                embedding_dim=config.transformer_dim,
                num_heads=config.decoder_num_heads,
                mlp_dim=config.mlp_dim,
                activation=nn.ReLU,
            )
            for _ in range(config.decoder_depth)
        ])

        self.norm_final = nn.LayerNorm(config.transformer_dim)

        self.num_mask_tokens = config.num_mask_tokens
        self.iou_token = nn.Embedding(1, config.transformer_dim)
        self.mask_tokens = nn.Embedding(config.num_mask_tokens,
                                       config.transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(config.transformer_dim, config.transformer_dim // 4,
                             kernel_size=2, stride=2),
            nn.GELU(),
            nn.ConvTranspose2d(config.transformer_dim // 4, config.transformer_dim // 8,
                             kernel_size=2, stride=2),
            nn.GELU(),
        )

        self.output_hypernetworks_mlps = nn.ModuleList([
            MLP(config.transformer_dim, config.transformer_dim,
                config.transformer_dim // 8, 3)
            for _ in range(config.num_mask_tokens)
        ])

        self.iou_prediction_head = MLP(
            config.transformer_dim,
            config.iou_head_hidden_dim,
            config.num_mask_tokens,
            config.iou_head_depth
        )

    def forward(self,
                image_embeddings,
                image_pe,
                sparse_prompt_embeddings,
                dense_prompt_embeddings):
        """
        Args:
            image_embeddings: (B, C, H, W) from image encoder
            image_pe: (1, C, H, W) positional encoding
            sparse_prompt_embeddings: (B, N, C) from prompt encoder
            dense_prompt_embeddings: (B, C, H, W) from prompt encoder
        Returns:
            masks: (B, num_mask_tokens, H*4, W*4)
            iou_predictions: (B, num_mask_tokens)
        """
        B = image_embeddings.shape[0]

        output_tokens = torch.cat([
            self.iou_token.weight.unsqueeze(0).expand(B, -1, -1),
            self.mask_tokens.weight.unsqueeze(0).expand(B, -1, -1)
        ], dim=1)

        tokens = output_tokens
        if sparse_prompt_embeddings is not None:
            tokens = torch.cat([tokens, sparse_prompt_embeddings], dim=1)

        src = image_embeddings + dense_prompt_embeddings
        B, C, H, W = src.shape
        src = src.flatten(2).permute(0, 2, 1)

        pos_src = image_pe.flatten(2).permute(0, 2, 1).expand(B, -1, -1)

        hs, src = self.run_transformer(tokens, src, pos_src)

        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1:(1+self.num_mask_tokens), :]

        src = src.transpose(1, 2).view(B, C, H, W)
        upscaled_embedding = self.output_upscaling(src)

        masks = []
        for i in range(self.num_mask_tokens):
            mask_token = mask_tokens_out[:, i, :]
            mlp = self.output_hypernetworks_mlps[i]

            weights = mlp(mask_token)
            weights = weights.unsqueeze(-1).unsqueeze(-1)

            mask = (upscaled_embedding * weights).sum(dim=1, keepdim=True)
            masks.append(mask)

        masks = torch.cat(masks, dim=1)

        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred

    def run_transformer(self, tokens, src, pos_src):
        """Run the transformer decoder"""
        queries = tokens
        keys = src

        for layer in self.transformer:
            queries, keys = layer(queries, keys, pos_src)

        queries = self.norm_final(queries)

        return queries, keys

    def predict_masks_iterative(self,
                                image_embeddings,
                                image_pe,
                                prompt_encoder,
                                points=None,
                                boxes=None,
                                prev_masks=None):
        """
        Iterative refinement for inter_num iterations
        """
        masks_list = []
        iou_list = []

        current_mask = prev_masks

        for i in range(self.inter_num):
            sparse_emb, dense_emb = prompt_encoder(
                points=points,
                boxes=boxes,
                masks=current_mask
            )

            masks, iou_pred = self.forward(
                image_embeddings, image_pe, sparse_emb, dense_emb
            )

            masks_list.append(masks)
            iou_list.append(iou_pred)

            best_mask_idx = torch.argmax(iou_pred, dim=1)
            current_mask = masks[torch.arange(masks.shape[0]), best_mask_idx].unsqueeze(1)

            current_mask = F.interpolate(
                current_mask,
                size=(self.config.img_size // 4, self.config.img_size // 4),
                mode='bilinear',
                align_corners=False
            )

        return masks_list, iou_list


class TwoWayTransformerBlock(nn.Module):
    """Two-way attention block for mask decoder"""
    def __init__(self, embedding_dim, num_heads, mlp_dim, activation):
        super().__init__()

        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = Attention(embedding_dim, num_heads)
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLP(embedding_dim, mlp_dim, embedding_dim, 2, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.cross_attn_image_to_token = Attention(embedding_dim, num_heads)
        self.norm4 = nn.LayerNorm(embedding_dim)

    def forward(self, queries, keys, query_pe):
        """
        Args:
            queries: (B, N_q, C) output tokens
            keys: (B, N_k, C) image tokens
            query_pe: (B, N_k, C) positional encoding
        """
        q = queries + self.self_attn(self.norm1(queries))

        q = q + self.cross_attn_token_to_image(
            self.norm2(q), keys, keys
        )

        queries = q + self.mlp(self.norm3(q))

        k = keys + self.cross_attn_image_to_token(
            self.norm4(keys), queries, queries
        )

        return queries, k


class Attention(nn.Module):
    """Multi-head attention module"""
    def __init__(self, embedding_dim, num_heads):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        self.qkv = nn.Linear(embedding_dim, embedding_dim * 3)
        self.proj = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, q, k=None, v=None):
        """
        Args:
            q: query (B, N, C)
            k: key (B, M, C) - if None, use q
            v: value (B, M, C) - if None, use k
        """
        B, N, C = q.shape

        if k is None:
            k = q
        if v is None:
            v = k

        q = self.qkv(q)[:, :, :C].reshape(B, N, self.num_heads, self.head_dim)
        k = self.qkv(k)[:, :, C:2*C].reshape(B, -1, self.num_heads, self.head_dim)
        v = self.qkv(v)[:, :, 2*C:].reshape(B, -1, self.num_heads, self.head_dim)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 3, 1)
        v = v.permute(0, 2, 1, 3)

        attn = (q @ k) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)

        out = attn @ v
        out = out.permute(0, 2, 1, 3).reshape(B, N, C)

        out = self.proj(out)

        return out


class MLP(nn.Module):
    """Multi-layer perceptron"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, activation=nn.GELU):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.activation = activation()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.activation(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

