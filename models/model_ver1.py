import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from models.performer_pytorch import Performer

"""
Model architecture:
1. Image Encoder: ResNet-18 → (N_spots, embed_dim)
2. SC Encoder: scBERT Performer → (B, embed_dim)
3. ST Encoder: MLP + Positional Encoding + Transformer → (B, embed_dim)
4. Fusion Layer: concat/attn → (B, fused_dim)
5. MIL Attention Pooling: (N_spots, fused_dim) → (fused_dim)
6. Classifier: Linear(fused_dim → num_classes)
"""

# =======================================================
# 1. Image Encoder (Patch → Spot-level visual feature)
# =======================================================

class ImageEncoder(nn.Module):

    def __init__(self, embed_dim=256, backbone='resnet18', pretrained=True):

        super().__init__()

        self.backbone = models.resnet18(pretrained=pretrained)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, embed_dim)

    def forward(self, x):
        """
        x: (N_spots, 3, 224, 224)
        return: (N_spots, embed_dim)
        """
        return self.backbone(x)

# =======================================================
# 2. SC Encoder (Gene expression → Cell-level feature)
# =======================================================

class SCEncoder(nn.Module):

    def __init__(self, num_genes, embed_dim=256, top_k_genes=None):  

        super().__init__()

        self.top_k_genes = top_k_genes  

        # scBERT Performer
        self.scbert = Performer(
            dim=256,
            depth=6,
            dim_head=32,
            heads=8,
            ff_mult=4,
            causal=False,
            attn_dropout=0.1
        )

        # Token embedding + projection
        self.token_emb = nn.Embedding(7, 256)
        self.cls_token = nn.Parameter(torch.randn(1, 1, 256))
        self.proj = nn.Linear(256, embed_dim)

    def preprocess(self, expr):
        """Raw counts → 7 bins"""
        expr = torch.log1p(F.normalize(expr, p=1, dim=-1) * 1e4)
        bins = torch.linspace(0, expr.max(), 8, device=expr.device)
        tokens = torch.bucketize(expr, bins[:-1]).long()
        tokens = torch.clamp(tokens, 0, 6)
        return tokens

    def forward(self, expr):  # (B, num_genes)

        B = expr.shape[0]

        if self.top_k_genes is not None and self.top_k_genes < expr.shape[1]:

            topk_values, topk_indices = torch.topk(expr, k=self.top_k_genes, dim=1)
            expr_filtered = topk_values  # (B, K)

        else:

            expr_filtered = expr  # (B, 2000)

        # 7-bin tokenization
        tokens = self.preprocess(expr_filtered)  # (B, K or 2000)

        # CLS + tokens
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, self.token_emb(tokens)], dim=1)  # (B, K+1, 256)

        # Performer forward
        emb = self.scbert(x)  # (B, K+1, 256)
        cell_emb = emb[:, 0]  # CLS pooling

        return self.proj(cell_emb)  # (B, embed_dim)

# =======================================================
# 3. ST Encoder (Spatial coordinates → Spot-level feature)
# =======================================================
class STEncoder(nn.Module):
    def __init__(self, embed_dim=256):

        super().__init__()

        # Convert (x, y) -> embedding
        self.spatial_embed = nn.Sequential(
            nn.Linear(2, embed_dim // 2),  # 2D -> D/2
            nn.LayerNorm(embed_dim // 2),
            nn.ReLU()
        )

        # Positional encoding
        self.pos_enc = nn.Sequential(
            nn.Linear(2, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.Tanh()   # range: [-1, 1]
        )

        # CLS token + spatial token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            batch_first=True,
            dropout=0.1,
            activation='gelu'
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

        # Final projection layer
        self.fc = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.Dropout(0.1)
        )

    def forward(self, x):

        # x: (B, 2) / spatial codinates (x, y)
        B = x.shape[0]

        spatial_emb = self.spatial_embed(x)   # (B, D/2)
        pos_emb = self.pos_enc(x)             # (B, D/2)

        token_emb = torch.cat([spatial_emb, pos_emb], dim=-1)

        # CLS token + spatial token
        cls = self.cls_token.expand(B, 1, -1)                 # (B, 1, D)
        seq = torch.cat([cls, token_emb.unsqueeze(1)], dim=1) # (B, 2, D)

        # Apply transformer
        seq = self.transformer(seq) # (B, 2, D)

        # Apply CLS token
        seq = seq[:, 0, :]            # (B, D)

        return self.fc(seq)


# =======================================================
# 4. Fusion Layer
# =======================================================
class FusionLayer(nn.Module):
    """
    Fusion options (fused_dim=256):

    - 'concat'  : concat([img, sc, st]) -> MLP -> fused (B, fused_dim)
    - 'attn'    : treat [img, sc, st] as 3 tokens -> self-attn -> pool -> fused (B, fused_dim)
    """

    def __init__(
            self,
            embed_dim: int = 256,
            fusion_option: str = 'concat',
            fused_dim: int = 128,
            attn_heads: int = 4,
            dropout: float = 0.2,
            use_l2norm_for_sim: bool = True
    ):

        super().__init__()

        self.embed_dim = embed_dim
        self.fusion_option = fusion_option
        self.fused_dim = fused_dim
        self.dropout = dropout
        self.use_l2norm_for_sim = use_l2norm_for_sim

        # pre-norm for modality feature alignment
        self.pre_norm_img = nn.LayerNorm(embed_dim)
        self.pre_norm_sc = nn.LayerNorm(embed_dim)
        self.pre_norm_st = nn.LayerNorm(embed_dim)

        if fusion_option == 'concat':

            self.concat_fusion = nn.Sequential(
                nn.Linear(embed_dim * 3, fused_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fused_dim * 2, fused_dim)
            )

        elif fusion_option == 'attn':

            self.attn_fusion = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=attn_heads,
                dropout=dropout,
                batch_first=True
            )

            self.norm1 = nn.LayerNorm(embed_dim)

            self.ffn = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 4, embed_dim),
                nn.Dropout(dropout)
            )

            self.norm2 = nn.LayerNorm(embed_dim)

            self.out_proj = nn.Linear(embed_dim, fused_dim)

        else:
            raise ValueError(f"Unknown fusion option={fusion_option}")

    def forward(self, img_feat: torch.Tensor, sc_feat: torch.Tensor, st_feat: torch.Tensor) -> torch.Tensor:

        if img_feat.ndim != 2 or st_feat.ndim != 2 or sc_feat.ndim != 2:
            raise ValueError("Input features must be 2D tensors of shape (B, D)")

        if not (img_feat.shape == sc_feat.shape == st_feat.shape):
            raise ValueError(f"Shape mismatch: {img_feat.shape} vs {sc_feat.shape} vs {st_feat.shape}")

        if not (img_feat.device == sc_feat.device == st_feat.device):
            raise ValueError(f"Device mismatch: {img_feat.device} vs {sc_feat.device} vs {st_feat.device}")

        if not (img_feat.dtype == sc_feat.dtype == st_feat.dtype):
            raise ValueError(f"Dtype mismatch: {img_feat.dtype} vs {sc_feat.dtype} vs {st_feat.dtype}")

        img_feat = self.pre_norm_img(img_feat)
        sc_feat = self.pre_norm_sc(sc_feat)
        st_feat = self.pre_norm_st(st_feat)

        if self.fusion_option == 'concat':

            x = torch.cat([img_feat, sc_feat, st_feat], dim=1)
            return self.concat_fusion(x)

        elif self.fusion_option == 'attn':

            tokens = torch.stack([img_feat, sc_feat, st_feat], dim=1)

            attn_out, _ = self.attn_fusion(tokens, tokens, tokens)
            tokens = self.norm1(tokens + attn_out)

            ffn_out = self.ffn(tokens)
            tokens = self.norm2(tokens + ffn_out)

            pooled = tokens.mean(dim=1)

            return self.out_proj(pooled)

# =======================================================
# 5. Spatial Attention Layer (Spot → Spot)
# =======================================================
class SpatialAttention(nn.Module):
    """
    Spot-level self-attnention restricted to k-nn spatial neighbors
    
    Input:
        spot_embeds : (N, D)
        coords      : (N, 2)
    Output:
        refined     : (N, D)
    """
    def __init__(
        self,
        embed_dim=256,
        num_heads=4,
        k=8,    # neighbors
        dropout=0.1,
        include_self=False
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.k = k
        self.include_self = include_self

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

        self.tau = 2.0  # temperature for attention scaling
        self.alpha = 0.3  # residual scaling factor

        self.attn_dropout = nn.Dropout(dropout)
    
    def _build_knn_mask(self, coords):
        """
        coords: (N, 2)
        return:
            attn_mask: (N, N) boolean mask
                True -> allowed
                False -> masked 
        """
        N = coords.size(0)
        device = coords.device

        # pairwise distance (higher if closer)
        dist = torch.cdist(coords, coords, p=2)  # (N, N)
        dist = dist / (dist.max() + 1e-6)  # normalize to [0, 1]
        dist_bias = -dist
        attn_scores = attn_scores + dist_bias.unsqueeze(0)

        # Find kNN excluding self
        if self.include_self:
            k_eff = min(self.k + 1, N)  # include self
            knn_idx = dist.topk(k_eff, largest=False).indices  # (N, k_eff)
        else:
            # self는 멀리 보내서 제외
            dist = dist + torch.eye(N, device=device) * 1e6
            k_eff = min(self.k, N - 1)
            knn_idx = dist.topk(k_eff, largest=False).indices  # (N, k_eff)

        mask = torch.zeros(N, N, device=device, dtype=torch.bool)
        row_idx = torch.arange(N, device=device).unsqueeze(1).expand_as(knn_idx)
        mask[row_idx, knn_idx] = True

        if self.include_self:
            mask.fill_diagonal_(True)

        return mask

    def forward(self, spot_embeds, coords, return_attn=False):
        """
        spot_embeds: (N, D)
        coords: (N, 2)
        """
        N, D = spot_embeds.shape

        x = self.norm1(spot_embeds)

        q = self.q_proj(x).view(N, self.num_heads, self.head_dim).transpose(0, 1)  # (H, N, Dh)
        k = self.k_proj(x).view(N, self.num_heads, self.head_dim).transpose(0, 1)  # (H, N, Dh)
        v = self.v_proj(x).view(N, self.num_heads, self.head_dim).transpose(0, 1)  # (H, N, Dh)

        # scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (H, N, N)    
        attn_scores = attn_scores / self.tau

        knn_mask = self._build_knn_mask(coords)  # (N, N)
        attn_scores = attn_scores.masked_fill(~knn_mask.unsqueeze(0), float('-inf'))
        
        attn = torch.softmax(attn_scores, dim=-1)  # (H, N, N)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v)  # (H, N, Dh)
        out = out.transpose(0, 1).contiguous().view(N, D)  # (N, D)
        out = self.out_proj(out)

        # residual(scaled)
        x = spot_embeds + self.alpha * out

        # FFN
        x = x + self.ffn(self.norm2(x))

        if return_attn:
            return x, attn
        return x
    
# =======================================================
# 6. MIL Attention Pooling (Spot → WSI)
# =======================================================

class MILAttentionPooling(nn.Module):

    def __init__(self, embed_dim=256, hidden_dim: int = None, dropout: float = 0.0):

        super().__init__()

        if hidden_dim is None:
            hidden_dim = max(64, embed_dim)

        self.attn_V = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh()
        )

        self.attn_U = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Sigmoid()
        )

        self.attn_w = nn.Linear(hidden_dim, 1)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, spot_embeds):

        A = self.attn_w(self.attn_V(spot_embeds) * self.attn_U(spot_embeds))
        weights = F.softmax(A, dim=0)
        wsi_embed = torch.sum(weights * spot_embeds, dim=0)

        return wsi_embed, weights


# =======================================================
# 7. Linear Head 
# =======================================================
class LinearHead(nn.Module):

    def __init__(self, dim: int, use_ln: bool = True):

        super().__init__()

        self.ln = nn.LayerNorm(dim) if use_ln else nn.Identity()
        self.fc = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.fc(self.ln(x))


# -------------------------------------------------------
# 6. Full Multi-Modal Model (Fusion + Classifier)
# -------------------------------------------------------
class MultiModalMILModel(nn.Module):

    def __init__(
            self,
            num_genes: int,
            modality_option: str = 'multi',
            num_classes: int = 2,
            embed_dim: int = 256,
            fusion_option: str = 'concat',
            top_k_genes: int = None,
            img_backbone: str = 'resnet18',
            img_pretrained: bool = True,
            mil_hidden_dim: int = None,
            mil_dropout: float = 0.0,
            fusion_dropout: float = 0.2,
            head_use_ln: bool = True,

            # spatial attn ablation
            use_spatial_attn: bool = True,
            spatial_attn_k: int = 8,
            spatial_attn_heads: int = 4,
            spatial_attn_dropout: float = 0.1,
    ):
 
        super().__init__()

        self.modality_option = modality_option
        self.embed_dim = embed_dim
        self.img_encoder = ImageEncoder(embed_dim=embed_dim, backbone=img_backbone, pretrained=img_pretrained)

        self.sc_encoder = SCEncoder(
            num_genes=num_genes,
            embed_dim=embed_dim,
            top_k_genes=top_k_genes
        )

        self.st_encoder = STEncoder(embed_dim=embed_dim)

        self.img_head = LinearHead(dim=embed_dim, use_ln=head_use_ln)
        self.sc_head = nn.Identity()
        self.st_head = nn.Identity()

        self.freeze_encoders()

        print(f"✓ Model initialized with fusion_option='{fusion_option}'")
        if top_k_genes:
            print(f"✓ Using top_k_genes={top_k_genes} for SCEncoder")

        self.fusion = FusionLayer(
            embed_dim=embed_dim,
            fusion_option=fusion_option,
            fused_dim=embed_dim,
            attn_heads=4,
            dropout=fusion_dropout,
            use_l2norm_for_sim=True,
        )

        self.use_spatial_attn = use_spatial_attn

        if self.use_spatial_attn:
            self.spatial_attn = SpatialAttention(
                embed_dim=embed_dim,
                num_heads=spatial_attn_heads,
                k=spatial_attn_k,
                dropout=spatial_attn_dropout,
                include_self=False,
            )
        else:
            self.spatial_attn = None

        self.mil_pooling = MILAttentionPooling(
            embed_dim=embed_dim,
            hidden_dim=mil_hidden_dim,
            dropout=mil_dropout,
        )

        self.classifier = nn.Linear(embed_dim, num_classes)

    def freeze_encoders(self):

        for param in self.img_encoder.parameters():
            param.requires_grad = False

        self.img_encoder.eval()

    def train(self, mode: bool = True):

        super().train(mode)
        self.img_encoder.eval()

    def forward(
        self,
        img: torch.Tensor,
        expr: torch.Tensor,
        coord: torch.Tensor,
        batch_spots: int = None,
        save_embeddings: bool = False,
    ):
        """
        img   : (N, 3, 224, 224)
        expr  : (N, G)
        coord : (N, 2)

        Returns end-to-end result for one WSI sample.
        Internally uses chunking for encoder/fusion stage if batch_spots is given.
        """

        if img.dim() != 4:
            raise ValueError(f"img must be (N, 3, H, W), got {img.shape}")
        if expr.dim() != 2:
            raise ValueError(f"expr must be (N, G), got {expr.shape}")
        if coord.dim() != 2:
            raise ValueError(f"coord must be (N, 2), got {coord.shape}")

        if not (img.shape[0] == expr.shape[0] == coord.shape[0]):
            raise ValueError(
                f"spot count mismatch: img={img.shape[0]}, expr={expr.shape[0]}, coord={coord.shape[0]}"
            )

        modality_option = self.modality_option
        n_spots = img.shape[0]

        if batch_spots is None or batch_spots <= 0:
            batch_spots = n_spots

        img_chunks = []
        sc_chunks = []
        st_chunks = []
        fused_chunks = []

        for i in range(0, n_spots, batch_spots):
            j = min(i + batch_spots, n_spots)

            img_b = img[i:j]
            expr_b = expr[i:j]
            coord_b = coord[i:j]

            with torch.no_grad():
                img_feat = self.img_encoder(img_b)

            if modality_option == "img":
                img_feat = self.img_head(img_feat)
                img_chunks.append(img_feat)

            elif modality_option == "multi":
                img_feat = self.img_head(img_feat)
                sc_feat = self.sc_head(self.sc_encoder(expr_b))
                st_feat = self.st_head(self.st_encoder(coord_b))

                fused_feat = self.fusion(img_feat, sc_feat, st_feat)

                img_chunks.append(img_feat)
                sc_chunks.append(sc_feat)
                st_chunks.append(st_feat)
                fused_chunks.append(fused_feat)

            else:
                raise ValueError(f"Unsupported modality_option={modality_option}")

        spatial_attn_map = None
        spot_embeds_before_spatial = None
        spot_embeds_after_spatial = None

        if modality_option == "img":
            mil_input = torch.cat(img_chunks, dim=0)

        elif modality_option == "multi":
            img_feat_all = torch.cat(img_chunks, dim=0)
            sc_feat_all = torch.cat(sc_chunks, dim=0)
            st_feat_all = torch.cat(st_chunks, dim=0)
            fused_all = torch.cat(fused_chunks, dim=0)

            spot_embeds_before_spatial = fused_all

            if self.use_spatial_attn:
                mil_input, spatial_attn_map = self.spatial_attn(
                    fused_all, coord, return_attn=True
                )
                spot_embeds_after_spatial = mil_input
            else:
                mil_input = fused_all

        wsi_embed, mil_attn = self.mil_pooling(mil_input)
        logits = self.classifier(wsi_embed.unsqueeze(0)).squeeze(0)

        out = {
            "logits": logits,
            "wsi_embed": wsi_embed,
            "attn_weights": mil_attn,
            "spatial_attn_map": spatial_attn_map,
            "spot_embeds_before_spatial": spot_embeds_before_spatial,
            "spot_embeds_after_spatial": spot_embeds_after_spatial,
        }

        if save_embeddings:
            if modality_option == "img":
                out["img_embed"] = mil_input
            elif modality_option == "multi":
                out["img_embed"] = img_feat_all
                out["sc_embed"] = sc_feat_all
                out["st_embed"] = st_feat_all
                out["spot_fusion_embed"] = fused_all
                if self.use_spatial_attn:
                    out["spatial_embed"] = mil_input
            out["coords"] = coord

        return out