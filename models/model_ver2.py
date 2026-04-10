import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from models.performer_pytorch import Performer

"""
Model architecture:
1. Image Encoder: ResNet-18 → (N_spots, embed_dim)
2. ST Encoder: scBERT-style Performer with spatial token → (N_spots, embed_dim)
3. Fusion Layer: concat/attn → (B, fused_dim)
4. MIL Attention Pooling: (N_spots, fused_dim) → (fused_dim)
5. Classifier: Linear(fused_dim → num_classes)
"""
# =======================================================
# 1. Image Encoder (Patch → Spot-level visual feature)
# =======================================================
class ImageEncoder(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.backbone = models.resnet18(weights="IMAGENET1K_V1")
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, embed_dim)

    def forward(self, x):
        """
        x: (N_spots, 3, 224, 224)
        return: (N_spots, embed_dim)
        """
        return self.backbone(x)

# =======================================================
# 2. Spatial ST Encoder (HVG-only, scBERT-style)
# =======================================================
class SpatialSTEncoder(nn.Module):
    """
    scBERT-style encoder with explicit spatial token

    Input:
      - expr   : (N_spots, K)   [already HVG-filtered]
      - coords : (N_spots, 2)   [normalized]

    Output:
      - (N_spots, embed_dim) spot-level ST embedding
    """

    def __init__(
        self,
        num_genes,        # K = number of HVGs (e.g., 2000)
        embed_dim=256,
        num_layers=2,
        num_heads=4,
        top_k_genes=None,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_genes = num_genes
        self.top_k_genes = top_k_genes

        # Gene identity embedding (HVG-only)
        self.gene_embedding = nn.Embedding(num_genes, embed_dim)

        # Gene positional embedding (gene order)
        self.gene_pos_embedding = nn.Embedding(num_genes, embed_dim)

        # Expression value embedding
        self.value_embedding = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )

        # Spatial token embedding
        self.spatial_embedding = nn.Sequential(
            nn.Linear(2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )

        # Performer (efficient transformer)
        self.transformer = Performer(
            dim=embed_dim,
            depth=num_layers,
            heads=num_heads,
            dim_head=embed_dim // num_heads,
            causal=False,
            ff_mult=4,
            attn_dropout=0.1,
            ff_dropout=0.1,
        )

        # Spatial-query pooling
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, expr, coords, return_gene_attn=False):
        """
        expr   : (N, K)
        coords : (N, 2)

        if return_gene_attn:
          returns:
            pooled: (N, D)
            gene_attn: (N, G_used)          # G_used = top_k_genes or K
            gene_indices: (N, G_used) (long) # global gene ids
        else:
          returns:
            pooled: (N, D)
        """
        N, K = expr.shape
        device = expr.device

        # Top-K gene selection
        if self.top_k_genes and self.top_k_genes < K:
            topk_values, topk_indices = torch.topk(expr, k=self.top_k_genes, dim=1)
            gene_indices = topk_indices.long()  # global gene ids

            gene_embed = self.gene_embedding(gene_indices)  # (N, top_k, D)
            gene_pos = self.gene_pos_embedding(gene_indices)
            value_emb = self.value_embedding(topk_values.unsqueeze(-1))
            gene_tokens = gene_embed + gene_pos + value_emb
        else:
            gene_ids = torch.arange(K, device=device).unsqueeze(0).expand(N, -1).long()
            gene_embed = self.gene_embedding(gene_ids)
            gene_pos = self.gene_pos_embedding(gene_ids)
            value_emb = self.value_embedding(expr.unsqueeze(-1))
            gene_tokens = gene_embed + gene_pos + value_emb

        spatial_token = self.spatial_embedding(coords).unsqueeze(1)
        tokens = torch.cat([spatial_token, gene_tokens], dim=1)

        tokens = self.transformer(tokens)

        spatial_out = tokens[:, :1]
        gene_out    = tokens[:, 1:]

        q = self.q_proj(spatial_out)
        k = self.k_proj(gene_out)
        v = self.v_proj(gene_out)

        attn = torch.softmax(
            torch.matmul(q, k.transpose(-2, -1)) / (self.embed_dim ** 0.5),
            dim=-1
        )

        pooled = torch.matmul(attn, v).squeeze(1)
        pooled = self.out_proj(pooled)

        if return_gene_attn:
            gene_attn = attn.squeeze(1)
            return pooled, gene_attn, gene_indices
        else:
            return pooled

# =======================================================
# 3. Spot Fusion Module (4 options: concat, attn, sim, gate)
# =======================================================
class FusionLayer(nn.Module):
    """
    Fusion options:
    - 'concat': Simple concatenation + MLP
    - 'attn': Cross-attention between img and st
    """
    def __init__(
        self,
        embed_dim=256,
        fusion_option='concat',
        attn_heads=4,
        dropout=0.2,
        use_l2norm_for_sim=True
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.fusion_option = fusion_option
        self.dropout = dropout
        self.use_l2norm_for_sim = use_l2norm_for_sim
        
        # Pre-normalization
        self.pre_norm_img = nn.LayerNorm(embed_dim)
        self.pre_norm_st = nn.LayerNorm(embed_dim)

        if fusion_option == 'concat':
            self.fuse = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        elif fusion_option == 'attn':
            self.attn = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=attn_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.norm1 = nn.LayerNorm(embed_dim)
            self.ffn = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 4, embed_dim),
                nn.Dropout(dropout),
            )
            self.norm2 = nn.LayerNorm(embed_dim)
            self.out_proj = nn.Linear(embed_dim, embed_dim)
        else:
            raise ValueError(f"Unknown fusion_option: {fusion_option}")

    def forward(self, img_feat, st_feat):
        """
        img_feat: (N, D)
        st_feat: (N, D)
        return: (N, D)
        """
        # Pre-norm
        img_feat = self.pre_norm_img(img_feat)
        st_feat = self.pre_norm_st(st_feat)

        if self.fusion_option == 'concat':
            # Simple concatenation
            x = torch.cat([img_feat, st_feat], dim=-1)  # (N, 2D)
            return self.fuse(x)  # (N, D)

        elif self.fusion_option == 'attn':
            # Cross-attention: [img, st] as 2 tokens
            tokens = torch.stack([img_feat, st_feat], dim=1)  # (N, 2, D)
            
            # Self-attention
            attn_out, _ = self.attn(tokens, tokens, tokens)  # (N, 2, D)
            tokens = self.norm1(tokens + attn_out)
            
            # FFN
            ffn_out = self.ffn(tokens)  # (N, 2, D)
            tokens = self.norm2(tokens + ffn_out)
            
            # Pool (average)
            pooled = tokens.mean(dim=1)  # (N, D)
            return self.out_proj(pooled)

# =======================================================
# 4. Spatial Attention Layer (Spot → Spot)
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

        # pairwise distance
        dist = torch.cdist(coords, coords, p=2)  # (N, N)

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

        # residual
        x = spot_embeds + out

        # FFN
        x = x + self.ffn(self.norm2(x))

        if return_attn:
            return x, attn
        return x

# =======================================================
# 5. MIL Attention Pooling (Spot → WSI)
# =======================================================
class MILAttentionPooling(nn.Module):
    def __init__(self, embed_dim=256, hidden_dim=128):
        super().__init__()
        self.attn_V = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh()
        )
        self.attn_U = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.attn_w = nn.Linear(hidden_dim, 1)

    def forward(self, spot_embeds):
        """
        spot_embeds: (N_spots, D)
        """
        A = self.attn_w(self.attn_V(spot_embeds) * self.attn_U(spot_embeds))
        weights = F.softmax(A, dim=0)
        wsi_embed = torch.sum(weights * spot_embeds, dim=0)
        return wsi_embed, weights

# =======================================================
# 6. Linear Head
# =======================================================
class LinearHead(nn.Module):
    def __init__(self, dim: int, use_ln: bool=True):
        super().__init__()
        self.ln = nn.LayerNorm(dim) if use_ln else nn.Identity()
        self.fc = nn.Linear(dim, dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.ln(x))
      
# =======================================================
# 6. Full Multi-Modal MIL Model
# =======================================================
class MultiModalMILModel(nn.Module):
    def __init__(
        self,
        num_genes=2000,
        num_classes=2,
        embed_dim=256,
        fusion_option='concat',
        top_k_genes=None,
        dropout=0.3,
        freeze_image_encoder=True,
        mil_hidden_dim=128,
        mil_dropout=0.0,
        fusion_dropout=0.2,
        head_use_ln=True,

        # ablation용
        use_image =True,
        use_st=True,

        # spatial attn
        use_spatial_attn=False,
        spatial_attn_k=8,
        spatial_attn_heads=4,
        spatial_attn_dropout=0.1,
    ):
        super().__init__()

        assert use_image or use_st, "At least one modality must be used!!"
        
        self.use_image = use_image
        self.use_st = use_st
        self.fusion_option = fusion_option
        self.freeze_image_encoder = freeze_image_encoder

        # Ablation: conditional encoder
        if self.use_image:
            self.img_encoder = ImageEncoder(embed_dim)
            self.img_head = LinearHead(dim=embed_dim, use_ln=head_use_ln)
        else:
            self.img_encoder = None
            self.img_head = None

        if self.use_st:
            self.st_encoder = SpatialSTEncoder(
                num_genes=num_genes,
                embed_dim=embed_dim,
                top_k_genes=top_k_genes,
            )
        else:
            self.st_encoder = None
        self.st_head = nn.Identity()

        # Freeze
        if self.use_image and freeze_image_encoder:
            self.freeze_encoders()
            
        if self.use_image and self.use_st:
            self.fusion = FusionLayer(
                embed_dim=embed_dim,
                fusion_option=fusion_option,
                dropout=fusion_dropout,
            )
        else:
            self.fusion = None

        self.use_spatial_attn = use_spatial_attn
        if use_spatial_attn:
            self.spatial_attn = SpatialAttention(
                embed_dim=embed_dim,
                num_heads=spatial_attn_heads,
                k=spatial_attn_k,
                dropout=spatial_attn_dropout,
                include_self=False
            )
        else:
            self.spatial_attn = None

        # MIL Pooling
        self.mil_pooling = MILAttentionPooling(
            embed_dim=embed_dim,
            hidden_dim=mil_hidden_dim,
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
        print(f"Model 1 initialized with fusion_option='{fusion_option}'")
        if freeze_image_encoder:
            print(f"Image Encoder frozen (only img_head trainable)")

    def freeze_encoders(self):
        """ResNet backbone만 freeze"""
        if self.img_encoder is None:    # Ablation
            return
        for param in self.img_encoder.parameters():
            param.requires_grad = False
        self.img_encoder.eval()
    
    def train(self, mode: bool=True):
        """Keep image encoder as eval during training"""
        super().train(mode)
        if self.use_image and self.freeze_image_encoder:
            self.img_encoder.eval()

    def forward(self, images, expr, coords, return_gene_attn=True, return_spot_embeds=True):
        """
        images: (N_spots, 3, 224, 224)
        expr  : (N_spots, K)
        coords: (N_spots, 2)
        
        Returns:
          logits: (num_classes,)
          attn: (N_spots, 1)
        """
        spot_embeds = None

        gene_attn = None
        gene_indices = None
        
        # Ablation: conditional encoding
        if self.use_image:  # img branch
            if self.freeze_image_encoder:
                with torch.no_grad():
                    img_feat = self.img_encoder(images)
            else:
                img_feat = self.img_encoder(images)
            img_feat = self.img_head(img_feat)  # FC layer (trainable)

        if self.use_st:     # st branch
            if return_gene_attn:
                st_feat, gene_attn, gene_indices = self.st_encoder(expr, coords, return_gene_attn=True)
            else:
                st_feat = self.st_encoder(expr, coords, return_gene_attn=False)
                gene_attn, gene_indices = None, None
        
        # Ablation: process spot embedding per modality
        if self.use_image and self.use_st:  # Both modalities: Fusion
            spot_embeds = self.fusion(img_feat, st_feat)
        elif self.use_image:    # Image only
            spot_embeds = img_feat
        elif self.use_st:       # ST only
            spot_embeds = st_feat

        spot_embeds_before_spatial = spot_embeds.clone()

        # Spatial spot-to-spot attention
        spatial_attn_map = None
        if self.use_spatial_attn:
            spot_embeds, spatial_attn_map = self.spatial_attn(
                spot_embeds, coords, return_attn=True
            )

        # MIL Pooling
        wsi_embed, mil_attn = self.mil_pooling(spot_embeds)
        # mil_attn = mil_attn.squeeze(-1)  # (N_spots,)

        # Classification
        logits = self.classifier(wsi_embed)
        
        out = {
            "logits": logits,
            "mil_attn": mil_attn,
            "gene_attn": gene_attn,
            "gene_indices": gene_indices,
            "spatial_attn_map": spatial_attn_map,
            "img_embed": img_feat if self.use_image else None,
            "st_embed": st_feat if self.use_st else None,
            "spot_embeds_before_spatial": spot_embeds_before_spatial,
            "spot_embeds_after_spatial": spot_embeds,
            "wsi_embed": wsi_embed,
        }
        if return_spot_embeds:
            out["spot_embeds"] = spot_embeds

        return out