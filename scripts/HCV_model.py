# package
import torch
import torch.nn as nn
import timm

# =====================================================
# DenseNet + ViT Model
# =====================================================

class DenseNetViT(nn.Module):
    def __init__(self, num_classes=14, freeze_backbone=False, freeze_vit=False, use_custom_proj=False, train_last_vit_layers=None, custom_proj_layers=512):
        super().__init__()
        self.use_custom_proj = use_custom_proj

        # DenseNet backbone
        self.backbone = timm.create_model("densenet121", pretrained=True, features_only=True)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # ViT
        # We replace the ViT patch embedding with DenseNet features
        # The positional embeddings are sliced to match the number of tokens
        # 1 CLS token + 49 patch tokens from DenseNet = 50 tokens.
        self.vit = timm.create_model("vit_base_patch16_224", pretrained=True)
        self.vit.head = nn.Identity()   # remove original classification head
        self.vit.patch_embed = nn.Identity()
        self.vit.pos_embed = nn.Parameter(self.vit.pos_embed[:, :50, :])

        if freeze_vit:
            # Optionally: Train only the last N ViT transformer blocks
            if train_last_vit_layers is not None and train_last_vit_layers > 0:
                total_blocks = len(self.vit.blocks)
                train_from = total_blocks - train_last_vit_layers

                # print(f"Training ViT blocks [{train_from} ... {total_blocks-1}]")

                # Freeze all blocks first
                for i, block in enumerate(self.vit.blocks):
                    for p in block.parameters():
                        p.requires_grad = False

                # Unfreeze last N blocks
                for i in range(train_from, total_blocks):
                    for p in self.vit.blocks[i].parameters():
                        p.requires_grad = True
            else:  # Otherwise freeze all ViT blocks
                for p in self.vit.parameters():
                    p.requires_grad = False


        # Optional custom projection layer: DenseNet -> Custom Projection -> ViT embedding
        if self.use_custom_proj:
            self.custom_proj = nn.Sequential(
                nn.Conv2d(1024, custom_proj_layers, kernel_size=1),
                nn.LeakyReLU(),
                nn.Conv2d(custom_proj_layers, self.vit.embed_dim, kernel_size=1)
            )
        else:
            self.direct_projection = nn.Conv2d(1024, self.vit.embed_dim, kernel_size=1)

        # New classification head
        self.classifier = nn.Linear(self.vit.embed_dim, num_classes)

    def forward(self, images):
        batch_size = images.size(0)
        densenet_features = self.backbone(images)[-1]

        if self.use_custom_proj:
            projected_features = self.custom_proj(densenet_features)
        else:
            projected_features = self.direct_projection(densenet_features)

        patch_embeddings = projected_features.flatten(2).transpose(1, 2)
        cls_tokens_expanded = self.vit.cls_token.expand(batch_size, -1, -1)
        vit_input = torch.cat((cls_tokens_expanded, patch_embeddings), dim=1)
        vit_input = vit_input + self.vit.pos_embed[:, :vit_input.size(1), :]
        vit_input = self.vit.pos_drop(vit_input)

        vit_output = self.vit.blocks(vit_input)
        vit_output = self.vit.norm(vit_output)

        logits = self.classifier(vit_output[:, 0])
        return logits