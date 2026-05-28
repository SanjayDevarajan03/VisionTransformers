import torch
import torch.nn as nn
from torchtyping import TensorType


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = model_dim // num_heads
        self.q = nn.Linear(model_dim, model_dim, bias=False)
        self.k = nn.Linear(model_dim, model_dim, bias=False)
        self.v = nn.Linear(model_dim, model_dim, bias=False)
        self.dropout = nn.Dropout(p=0.1)
        self.out_projection = nn.Linear(model_dim, model_dim)

    def forward(self, embedded: TensorType[float]):
        B, T, C = embedded.shape
        q = self.q(embedded).view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        k = self.k(embedded).view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        v = self.v(embedded).view(B, T, self.num_heads, self.head_size).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / (self.head_size ** 0.5)
        scores = nn.functional.softmax(scores, dim=-1)
        output = self.dropout(scores) @ v
        output = output.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.head_size)
        return self.out_projection(output)


class NeuralNetwork(nn.Module):
    def __init__(self, model_dim: int):
        super().__init__()
        self.layer1 = nn.Linear(model_dim, model_dim * 4)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(p=0.2)
        self.layer2 = nn.Linear(model_dim * 4, model_dim)

    def forward(self, embedded: TensorType[float]):
        return self.layer2(self.dropout(self.gelu(self.layer1(embedded))))


class TransformerBlock(nn.Module):
    def __init__(self, model_dim: int, num_heads: int):
        super().__init__()
        self.attention = MultiHeadSelfAttention(model_dim, num_heads)
        self.mlp = NeuralNetwork(model_dim)
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)

    def forward(self, embedded: TensorType[float]):
        embedded = embedded + self.attention(self.norm1(embedded))
        embedded = embedded + self.mlp(self.norm2(embedded))
        return embedded


class VisionTransformer(nn.Module):
    def __init__(self, patch_size, num_heads, num_blocks, img_size, model_dim, num_classes, **kwargs):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_embeddings = nn.Conv2d(3, model_dim, kernel_size=patch_size, stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.num_patches + 1, model_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, model_dim))
        self.embedding_dropout = nn.Dropout(p=0.1)
        self.transformer_blocks = nn.ModuleList([TransformerBlock(model_dim, num_heads) for _ in range(num_blocks)])
        self.final_norm = nn.LayerNorm(model_dim)
        self.classifier = nn.Linear(model_dim, num_classes)

    def forward(self, images: TensorType[float]):
        x = self.patch_embeddings(images).flatten(2).transpose(1, 2)
        cls_token = self.cls_token.expand(images.shape[0], -1, -1)
        x = self.embedding_dropout(torch.cat((cls_token, x), dim=1) + self.position_embeddings)
        for block in self.transformer_blocks:
            x = block(x)
        return self.classifier(self.final_norm(x)[:, 0])