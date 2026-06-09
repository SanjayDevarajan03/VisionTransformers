# Vision Transformer (ViT) — From Scratch Implementation

A PyTorch implementation of the Vision Transformer (ViT) architecture from the paper [*An Image is Worth 16x16 Words*](https://arxiv.org/abs/2010.11929), trained from scratch on CIFAR-10.

## Result

**63% accuracy on CIFAR-10** trained from scratch with a small ViT (128 model dim, 6 blocks, 4 heads).

## Architecture

- Patch embedding via Conv2d (4×4 patches → 64 patches per image)
- Learnable CLS token and positional embeddings
- 6 Transformer blocks with pre-norm (LayerNorm before attention)
- Batched multi-head self-attention with scaled dot-product
- MLP with 4× expansion and GELU activation
- Linear classifier on CLS token output

## Project Structure

```
VIT/
├── notebook/
│   └── VisionTransformers.ipynb   # original development notebook
├── model.py                       # VisionTransformer, TransformerBlock, MultiHeadSelfAttention
├── train.py                       # train(), validate(), overfit_test()
├── data.py                        # CIFAR-10 dataloaders and transforms
├── config.py                      # hyperparameters
├── main.py                        # entry point
└── requirements.txt
```

## Quickstart

```bash
git clone https://github.com/yourusername/vit.git
cd vit
pip install -r requirements.txt
python main.py
```

## Configuration

All hyperparameters are in `config.py`:

| Parameter | Value |
|---|---|
| Patch size | 4 |
| Model dim | 128 |
| Num heads | 4 |
| Num blocks | 6 |
| Learning rate | 3e-4 |
| Epochs | 50 |
| Batch size | 64 |

## Training Details

- Optimizer: Adam (β1=0.9, β2=0.999)
- LR schedule: linear warmup (10% of steps) then cosine decay to 0
- Data augmentation: random horizontal flip, random crop with padding 4
- Gradient clipping: max norm 1.0

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- torchtyping
- numpy
