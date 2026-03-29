import torch

# 加载词嵌入文件
embedding_path = './pretrained_weights/word_embedding_38class.pth'
embedding = torch.load(embedding_path)

print(f"词嵌入形状: {embedding.shape}")
print(f"词嵌入大小: {embedding.shape[0]}")
