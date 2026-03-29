import torch

# 加载词嵌入文件
word_embedding_path = './pretrained_weights/word_embedding_38class.pth'
word_embedding = torch.load(word_embedding_path)

# 查看词嵌入的大小
print(f"词嵌入形状: {word_embedding.shape}")
print(f"类别数量: {word_embedding.shape[0]}")
print(f"嵌入维度: {word_embedding.shape[1]}")
