import torch

# 检查默认词嵌入文件的路径
word_embedding_path = './pretrained_weights/txt_encoding.pth'
print(f"默认词嵌入文件路径: {word_embedding_path}")

# 加载词嵌入文件
word_embedding = torch.load(word_embedding_path)

# 查看词嵌入的详细信息
print(f"词嵌入类型: {type(word_embedding)}")
print(f"词嵌入形状: {word_embedding.shape}")
print(f"类别数量: {word_embedding.shape[0]}")
print(f"嵌入维度: {word_embedding.shape[1]}")
