# 常见知识

## Transformer

Transformer 是一种神经网络结构，擅长处理序列数据，尤其是文本。它的核心组件是 self-attention。

你可以先把它理解成：

```text
模型读一句话时，会动态判断每个词应该重点关注哪些其他词。
```

## Attention

Attention 的直觉是“分配注意力”。一句话里不是每个词都同样重要，模型会学习哪些词之间关系更强。

例如“苹果发布了新手机，它很受欢迎”，这里“它”大概率指“新手机”，attention 可以帮助模型建立这种联系。

## Embedding

Embedding 是文本的数字表示。模型不能直接理解文字，所以要把文字转换成向量。

语义相近的文本，向量距离通常也更近。

## RAG

RAG 是 Retrieval-Augmented Generation，意思是“检索增强生成”。

普通生成模型只靠自身参数回答，容易记错或胡编。RAG 会先检索资料，再基于资料回答。

## Chunk

Chunk 是文档切出来的小片段。RAG 通常不是按整篇文档检索，而是按片段检索。

## Cosine Similarity

余弦相似度用来衡量两个向量方向是否接近。方向越接近，语义越可能相近。

## Encoder 和 Decoder

Encoder 更适合理解文本，比如分类、检索、向量化。

Decoder 更适合生成文本，比如聊天、续写、总结。

本项目主要用 Encoder 风格的模型做向量化。
