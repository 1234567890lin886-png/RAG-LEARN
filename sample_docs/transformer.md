# Transformer 简介

Transformer 是一种用于处理序列数据的神经网络结构。它最重要的创新是 self-attention，也就是模型可以在处理一个词时，同时关注句子里的其他词。

传统循环神经网络会按顺序读文本，长距离依赖比较难处理。Transformer 可以并行处理 token，并通过注意力机制建模词与词之间的关系。

在自然语言处理中，Transformer 可以用于文本分类、语义检索、摘要、翻译、问答和聊天机器人。BERT 更偏向理解文本，GPT 更偏向生成文本。

在检索系统里，我们常用 Transformer encoder 把文本转换成 embedding。语义相近的文本通常会得到距离更近的向量。
