# AlphaFold3 Input Embedding 理解总结与修正

## 问题

所以我来总结一下inputembedding这个模块的输入和输出，输入就是msa得到的残基或者核苷酸的类型，msa profile 和deletion信息，共63维，还有每个残基原子固定坐标，以及小分子原子的rdkit生成的构象坐标，这些残基原子和小分子原子，每个原子的张量表示包括原子类型、电荷、坐标等等信息，然后经过linear变成128维，得到原子数量，128维的张量，这个张量进行自注意力其实就是原子间的交叉注意力，这样就更新了每个原子的张量，这个张量128再经过linear变成384维，然后再按照每个token聚合，比如每个残基为一个token，把该残基的所有原子张量相加，最终得到就是N，384维的张量，然后再加上之前的63维，如果是小分子那就这63维置为0或者特殊标记，反正最终得到的就是N，447维的张量，input embedding模块输出的inputs就是N，447，pair就是N，N，128，single就是N，384

## 回答

你的总结整体是对的，只有几个细节需要修正。

### 正确的整体流程

```text
MSA / token / deletion 信息
    → 约 63 维 token-level 特征

蛋白质残基的参考原子位置
+
配体原子的 conformer 坐标
+
元素、电荷、原子类型等
    → atom-level 特征

atom-level attention
    → 更新每个原子的表示

原子表示投影到 384 维
    → 按 token 聚合

得到 [N, 384] 的 atom-derived token embedding
    → 与 [N, 63] 的序列特征拼接

得到 target_feat：
    [N, 447]

再由它生成：

single：
    [N, 384]

pair：
    [N, N, 128]
```

### 需要修正的几点

1. **原子特征不是简单把所有信息直接拼起来后线性变成 128 维。**

原子类型、电荷等形成原子属性；坐标主要用于构造原子之间的相对位移和距离，然后这些信息一起进入 atom cross-attention。

2. **atom attention 的输出概念上是每个原子的更新表示。**

如果有 12 个有效原子，可以理解为：

```text
12 个原子 → 12 个更新后的原子表示
```

实际张量还会包含 padding 槽位，例如：

```text
[N, A, 128]
```

3. **聚合通常不是简单求和，而是 mask mean。**

同一个蛋白质残基 token 内的多个原子表示会被聚合；配体原子 token 通常只有一个有效原子。

```text
[N, A, 128]
    → Linear
[N, A, 384]
    → 按 token 对有效原子做 mask mean
[N, 384]
```

4. **小分子的 63 维特征不一定全部是 0。**

更准确地说：

- token 类型使用 unknown 或非聚合物标记；
- MSA profile 通常没有有效信息，会是零值或 masked；
- deletion mean 通常为零值或 masked；
- 配体真正的信息来自原子属性、化学键和 conformer。

5. **`[N,447]`、`[N,384]`、`[N,N,128]` 是不同阶段的表示。**

在当前 AF3 典型配置下：

```text
target_feat / s_inputs：
    [N, 447]

初始 single：
    [N, 384]

初始 pair：
    [N, N, 128]
```

之后 Template Module、MSA Module、Pairformer 和 recycling 会继续更新 single 和 pair，但形状通常保持：

```text
single：[N,384]
pair：  [N,N,128]
```

所以你最后的核心总结可以写成：

> Input embedding 把序列、MSA、原子属性和参考几何信息统一编码成 token 表示；atom cross-attention 先更新原子级特征，再把同一 token 内的原子聚合成 384 维 token 特征；这些 token 特征用于初始化 `[N,384]` 的 single 和 `[N,N,128]` 的 pair。
