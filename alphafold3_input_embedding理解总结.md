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

得到内部 target_feat：
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

5. **必须区分内部特征和图中标记的三个输出。**

在当前 AF3 典型配置下，`[N,447]` 是代码中的内部 `target_feat`，还不是图中标记的 `inputs` 输出。图中三个表示应明确写成：

```text
inputs / s_i^inputs：
    [N, 384]

single / s_i^init：
    [N, 384]

pair / z_ij^init：
    [N, N, 128]
```

它们之间的关系是：

```text
原始特征和 atom embedding
    → 内部 target_feat：[N,447]
    → InputFeatureEmbedder
    → inputs：[N,384]

inputs：[N,384]
    ├── LinearNoBias → single：[N,384]
    └── 左右 pair 投影 → pair：[N,N,128]
```

`inputs` 和 `single` 的通道数相同，但不是同一个张量：`inputs` 是输入编码器的输出，`single` 是经过独立线性变换后交给主干网络的初始 single 状态。这个线性层的作用是重新混合特征通道和适配后续网络，不是为了改变维度或专门为了反向传播。

之后 Template Module、MSA Module、Pairformer 和 recycling 会继续更新 single 和 pair，但形状通常保持：

```text
single：[N,384]
pair：  [N,N,128]
```

所以更准确的核心总结是：

> Input embedding 内部先把序列、MSA、原子属性和参考几何信息编码成内部的 `[N,447]` 特征，再投影为图中明确标记的 `inputs [N,384]`。`inputs` 经过独立变换产生 `single [N,384]`，同时通过左右投影构造 `pair [N,N,128]`。因此，图中的三个输出应当明确区分为 `inputs`、`single` 和 `pair`，而 `[N,447]` 只是内部中间特征。

## MSA 分支与 token 分支的区别

MSA 的原始特征和 token-level 的输入特征是两条并行分支，不能把它们的维度直接相加。

### MSA Module 分支

每条 MSA 序列、每个位置的特征大致由以下部分组成：

```text
31 维 residue/nucleotide 类型
+ 1 维 has_deletion
+ 1 维 deletion_value
= 33 维
```

因此原始 MSA 特征为：

```text
[M, N, 33]
```

经过线性层后得到：

```text
[M, N, 33]
    ↓ Linear
[M, N, 64]
```

这个 `[M,N,64]` 表示进入 AF3 的 MSA Module/MSA stack，用于 MSA row attention、MSA column attention 等操作，并通过 MSA 相关模块把进化信息传递给 pair 表示。

它不会直接拼接到 `[N,447]` 中。

### Token-level 输入分支

另一条分支使用 MSA 的统计摘要，而不是保存每一条 MSA 序列的完整表示：

```text
token type one-hot：[N,31]
MSA profile：      [N,31]
deletion mean：    [N,1]
```

这部分是：

```text
31 + 31 + 1 = 63 维
```

再加上 atom/conformer encoder 产生的：

```text
[N,384]
```

形成内部的：

```text
[N,63] + [N,384]
    ↓
[N,447]
```

然后经过 InputFeatureEmbedder 得到图中标记的：

```text
inputs：[N,384]
```

再由 `inputs` 产生：

```text
single：[N,384]
pair：  [N,N,128]
```

因此，维度关系不是 `33 + 64 + 384`，而是两条并行路径：

```text
MSA 原始特征 [M,N,33]
    ↓ Linear
MSA 表示 [M,N,64]
    ↓
MSA Module
    ↓
影响 pair 表示
```

```text
token type + profile + deletion mean + atom embedding
    ↓
内部 target_feat [N,447]
    ↓ InputFeatureEmbedder
inputs [N,384]
    ├── LinearNoBias → single [N,384]
    └── 左右 pair 投影 → pair [N,N,128]
```

所以可以记住：

- `[M,N,64]`：给 MSA Module 使用；
- `[N,447]`：token-level 的内部中间输入特征；
- `[N,384]`：图中 Input Embedder 输出的 `inputs`，也是 single 初始化和 pair 初始化的来源；
- `[N,N,128]`：token 对之间的 pair 表示。
