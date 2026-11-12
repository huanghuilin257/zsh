# AlphaFold 3 完整流程理解与扩散/流匹配对照

本文按照 AlphaFold 3 的整体流程，说明输入信息如何经过搜索、特征化、MSA Module、Template Module、Pairformer 和 diffusion，最后生成蛋白质、核酸和配体的三维原子坐标。文中也单独区分 diffusion 网络、训练时的随机加噪和推理时的多步采样，并解释 AF3 为什么看起来接近 Flow Matching、但不能直接等同于 Flow Matching。

阅读时最重要的边界是：

```text
MSA / template / conformer：提供条件信息
Input embedding / MSA Module / Pairformer：形成 single、pair 条件
DiffusionHead：给定当前带噪坐标，做一次去噪预测
Sampler：决定 sigma schedule，并反复调用 DiffusionHead
```

## 1. AF3 要解决什么问题

AF2 主要围绕蛋白质残基建模，核心表示是：

```text
MSA 表示
Pair 表示
残基级结构表示
```

AF3 的目标更一般，需要同时预测：

- 蛋白质；
- DNA/RNA；
- 小分子配体；
- 离子和水；
- 蛋白质-配体、蛋白质-核酸等复合物。

因此 AF3 将输入统一表示为 token，并使用原子级 diffusion 直接生成最终复合物坐标。

整体流程是：

```text
输入序列、配体和共价键
        ↓
MSA search / Template search / Conformer generation
        ↓
输入特征和 token 化
        ↓
Input Feature Embedding
        ↓
MSA Module + Template Module
        ↓
Pairformer
        ↓
Recycling
        ↓
Diffusion Module
        ↓
最终原子坐标
```

---

## 2. 三类外部输入信息

### 2.1 MSA / genetic search

MSA search 在数据库中寻找目标蛋白质或核酸的同源序列。

假设有 (M) 条 MSA 序列、(N) 个 token，则原始 MSA 可以看成：

```text
[M, N]
```

每个位置最初是离散的氨基酸、核苷酸或 gap 编号。

模型将每个 MSA 单元格特征化为：

```text
31 维 residue/nucleotide 类型
1 维 has_deletion
1 维 deletion_value
```

因此：

```text
MSA 原始特征：[M, N, 33]
```

经过线性层：

```text
[M, N, 33]
    ↓ Linear
[M, N, 64]
```

这个 `[M,N,64]` 是完整的 MSA 表示，供 MSA Module 使用。

此外，MSA 还会统计每个位置的摘要：

```text
MSA profile：[N,31]
deletion mean：[N,1]
```

MSA profile 表示一个位置在所有同源序列中出现各种残基的概率；deletion mean 表示该位置附近在同源序列中平均出现的 deletion/insertion 长度。

要区分两种 MSA 信息：

```text
[M,N,64]：保留每条同源序列的细节，进入 MSA Module
[N,31] + [N,1]：对 MSA 的统计摘要，进入 token-level 输入
```

MSA profile 只是每个位置的边缘统计，完整 MSA 还保留不同位置之间的协同变化关系。

---

### 2.2 Template search

Template search 在已知结构数据库中寻找与目标蛋白质或核酸相似的结构。

模板信息包括：

- 模板残基类型；
- 模板与目标序列的对齐关系；
- 模板中每个残基的原子坐标；
- 原子 mask；
- 模板结构的有效性信息。

假设有 (T) 个模板，每个模板覆盖 (N) 个 token，每个残基最多保存 24 个原子，则模板原子坐标大致是：

```text
template_atom_positions：[T, N, 24, 3]
template_atom_mask：    [T, N, 24]
template_aatype：       [T, N]
```

Template 主要提供已知结构中的：

- 残基对距离；
- 残基对方向；
- 模板残基类型；
- 模板与目标 token 的对应关系。

Template 不是最终结构，只是一个结构先验。

---

### 2.3 Conformer generation

Conformer generation 主要针对配体等小分子。

它根据配体的：

- 原子类型；
- 化学键；
- 电荷；
- 分子图拓扑；

生成配体原子的一个或多个合理参考构象。

例如配体包含原子 (L_1,L_2)，conformer 会提供：

```text
L1 → (x1, y1, z1)
L2 → (x2, y2, z2)
```

它不是最终坐标，只是小分子的参考几何。

需要区分：

```text
Template search：从数据库寻找相似的大分子结构
Conformer generation：根据小分子化学结构生成参考三维构象
```

---

## 3. Token 化

AF3 的 token 规则通常是：

```text
一个标准蛋白质残基 = 一个 token
一个 DNA/RNA 核苷酸 = 一个 token
一个配体原子 = 一个 token
```

例如：

```text
3 个蛋白质残基 + 2 个配体原子
    = 5 个 token
```

这里的 token 数是：

```text
N = 5
```

但原子布局是另一件事。蛋白质残基 token 可能包含多个原子，而配体原子 token 通常只有一个真实原子。

为了统一张量形状，每个 token 预留最多 (A=24) 个原子槽位：

```text
reference positions：[N, A, 3]
atom mask：         [N, A]
```

所以 `[N,A,3]` 中：

- (N) 是 token 数；
- (A) 是每个 token 的原子槽位数；
- 3 是 x、y、z 坐标。

对于蛋白质残基，一个 token 可能占用多个原子槽位；对于配体原子 token，通常只有第一个槽位有效，其余是 padding。

---

## 4. Input Feature Embedding

图中的 InputFeatureEmbedder 是一个抽象模块。在代码中，它由 token 特征构造、atom cross-attention 等多个部分共同实现。

### 4.1 Token-level 的 63 维输入

每个 token 的序列/MSA 摘要大致是：

```text
token type one-hot：[N,31]
MSA profile：      [N,31]
deletion mean：    [N,1]
```

合起来：

```text
[N, 63]
```

对于配体原子 token：

- token 类型使用 unknown 或非聚合物标记；
- MSA profile 通常没有有效信息，是零值或 masked；
- deletion mean 通常是 0 或 masked；
- 配体的主要信息由原子属性、化学键和 conformer 提供。

---

### 4.2 Atom-level 输入

原子级输入包括：

```text
参考原子坐标：[N,A,3]
原子类型
元素
电荷
原子名称
atom mask
reference-space 信息
```

注意，conformer generation 只负责生成配体构象；统一的 `[N,A,3]` 参考布局还会包含蛋白质残基的标准参考原子位置。

原子级模块首先让原子之间交换局部几何信息。概念上：

```text
原子坐标 + 原子属性
    ↓
atom-level attention
    ↓
每个原子的更新表示：[N,A,128]
```

同一个 token 内的原子再被聚合成 token 表示：

```text
[N,A,128]
    ↓ Linear
[N,A,384]
    ↓ 对每个 token 的有效原子做 mask mean
[N,384]
```

因此，atom/conformer 分支产生一个：

```text
[N,384]
```

的 atom-derived token embedding。

---

### 4.3 内部 target feature 和图中的 inputs

在当前典型配置下，token-level 的 63 维输入与 atom-derived 的 384 维 embedding 拼接：

```text
[N,63] + [N,384]
    ↓
内部 target_feat：[N,447]
```

但是，图中标记的 `inputs` 不是 `[N,447]`，而是经过 InputFeatureEmbedder 投影后的：

```text
inputs / s_i^inputs：[N,384]
```

所以要严格区分：

```text
target_feat：内部中间特征 [N,447]
inputs：     InputFeatureEmbedder 输出 [N,384]
```

`inputs` 再经过不同的线性投影，产生初始 single 和 pair：

```text
inputs：[N,384]
    ├── LinearNoBias → single：[N,384]
    └── 左右投影     → pair：  [N,N,128]
```

`inputs` 与 `single` 通道数相同，但不是同一个张量。`inputs` 是输入编码器输出，`single` 是交给主干网络的初始 single 状态。

---

## 5. 初始 pair 表示

对 token (i,j)，初始 pair 大致由以下信息构成：

```text
token i 的输入表示
token j 的输入表示
相对位置编码
token 之间的化学键关系
```

可以理解为：

```text
inputs[i] → 左侧 pair 特征
inputs[j] → 右侧 pair 特征
两者相加 → pair[i,j]
```

初始 pair 的形状是：

```text
[N,N,128]
```

后面还会加入：

- template pair embedding；
- MSA 产生的 pair 更新；
- recycling 的 pair 表示。

---

## 6. Template Module

Template Module 的输入主要是：

```text
当前 pair 表示：[N,N,128]
模板信息：     [T,N,24,3]、[T,N,24]、[T,N]
```

它对每一个模板分别处理。

### 6.1 模板残基对距离

模板中每个残基先提取代表位置，例如 pseudo-beta 位置，然后计算任意两个残基的距离：

```text
模板残基 i 和模板残基 j 的距离
```

距离被离散成 distogram bins，并编码成 pair 特征。

### 6.2 模板残基类型

模板中位置 (i) 和位置 (j) 的残基类型分别做 one-hot，再作为 pair 特征的一部分。

### 6.3 模板相对方向

先计算模板中残基 (i) 指向残基 (j) 的全局相对向量：

```text
模板 j 的坐标 − 模板 i 的坐标
```

然后乘以残基 (i) 的 backbone 局部坐标系逆变换，得到：

```text
从残基 i 自己的局部坐标系看，残基 j 位于哪个方向
```

这样得到的方向不依赖整个模板结构的全局旋转。

### 6.4 模板特征处理

距离、类型、方向、mask 和当前 pair embedding 分别经过线性层，投影到模板通道，形成：

```text
[N,N,64]
```

之后经过小型 Template Pairformer，最后投影到：

```text
template pair embedding：[N,N,128]
```

多个模板的结果进行平均或聚合，然后加入当前 pair：

```text
当前 pair：[N,N,128]
+ 模板 pair：[N,N,128]
    ↓
更新后的 pair：[N,N,128]
```

Template 的核心作用是把已知结构中的：

```text
残基类型 + 残基间距离 + 残基间方向
```

编码成 pair-level 结构先验。

---

## 7. MSA Module

MSA Module 使用完整的 MSA 表示：

```text
[M,N,64]
```

它与 token-level 的 MSA profile 不同：

```text
profile：每个位置单独的统计
完整 MSA：保留每条同源序列中各位置的联合模式
```

### 7.1 AF3 中的 MSA attention

AF3 没有 AF2 那种独立的 MSA column attention。

AF3 的 MSA attention 更准确地说是一个沿序列位置维度工作的 row-wise attention-like 操作：

```text
MSA：[M,N,64]
```

Pair 表示先产生位置 (i,j) 之间的权重。然后对每一条 MSA 序列，使用这组位置权重聚合该行不同位置的 MSA value：

```text
pair 决定位置之间的关注权重
MSA 每一行提供自己的 value
```

因此：

```text
[M,N,64] → [M,N,64]
```

不同 MSA 行共享位置权重，但每一行的 value 不同，所以输出仍然不同。

### 7.2 Outer Product Mean

对同一条 MSA 序列中的位置 (i,j)：

```text
取位置 i 的表示
取位置 j 的表示
分别投影成 left 和 right
做外积
```

对所有 MSA 序列的外积结果取平均，再展平并经过线性层，得到 pair 更新量：

```text
MSA：[M,N,64]
    ↓ left/right 投影
left/right：[M,N,32]
    ↓ 对每个 i,j 做外积
    ↓ 沿 M 求平均
    ↓ 线性投影
MSA pair update：[N,N,128]
```

然后加入当前 pair：

```text
pair：[N,N,128]
+ MSA pair update：[N,N,128]
    ↓
更新后的 pair：[N,N,128]
```

Outer Product Mean 的作用是把 MSA 中的协同进化关系转成 token 对之间的 pair 信息。

---

## 8. Pairformer

AF3 的 Pairformer 不再携带完整 MSA 矩阵，主要处理：

```text
single：[N,384]
pair：  [N,N,128]
```

它主要包含：

```text
Pair triangle multiplication
Pair triangle attention
Pair transition
Single self-attention with pair bias
Single transition
```

### 8.1 Pair 分支

Pairformer 继续通过 triangle multiplication 和 triangle attention 建模 token 对之间的三角关系：

```text
pair[i,k] 和 pair[k,j]
    ↓
更新 pair[i,j]
```

### 8.2 Single 分支

Single 进行 token-level self-attention：

```text
single：[N,384]
```

Pair 经过线性层变成 attention bias：

```text
pair：[N,N,128]
    ↓ Linear
pair attention bias
```

然后加到 single attention 的注意力分数中。

因此 AF3 的 Pairformer 可以概括为：

```text
没有完整 MSA 更新
single 进行带 pair bias 的 attention
pair 继续进行 triangle 类更新
```

AF3 的 MSA Module 较短，而 Pairformer 可以堆叠较多 block。这样后面的深层计算不再携带 MSA 的 (M) 维度。

---

## 9. Recycling

AF3 会把上一轮 trunk 的结果反馈给下一轮。

上一轮主要保存：

```text
上一轮 single：[N,384]
上一轮 pair：  [N,N,128]
```

下一轮重新运行 MSA/Template/Pairformer 主干时，会把它们作为额外输入。

概念上：

```text
第 1 轮：输入特征 → MSA/Template/Pairformer → single、pair
第 2 轮：输入特征 + 第 1 轮 single/pair → 主干 → 更新 single、pair
第 3 轮：输入特征 + 第 2 轮 single/pair → 主干 → 更新 single、pair
```

Recycling 不是简单把上一轮坐标直接复制给下一轮，而是把上一轮表示作为新的条件重新运行主干网络。

---

## 10. Diffusion 的输入

经过 recycling 后，主干网络得到最终的：

```text
single：[N,384]
pair：  [N,N,128]
target_feat：[N,447]，作为内部输入特征
```

Diffusion 每一个去噪步骤还接收：

```text
当前带噪原子坐标：x_noisy [N,A,3]
当前噪声水平：sigma，一个标量
原子属性、原子 mask、参考构象等
```

这里的 `x_noisy` 是扩散过程中真正被修改的对象。

AF3 不像 AF2 那样为每个残基显式预测一个旋转矩阵和平移向量，而是直接对原子坐标的 x、y、z 进行去噪。

---

## 11. Diffusion conditioning

### 11.1 Single condition

Diffusion 会把 trunk single 和 target features 拼接：

```text
single：[N,384]
target_feat：[N,447]
    ↓ 拼接
[N,831]
    ↓ Linear
single_cond：[N,384]
```

噪声水平也会编码成 noise embedding，并加入 single condition。

### 11.2 Pair condition

Diffusion 会把 trunk pair 和相对位置编码拼接：

```text
pair：[N,N,128]
relative encoding：[N,N,139]，具体随配置变化
    ↓ 拼接
[N,N,267]
    ↓ Linear
pair_cond：[N,N,128]
```

在 diffusion Transformer 中：

```text
single_cond → 调节 token 特征
pair_cond   → 生成 token attention bias
```

`single_cond` 的调节使用 adaptive LayerNorm。也就是先对当前特征做 LayerNorm，再由 single condition 生成每个通道的 scale 和 bias：

```text
归一化后的特征
    × single_cond 产生的 scale
    + single_cond 产生的 bias
```

---

## 12. Diffusion 的三个主要网络部分

Diffusion 的一次去噪步骤可以看成三段：

```text
Atom Attention Encoder
        ↓
Global Attention Transformer
        ↓
Atom Attention Decoder
```

### 12.1 Atom Attention Encoder

它的输入不是只有 `x_noisy`，而是：

```text
x_noisy：[N,A,3]
原子属性和参考几何
single 条件
pair 条件
```

首先构造每个原子的特征：

```text
x_noisy + 原子属性
    ↓
原子特征：[N,A,128]
```

Single 条件会从 token 级别投影到原子级别：

```text
single：[N,384]
    ↓ Linear
[N,128]
    ↓ 广播到每个 token 的原子
[N,A,128]
```

这个原子级条件通过 adaptive LayerNorm 产生 atom-level 的 scale 和 bias，调节原子特征。

Pair 条件则根据原子所属的 token 映射到原子对，并作为 atom attention 的 pair bias。

因此 atom attention 的概念流程是：

```text
x_noisy + 原子属性 + single 条件
    ↓
原子特征：[N,A,128]
    ↓
生成 Q、K、V
    ↓
QK 注意力分数 + atom pair bias
    ↓
更新每个原子的表示：[N,A,128]
```

之后将原子通道投影到 384 维，并按 token 聚合：

```text
[N,A,128]
    ↓ Linear
[N,A,384]
    ↓ 对同一 token 的有效原子做 mask mean
[N,384]
```

这个 `[N,384]` 是当前带噪坐标对应的 token-level 几何表示。

Atom Encoder 还会保留 atom-level skip features，供后面的 Atom Decoder 使用。

### 12.2 Global Attention Transformer

Global Attention 不再处理原子维度 (A)，而是处理 token：

```text
Atom Encoder 输出：[N,384]
single_cond：     [N,384]
pair_cond：       [N,N,128]
token mask：      [N]
```

流程是：

```text
Atom Encoder 的 token 特征
    ↓
single_cond 生成 adaptive LayerNorm 的 scale/bias
    ↓
生成 token-level Q、K、V
    ↓
QK 注意力分数 + pair_cond 产生的 bias
    ↓
更新 token 表示：[N,384]
```

Global Attention 的作用是让不同 token 之间进行全局结构交互。

它回答的是：

```text
结合整个复合物的关系，当前 token 应该如何调整？
```

### 12.3 Atom Attention Decoder

Global Transformer 的输出是 token-level 的：

```text
[N,384]
```

Atom Decoder 将它重新传回原子级，并结合 Atom Encoder 的 skip features：

```text
token 表示
+ Atom Encoder 的原子 skip features
+ per-atom conditioning
    ↓
Atom Decoder
    ↓
position_update：[N,A,3]
```

这个 `position_update` 表示每个原子的坐标应该如何修正。

---

## 13. Diffusion 去噪公式如何使用网络输出

三个 attention 模块组成一个去噪网络：

```text
Denoiser(x_noisy, sigma, single_cond, pair_cond)
    → position_update：[N,A,3]
```

网络输出不会直接变成旋转矩阵和平移矩阵，而是直接进入坐标更新。

AF3 使用带噪坐标和网络输出的组合，得到当前步骤的去噪坐标：

```text
denoised_position
    = skip_scaling × x_noisy
    + out_scaling × position_update
```

其中 `skip_scaling` 和 `out_scaling` 由当前噪声水平决定。

然后扩散采样器把当前坐标从较高噪声水平推进到较低噪声水平：

```text
当前带噪坐标
    ↓ denoiser
预测的去噪坐标
    ↓ diffusion sampler
下一噪声水平的坐标
```

不断重复这个过程：

```text
随机噪声坐标
    ↓
高噪声结构
    ↓
中等噪声结构
    ↓
低噪声结构
    ↓
最终原子坐标
```

AF3 不显式维护 AF2 那种每个残基的 (R,t) 刚体 frame，但仍然通过：

- 原子间相对位置；
- 化学键；
- pair 条件；
- 随机旋转和平移增强；
- 坐标去噪训练；

来学习三维几何关系。

---

## 14. Diffusion 的最终输出

如果生成一个结构，输出原子坐标大致是：

```text
atom_positions：[N,A,3]
atom_mask：     [N,A]
```

如果生成 (S) 个采样结构，则是：

```text
atom_positions：[S,N,A,3]
atom_mask：     [S,N,A]
```

其中：

- (S)：采样结构数量；
- (N)：token 数；
- (A)：每个 token 的原子槽位数；
- 3：原子的 x、y、z 坐标。

padding 原子通过 mask 忽略，最终再转换成完整的蛋白质、核酸和配体结构。

---

## 15. AF2 与 AF3 的核心区别

### AF2

```text
MSA + Pair
    ↓ 48 个 Evoformer blocks
single + pair
    ↓ IPA / Structure Module
残基刚体 frame R,t
    ↓
原子坐标
```

AF2 的 Structure Module 中显式维护局部坐标系和刚体变换。

### AF3

```text
MSA Module：较少的 MSA 处理
    ↓
Pairformer：继续更新 single 和 pair
    ↓
Atom-level diffusion
    ↓
直接生成原子 x,y,z 坐标
```

AF3 的主要变化是：

- token 不再只代表蛋白质残基；
- MSA 处理和 Pairformer 分开；
- Pairformer 阶段不再携带完整 MSA；
- 配体通过 atom-level 表示和 conformer 信息进入模型；
- 去掉 AF2 的显式 IPA 和 per-residue rigid frame；
- 使用 diffusion 直接生成所有原子的三维坐标。

最核心的总结是：

> AF3 用 MSA 和模板学习 token/pair 条件，用 atom attention 处理当前原子几何，用 global token attention 传播全局结构关系，再用 atom decoder 直接预测每个原子的坐标去噪更新。

---

## 16. Diffusion 中“加噪”和“去噪”到底在哪里

理解 AF3 diffusion 时，需要把两个部分分开：

```text
DiffusionHead：单步去噪网络
sample()：     加噪、噪声调度、多步采样和坐标更新
```

图中的 Atom Attention Encoder、Global Attention Transformer 和 Atom Attention Decoder 属于 `DiffusionHead`。它们只负责：

```text
给定当前带噪坐标、噪声水平和结构条件
    ↓
预测当前坐标应该去噪到哪里
```

真正的多步扩散循环由外部采样器完成。

---

## 17. 训练时如何加噪

假设训练样本的真实原子坐标是：

```text
x_true：[N,A,3]
```

训练时随机选择一个噪声强度 `sigma`，再生成同样形状的高斯噪声：

```text
noise：[N,A,3]
```

构造带噪坐标：

```text
x_noisy = x_true + sigma × noise
```

然后将以下信息输入同一个 Diffusion 网络：

```text
x_noisy
sigma
single 条件
pair 条件
原子属性和 mask
```

网络输出一个去噪坐标估计或坐标更新，训练目标是让它接近真实结构对应的去噪目标。直观上可以理解为：

```text
x_true
    ↓ 随机加噪
x_noisy
    ↓ DiffusionHead
x_pred
    ↓ 与真实坐标/去噪目标比较
计算损失
```

训练时不需要先把一个结构完整地加噪 200 次，再依次展开 200 次去噪。每个训练样本通常随机抽取一个噪声强度，做一次前向计算和一次损失计算。

---

## 18. 为什么要使用很多个 sigma

`sigma` 表示当前输入坐标的噪声强度。它可以是连续范围中的任意值，而不是只能取固定的第 1 步、第 2 步或第 3 步。

例如：

```text
sigma 很大：几乎是随机结构
sigma 中等：整体结构已经有一定形状
sigma 很小：只剩下原子位置的小误差
```

同一个网络需要学会处理所有这些情况，因此训练过程中不断随机采样不同的 `sigma`：

```text
训练样本 1：sigma = 80
训练样本 2：sigma = 12
训练样本 3：sigma = 0.5
训练样本 4：sigma = 3
...
```

这些不同 sigma 的输入都使用同一套 Diffusion 网络权重，但网络会收到 sigma embedding，因此知道当前噪声有多大。

```text
x_noisy + sigma embedding
    ↓ 同一个网络
去噪坐标估计
```

网络因此逐渐学会：

```text
高 sigma：恢复整体拓扑和粗略构象
中 sigma：恢复残基、核苷酸和配体的相对布局
低 sigma：修正精细原子坐标
```

---

## 19. 不同 sigma 如何共享权重

假设一个 batch 中有三个不同噪声水平的训练样本：

```text
样本 1：x_noisy_1、sigma_1
样本 2：x_noisy_2、sigma_2
样本 3：x_noisy_3、sigma_3
```

它们的输入当然不同，但都调用同一个网络：

```text
DiffusionModel(x_noisy_1, sigma_1) → prediction_1
DiffusionModel(x_noisy_2, sigma_2) → prediction_2
DiffusionModel(x_noisy_3, sigma_3) → prediction_3
```

如果三个样本是不同的训练结构，它们各自有自己的真实坐标；如果是同一个结构重复使用不同 sigma，那么三个预测都和同一个真实坐标比较。

得到各自损失后进行平均：

```text
loss = average(loss_1, loss_2, loss_3)
    ↓
一次反向传播
    ↓
更新同一套网络参数
```

这里的“并行”不是说不同 sigma 的输入相同，而是说：

```text
不同输入可以同时经过同一个网络
```

训练时通常不是：

```text
sigma_1 → 更新参数
sigma_2 → 再更新参数
sigma_3 → 再更新参数
```

而是：

```text
不同 sigma 的样本
    → 共享网络
    → 汇总损失
    → 一起更新权重
```

---

## 20. 推理时的多步去噪

训练完成后，推理阶段没有真实坐标作为起点，因此从随机坐标开始：

```text
positions = 随机高斯坐标
positions *= 最大噪声水平
```

然后建立从大到小的噪声 schedule：

```text
sigma_0 > sigma_1 > sigma_2 > ... > sigma_T
```

在每一个采样步骤中：

```text
当前 positions
    ↓ 根据当前噪声水平加入少量噪声
positions_noisy
    ↓ 调用同一个 DiffusionHead
positions_denoised
    ↓ 采样器根据去噪结果更新坐标
positions_next
```

伪代码是：

```text
positions = 初始随机坐标

for sigma in noise_schedule:
    positions_noisy = positions + 少量随机噪声

    positions_denoised = DiffusionHead(
        positions_noisy,
        sigma,
        single,
        pair,
    )

    positions = sampler_update(
        positions_noisy,
        positions_denoised,
        sigma,
    )
```

每一步都调用相同的网络权重，但输入的坐标和 sigma 不同。推理阶段不进行反向传播，也不更新模型参数。

在当前代码中，扩散采样器负责：

- 创建噪声水平 schedule；
- 初始化随机坐标；
- 在相邻噪声水平之间添加少量随机噪声；
- 调用 denoising step；
- 根据当前和下一噪声水平推进坐标。

因此图中看到的：

```text
x_noisy → Atom Encoder → Global Transformer → Atom Decoder → 新坐标
```

只是一次去噪网络调用，不是完整的扩散过程。完整过程是外部采样器多次调用这套网络。

---

## 21. AF3 为什么看起来像 Flow Matching

AF3 不是传统 DDPM 中常见的“直接预测本步加入的噪声”，而是更接近连续噪声水平的 denoising diffusion/EDM 风格：

```text
带噪坐标 + sigma
    ↓
预测去噪后的坐标
```

采样器再根据当前带噪坐标和去噪坐标计算下一步的移动方向。因此它看起来像：

```text
当前位置
    ↓ 预测方向
沿方向移动一步
```

这和 Flow Matching 的直觉很接近，但二者的训练目标不同：

```text
AF3 diffusion：
    预测去噪坐标或干净坐标估计

Flow Matching：
    直接预测从噪声到数据的速度/向量场
```

可以用下面的方式区分：

```text
传统 DDPM：预测噪声
AF3 diffusion：预测去噪坐标
Flow Matching：预测运动速度
```

AF3 和 Flow Matching 在训练效率上有相似之处：训练一个样本时只需要随机选择一个噪声/时间点，进行一次网络前向和一次损失计算，不需要在训练时展开完整的多步生成轨迹。

但推理时，二者都可能需要多次调用网络，最终速度取决于采样步数和每次网络计算成本。

---

## 22. Diffusion 部分的最终总结

```text
训练：
    真实坐标
        ↓ 随机选择 sigma 并加噪
    带噪坐标
        ↓ 同一个 Diffusion 网络
    预测去噪坐标
        ↓ 和真实去噪目标比较
    计算损失并更新共享权重
```

```text
推理：
    随机坐标
        ↓ 大 sigma
    调用 Diffusion 网络
        ↓
    坐标稍微变干净
        ↓ 更小 sigma，再调用同一个网络
        ↓
    重复多次
        ↓
    最终原子坐标
```

因此，三个 attention 模块是“单步去噪器”，而不是整个扩散算法；加噪公式、sigma schedule 和多步坐标更新由训练过程和外部 sampler 共同完成。

---

## 23. AF3 的 diffusion 和 Flow Matching 到底是什么关系

### 23.1 两者都可以从噪声开始生成结构

二者都有相似的外层框架：

```text
随机噪声坐标
    ↓ 多次调用神经网络
逐渐变得有结构
    ↓
最终原子坐标
```

所以从直观的“从噪声走向结构”来看，AF3 diffusion 和 Flow Matching 很像。

但是，真正决定二者区别的是网络训练时要预测什么。

### 23.2 AF3 diffusion 的训练目标

给定真实坐标 `x_true`，随机采样一个连续噪声强度 `sigma`，构造：

```text
x_noisy = x_true + sigma × noise
```

网络接收：

```text
x_noisy、sigma、single 条件、pair 条件、原子属性
```

并输出一个去噪坐标估计，记作：

```text
D_theta(x_noisy, sigma, condition)
```

训练时主要要求这个估计接近真实坐标或对应的去噪目标：

```text
D_theta(...) ≈ x_true
```

因此 AF3 的核心问题是：

> 当前坐标带有这么大的噪声，在给定生物学条件后，它应该被拉回到什么样的干净结构附近？

### 23.3 Flow Matching 的训练目标

Flow Matching 通常先规定一条从噪声到真实结构的路径。例如最简单的直线路径是：

```text
x_t = (1 - t) × x_noise + t × x_true
```

这条路径的目标速度是：

```text
v_target = x_true - x_noise
```

网络接收 `x_t` 和 `t`，直接预测速度：

```text
v_theta(x_t, t, condition) ≈ v_target
```

推理时使用预测速度更新坐标：

```text
x_next = x_current + step_size × v_theta(...)
```

因此 Flow Matching 的核心问题是：

> 当前处在路径的这个位置，坐标应该以什么方向和速度移动？

### 23.4 最简区别

```text
AF3 diffusion：预测“应该去哪里”——去噪坐标
Flow Matching：预测“应该怎么走”——速度向量
```

根据一个去噪坐标，也可以在采样器中计算出近似更新方向；因此两者的采样过程可能看起来相似。但“可以由一个量推导出另一个量”不代表训练目标和模型已经相同。

---

## 24. 为什么 AF3 的采样不是完全随机的

用户容易产生这样的疑问：如果每次初始坐标都是随机噪声，AF3 怎么保证最后能得到真实结构？关键在于随机性只决定从分布中的哪一个位置出发，而不是让模型无规则地移动。

### 24.1 条件信息限制了可行结构

每次 DiffusionHead 都会看到相同的结构条件：

```text
序列和 token 类型
MSA / template 信息
pair 表示
原子属性和化学键
配体构象信息
```

因此网络不是对任意结构做无条件去噪，而是在学习：

```text
p(结构 | 序列、模板、MSA、配体和其他条件)
```

这些条件把随机坐标限制到符合生物学和化学条件的结构分布中。

### 24.2 网络提供方向，sampler 控制步长

在每一步，随机性主要来自初始噪声，以及具体采样方案中可能使用的噪声扰动；真正决定坐标如何变化的是 DiffusionHead 的去噪预测和 sampler 的更新规则：

```text
当前坐标 + 当前 sigma
    ↓ DiffusionHead
去噪方向/去噪坐标信息
    ↓ sampler 控制步长和下一 sigma
下一状态
```

所以它不是“每一步随机猜一个坐标”，而是：

> 在条件信息约束下，用共享网络反复提供有方向的去噪更新。

### 24.3 终点不一定是唯一实验构象

结构预测的“正确”并不总是唯一坐标。蛋白质侧链、配体构象、柔性环和复合物结合方式可能存在多个合理状态。因此 AF3 的随机采样并不要求所有随机种子都回到同一个坐标，而是希望多个样本覆盖条件分布中的合理结构。

可以把目标理解成：

```text
不是保证每次得到唯一的 x_true
而是让生成结果落在合理的 p(structure | condition) 中
```

---

## 25. 如果把 AF3 改成 Flow Matching，会发生什么

这不是简单把网络最后的输出名称从 `position_update` 改成 `velocity`。至少需要重新设计以下部分：

```text
1. 噪声到真实结构的路径 x_t
2. 时间变量 t 或噪声变量的编码方式
3. 速度监督目标 v_target
4. 坐标更新的 ODE/integrator
5. 原子对称性、旋转平移和多构象处理
6. 蛋白质、核酸、配体混合体系的损失
```

### 25.1 可能的优势

- 可以使用较少的积分步骤，推理速度可能更快；
- 速度场形式直接，采样轨迹更容易解释；
- 可以使用确定性 ODE 采样，重复性可能更强；
- 如果路径设计得好，可能减少采样过程中的随机扰动。

### 25.2 可能的代价

- 直线路径不一定适合复杂蛋白质和配体的多模态结构分布；
- 一个噪声结构到多个合理终点时，单一速度目标可能产生平均化趋势；
- 原子坐标存在整体旋转、平移、对称原子交换等等价表示；
- 配体的键长、键角、手性和蛋白质局部几何需要额外处理；
- 少采样步数不代表单步网络一定更容易训练；
- 直接替换目标函数后，原有 AF3 权重通常不能直接继续使用，需要重新训练或大规模微调。

因此 Flow Matching 可能使 AF3 更快，也可能改变生成结构的多样性、稳定性和精度。它是一个合理的研究方向，但不能从理论上保证一定优于 AF3 的 diffusion 方案。

---

## 26. Diffusion、Flow Matching 和 AF3 的统一对照表

| 项目 | 传统噪声预测扩散 | AF3 风格连续噪声去噪 | Flow Matching |
|---|---|---|---|
| 输入 | 带噪坐标和离散/连续时间 | 带噪原子坐标和连续 sigma | 路径上的中间坐标和 t |
| 网络输出 | 噪声或 score | 干净坐标/去噪坐标估计 | 速度向量 |
| 训练标签 | 注入的噪声或 score | 真实坐标或去噪目标 | 路径的目标速度 |
| 推理方式 | 按噪声 schedule 反向采样 | 按 sigma 从大到小反复去噪 | 积分速度场 |
| 随机性 | 初始噪声及采样器 | 初始噪声及采样器 | 通常可使用确定性 ODE，也可保留随机性 |
| 结构多样性 | 通过不同噪声样本获得 | 通过不同随机初始状态获得 | 取决于路径、速度场和采样方式 |
| AF3 是否属于该类 | 不是主要表述 | 是更合适的描述 | 不是当前 AF3 的直接定义 |

---

## 27. 整个 AF3 流程的一句话版本

```text
搜索模块收集序列、模板和配体几何先验
    ↓
Input Embedding 把原子、token、MSA 摘要和位置关系编码成 inputs、single、pair
    ↓
MSA Module 把同源序列中的细粒度协同变化注入 pair
    ↓
Template Module 把已知三维结构先验注入 pair
    ↓
Pairformer 反复更新 single 和 pair，形成复合物级条件表示
    ↓
DiffusionHead 在每个 sigma 下处理带噪原子坐标
    ↓
Atom Encoder 聚合局部原子信息，Global Transformer 传播全局 token 信息，Atom Decoder 输出原子坐标更新
    ↓
Sampler 按 sigma schedule 多次调用共享的 DiffusionHead
    ↓
得到蛋白质、核酸和配体的最终 atom_positions：[S,N,A,3]
```

最需要记住的是：

```text
single / pair：告诉扩散模型“应该生成什么结构”
x_noisy：       告诉扩散模型“当前已经生成到什么程度”
sigma：         告诉扩散模型“当前噪声有多大”
DiffusionHead： 给出一次去噪预测
Sampler：       把一次次预测串成完整生成过程
```
