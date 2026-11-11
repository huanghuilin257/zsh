# AlphaFold2 理解总结与修正

## 问题

好的，alphafold2我已经基本上看懂了，就是利用msa多条序列的信息，得到初始的msa矩阵，矩阵的每个初始都是氨基酸onthot和同源搜索得到的这个位置每种氨基酸的概率，两个都是20多维，加起来矩阵的每个位置大概是40多维，还有pair矩阵，初始时氨基酸onehot经过线性变换得到128维的信息，然后msa矩阵经过行和列的自注意力，然后外积然后经过变化加到pair矩阵的每个对应的位置ij，pair矩阵会经过以k为起点和以k为终点的入度和出度的三角乘法矩阵，和以行进行的三角注意力矩阵和以列进行的三角注意力矩阵，这样更新完成后得到信息更新后的pair矩阵，这样编码器就算结束了，然后取出msa中需要的那一行序列比如人源的那一行序列结合pair矩阵作为解码器的输入，解码器有ipa不变点注意力模块组成，这个模块本质上就是聚合各种信息来更新序列比如更新s i，先获取aij也就是注意力权重，权重的来源有三个，第一个是si和sj做自注意力也就是q乘以k得到第一个代表了相似信息得到的权重就i越大，第二个是si和sj分别乘以各自局部坐标系得到全局的信息，si的global减去sj的global得到距离信息，距离越大sj对si的权重就越小，第三个是原先pair矩阵中的ij这个代表之前编码器得到的演化信息，所以是第一个+第二个-第三个，这样就得到了aij，然后aij因为来自于三种信息，所以aij也要乘以三种信息得到更新后的si，比如第一种是aij乘以qj，第二种是aij乘以pair的ij也就是Zij，第三种是aij先乘以qj然后乘以Rj这样就转换到了全局，再乘以Ri的负这样就转换到了i的局部坐标系，最终把这些三个信息concat，并且还要concat第三种的标量也就是向量变标量。最终就更新了Si。然后后如何从si这种张量变成坐标呢，其实就是si线性转换为6个数值，前三个是后面转换为秩为1的R矩阵，后三个是平移矩阵t。这样反复8次，就得到8个R和8个t，再乘以原始坐标x，就得到新的坐标。

## 修正与更准确的理解

你的整体理解已经很接近了。需要修正几个关键点：

### 1. MSA 初始特征不只是 one-hot + 氨基酸概率

还会包含缺失信息、删除信息等特征。可以简化为：

\[
\text{MSA特征}
\approx
\text{每行序列特征}
+
\text{MSA profile}
+
\text{缺失/删除特征}
\]

初始 pair 表示也不只来自氨基酸 one-hot 的线性变换，还可能包含相对位置、模板和 recycling 信息。

### 2. “解码器”只是类比

Evoformer 后面严格来说是 Structure Module，不完全等同于 Transformer 的 decoder。它接收：

\[
s_i=\text{MSA第一行经过变换后的 single 表示}
\]

\[
Z_{ij}=\text{Evoformer输出的 pair 表示}
\]

以及当前残基的三维刚体坐标系。

### 3. IPA 的注意力分数不是“第一项 + 第二项 - 第三项”

更准确的是：

\[
\ell_{ijh}
=
\ell_{ijh}^{scalar}
+
\ell_{ijh}^{point}
+
\ell_{ijh}^{pair}
\]

其中：

\[
\ell_{ijh}^{scalar}
=
q_{ih}^{scalar}\cdot k_{jh}^{scalar}
\]

\[
\ell_{ijh}^{point}
=
-\frac12
\sum_p
\left\|
q_{ihp}^{global}
-
k_{jhp}^{global}
\right\|^2
\]

\[
\ell_{ijh}^{pair}
=
W_h^z Z_{ij}
\]

point 项因为前面有负号，所以距离越远，分数越低；但 pair 项本身可以是正也可以是负，不应该统一称为“减去第三项”。

### 4. Value 不是三种信息互相相乘

同一个注意力权重分别聚合三类信息：

\[
y_i^{scalar}
=
\sum_j a_{ijh}v_{jh}^{scalar}
\]

\[
y_i^{point,global}
=
\sum_j a_{ijh}v_{jh}^{point,global}
\]

\[
y_i^{pair}
=
\sum_j a_{ijh}Z_{ij}
\]

其中 point value 的过程是：

\[
v_j^{point,local}
\overset{T_j}{\longrightarrow}
v_j^{point,global}
\]

聚合后再：

\[
y_i^{point,global}
\overset{T_i^{-1}}{\longrightarrow}
y_i^{point,local}
\]

最后拼接：

\[
\operatorname{Concat}
\left(
y_i^{scalar},
y_i^{point,local},
\left\|y_i^{point,local}\right\|,
y_i^{pair}
\right)
\]

所以不是把三种 Value 互相相乘，而是“分别加权聚合，再拼接”。

### 5. \(s_i\) 不是直接生成最终的 \(R,t\)

每次 Structure Module 迭代时：

\[
s_i^{(k)}
\rightarrow
\Delta T_i^{(k)}
\]

其中 \(\Delta T_i^{(k)}\) 通常由 6 个数表示：

\[
\Delta T_i^{(k)}
=
(\Delta r_i^{(k)},\Delta t_i^{(k)})
\]

前三个数不是直接的旋转矩阵，而是旋转更新的参数，通常对应小的 quaternion update；后三个数是平移更新。

然后更新已有坐标系：

\[
T_i^{(k)}
=
\Delta T_i^{(k)}
\circ
T_i^{(k-1)}
\]

因此不是每次独立预测一个全新的 \(R,t\)，而是不断更新当前的刚体坐标系。

### 6. 最后不是和“原始结构坐标”相乘

最终得到：

\[
T_i^{(8)}=(R_i^{(8)},t_i^{(8)})
\]

再把残基局部坐标系中的原子模板坐标变换到全局空间：

\[
x_{ia}^{global}
=
R_i^{(8)}
x_{ia}^{local}
+
t_i^{(8)}
\]

这里的 \(x_{ia}^{local}\) 是标准残基几何和侧链扭转角决定的局部原子坐标，不是真实结构中的原始坐标。

侧链扭转角也是由最终的 \(s_i\) 预测的，因此完整过程是：

\[
s_i^{(8)}
\rightarrow
\text{刚体坐标系更新}
+
\text{侧链扭转角}
\rightarrow
\text{局部原子坐标}
\rightarrow
\text{全局原子坐标}
\]

最后还要区分：

- **8 次 Structure Module 迭代**：每次运行 IPA 并更新结构；
- **recycling**：把上一轮完整预测的结构和表示反馈给下一轮 Evoformer。
