---
title: Continuous Multimodal Field Operator
aliases:
  - 连续多模态场算子
  - Token-Free Multimodal Architecture
type: idea
status: incubating
domains:
  - ai-ml
  - multimodal
created: 2026-07-27
updated: 2026-08-04
tags: [idea, multimodal, neural-operator, continuous-representation, token-free, representation-learning, foundation-model]
---

# Continuous Multimodal Field Operator

## 1. 核心想法

当前大模型通常将文本切分为 token、将图像切分为 patch，再使用 Transformer 在有限序列上进行计算。即使 token 或 patch 的长度可以动态变化，模型内部的基本状态仍然是一个有限矩阵：

$$
H\in\mathbb{R}^{N\times d},
$$

其中每一行都对应一个预先确定或动态产生的离散计算单元。

本方案希望改变的不是 tokenizer 本身，而是模型的**状态类型**：

> 不再将 token、patch、slot、对象或语义实体作为网络的基本计算单元，而是将每种模态表示为其原生坐标域上的函数，并学习函数空间之间的连续算子。

整体结构为：

$$
\boxed{
\text{Raw observations}
\rightarrow
\text{Continuous field lifting}
\rightarrow
\text{Coupled field operators}
\rightarrow
\text{Output field}
\rightarrow
\text{Boundary discretization}
}
$$

输入端的 byte、像素和音频采样点只是原始观测；计算中的有限节点只是数值求积点；输出端的字符和像素只是接口表示。离散性仍然存在于计算机实现中，但不再决定模型内部的基本语义单位。

---

## 2. 设计原则

### 2.1 不预定义语义单位

架构中不显式设置：

- token 或 subword；
- 固定图像 patch；
- object slot；
- 语义实体节点；
- 关系图；
- 固定数量的 latent token；
- `[CLS]` 全局向量；
- 预定义的短语、对象或事件边界。

实体、关系、短语和概念应当是训练后隐状态中的涌现结构，而不是架构中的变量类型。

### 2.2 保留各模态的原生几何

不同模态不需要被强制拉平成同一条序列。

- 文本保留一维顺序域；
- 图像保留二维空间域；
- 音频保留一维时间域；
- 视频保留二维空间与时间构成的时空域；
- 3D 数据保留三维空间域。

跨模态统一发生在**函数变换规律**和**通道空间**中，而不是通过共享一个统一的离散位置集合。

### 2.3 连续域不等于强制平滑

文字、图像边缘、代码和公式包含大量不连续或高频信息。因此隐状态不应被限制为处处光滑函数，而更适合定义在：

$$
h_m\in L^2(\Omega_m;\mathbb{R}^d)
$$

或允许跳变的有界变差空间：

$$
h_m\in BV(\Omega_m;\mathbb{R}^d).
$$

这里的"连续"是指状态定义在连续坐标域上，而不是要求函数值处处平滑。

### 2.4 数值离散与语义离散分离

实际计算仍然需要有限采样点，但这些点必须满足：

1. 没有持久身份；
2. 位置可以在不同 batch 中变化；
3. 数量可以在训练与推理时变化；
4. 模型参数不依赖采样点数量；
5. 增加采样点时结果应逐渐收敛。

因此，数值求积点不等同于 token。

---

# 3. 输入表示：坐标化原始观测

对于模态 \(m\)，输入表示为带坐标的观测集合：

$$
X_m=
\left\{
(\xi_i^m,v_i^m)
\right\}_{i=1}^{N_m},
$$

其中：

- \(\xi_i^m\) 是观测坐标；
- \(v_i^m\) 是该位置的原始值；
- \(N_m\) 可以随输入分辨率、长度或采样率变化。

不同模态的原生域如下：

| 模态 | 原生坐标域 \(\Omega_m\) | 原始观测 |
|---|---|---|
| 文本 | 一维有序域 | UTF-8 bytes 或 Unicode code points |
| 图像 | 二维空间域 | RGB、深度或其他像素值 |
| 音频 | 一维时间域 | waveform samples |
| 视频 | 二维空间 × 时间 | RGB 时空采样 |
| 3D | 三维空间域 | 点、体素或传感器观测 |

统一地，可以将输入写成向量值观测测度：

$$
\mu_m
=
\sum_{i=1}^{N_m}
\omega_i^m
E_m(v_i^m)
\delta_{\xi_i^m},
$$

其中：

- \(E_m\) 是最底层的观测值编码器；
- \(\omega_i^m\) 是与采样密度相关的数值积分权重；
- \(\delta_{\xi_i^m}\) 表示在该坐标存在一个观测。

该测度只描述"在哪里观察到了什么"，不携带任何对象、单词或语义实体假设。

---

# 4. 连续提升层

每个模态具有一个 lifting operator：

$$
\mathcal{L}_m:\mu_m\rightarrow h_m^0,
$$

其中：

$$
h_m^0:\Omega_m\rightarrow\mathbb{R}^d.
$$

其一般形式为：

$$
h_m^0(x)
=
b_m(x,\eta_m)
+
\int
K_m^{\mathrm{lift}}
\left(
x,\xi,E_m(v),\eta_m
\right)
\,d\mu_m(\xi,v).
$$

离散实现为：

$$
h_m^0(x)
=
b_m(x,\eta_m)
+
\sum_{i=1}^{N_m}
\omega_i^m
K_m^{\mathrm{lift}}
\left(
x,\xi_i^m,E_m(v_i^m),\eta_m
\right).
$$

其中 \(\eta_m\) 可以包含：

- 文本长度；
- 图像宽高；
- 音频采样率；
- 视频帧率；
- 坐标归一化尺度；
- 传感器参数。

## 4.1 文本提升

文本可以使用 UTF-8 byte 作为输入观测：

$$
X_{\mathrm{text}}
=
\{(i,b_i)\}_{i=1}^{L},
\qquad
b_i\in\{0,\ldots,255\}.
$$

对应的初始文本场为：

$$
h_t^0(x)
=
\sum_{i=1}^{L}
K_t^{\mathrm{lift}}(x,i,b_i)
E_{\mathrm{byte}}(b_i).
$$

byte 仅用于构造初始场。进入主干后，不再保留"第 \(i\) 个 byte 对应第 \(i\) 个隐藏向量"的固定结构。

文字顺序由连续坐标 \(x\) 和相对位置 \(x-\xi\) 表达，而不是由 token position embedding 表达。

## 4.2 图像提升

对于图像：

$$
X_{\mathrm{img}}
=
\left\{
((x_i,y_i),c_i)
\right\}_{i=1}^{HW},
$$

其中 \(c_i\) 为像素值。

初始图像场为：

$$
h_i^0(x,y)
=
\sum_j
\omega_j
K_i^{\mathrm{lift}}
\left(
(x,y),(x_j,y_j),c_j
\right)
E_{\mathrm{rgb}}(c_j).
$$

图像不需要被切成固定大小的 patch。改变输入分辨率只会改变观测数量和积分精度，不改变模型结构。

## 4.3 音频与视频提升

音频：

$$
h_a^0(t)
=
\sum_j
\omega_j
K_a^{\mathrm{lift}}(t,t_j,s_j)
E_a(s_j).
$$

视频：

$$
h_v^0(x,y,t)
=
\sum_j
\omega_j
K_v^{\mathrm{lift}}
\left(
(x,y,t),(x_j,y_j,t_j),c_j
\right)
E_v(c_j).
$$

---

# 5. 主干状态

模型在深度 \(\tau\) 处的完整状态为：

$$
\mathcal{H}(\tau)
=
\left\{
h_m(\cdot,\tau)
\right\}_{m=1}^{M}.
$$

例如文字—图像模型：

$$
\mathcal{H}(\tau)
=
\left(
h_t(\cdot,\tau),
h_i(\cdot,\tau)
\right).
$$

其中：

$$
h_t:\Omega_t\times[0,T]\rightarrow\mathbb{R}^d,
$$

$$
h_i:\Omega_i\times[0,T]\rightarrow\mathbb{R}^d.
$$

不同模态保持在各自的函数空间中：

$$
h_t\in\mathcal{H}_t,
\qquad
h_i\in\mathcal{H}_i,
\qquad
h_a\in\mathcal{H}_a.
$$

跨模态统一不再表示为：

$$
E_{\mathrm{text}}(x)
\approx
E_{\mathrm{image}}(y)
\in\mathbb{R}^d,
$$

而是学习不同函数空间之间的映射：

$$
\mathcal{K}_{m\rightarrow n}:
\mathcal{H}_m\rightarrow\mathcal{H}_n.
$$

---

# 6. 连续多模态算子主干

主干学习耦合函数动力学：

$$
\frac{\partial h_m(x,\tau)}{\partial\tau}
=
\Phi_{m,\theta}
\left(
x,
\{h_n(\cdot,\tau)\}_{n=1}^{M}
\right).
$$

展开为：

$$
\frac{\partial h_m(x,\tau)}{\partial\tau}
=
P_{m,\theta}(h_m(x,\tau),x,\tau)
+
\mathcal{S}_{m,\theta}[h_m](x,\tau)
+
\sum_{n\neq m}
\mathcal{C}_{n\rightarrow m,\theta}
[h_n,h_m](x,\tau).
$$

其中包含三类操作：

1. 点态通道变换；
2. 模态内连续传播；
3. 跨模态连续传播。

---

## 6.1 点态通道变换

点态通道变换作用于函数在任意坐标处的取值：

$$
P_m(h(x),x,\tau)
=
W_{2,m}
\sigma
\left(
W_{1,m}h(x)
+
e_m(x,\tau)
\right).
$$

它类似 Transformer 中的 FFN，但输入不是 token 矩阵中的某一行，而是函数在坐标 \(x\) 处的值。

---

## 6.2 模态内局部算子

局部结构通过邻域积分传播：

$$
\mathcal{S}_m^{\mathrm{local}}[h](x)
=
\int_{\mathcal{N}_m(x)}
K_m^{\mathrm{local}}
\left(
x,y,h(x),h(y)
\right)
V_mh(y)\,dy.
$$

其中 \(\mathcal{N}_m(x)\) 是原生坐标域中的局部邻域。

局部邻域不代表：

- 单词；
- 句子；
- 图像物体；
- 固定 patch；
- 语义实体。

它只是数值和几何上的局部区域。语义结构由核函数和隐藏状态自行学习。

---

## 6.3 模态内全局算子

长距离依赖通过非局部积分实现：

$$
\mathcal{S}_m^{\mathrm{global}}[h](x)
=
\frac{
\displaystyle
\int_{\Omega_m}
\exp s_m(x,y)
V_mh(y)\,dy
}{
\displaystyle
\int_{\Omega_m}
\exp s_m(x,y)\,dy
},
$$

其中：

$$
s_m(x,y)
=
\frac{
q_m(h(x),x)^\top
k_m(h(y),y)
}{
\sqrt{d}
}
+
b_m(x,y).
$$

这可以被理解为定义在函数空间上的连续注意力。普通 attention 是该积分算子的有限数值近似，而不是其基本定义。

---

## 6.4 跨模态算子

从模态 \(n\) 到模态 \(m\) 的更新为：

$$
\mathcal{C}_{n\rightarrow m}
[h_n,h_m](x)
=
\frac{
\displaystyle
\int_{\Omega_n}
\exp s_{mn}(x,y)
V_{mn}h_n(y)\,dy
}{
\displaystyle
\int_{\Omega_n}
\exp s_{mn}(x,y)\,dy
},
$$

其中：

$$
s_{mn}(x,y)
=
q_m(h_m(x),x)^\top
k_n(h_n(y),y)
+
b_{mn}(x,y).
$$

例如图像到文本：

$$
\mathcal{C}_{i\rightarrow t}
[h_i,h_t](x)
=
\frac{
\displaystyle
\int_{\Omega_i}
\exp s_{ti}(x,y)
V_{ti}h_i(y)\,dy
}{
\displaystyle
\int_{\Omega_i}
\exp s_{ti}(x,y)\,dy
}.
$$

文本到图像：

$$
\mathcal{C}_{t\rightarrow i}
[h_t,h_i](y)
=
\frac{
\displaystyle
\int_{\Omega_t}
\exp s_{it}(y,x)
V_{it}h_t(x)\,dx
}{
\displaystyle
\int_{\Omega_t}
\exp s_{it}(y,x)\,dx
}.
$$

架构不规定哪些区域对应对象、短语或关系。如果训练后某一段文字与某一区域形成稳定高耦合，该结构是模型学习出的结果。

---

# 7. 连续深度与离散实现

理想模型将深度写成连续变量：

$$
\frac{\partial \mathcal{H}(\tau)}{\partial\tau}
=
\Phi_\theta(\mathcal{H}(\tau),\tau).
$$

实际第一版可以使用残差算子层进行显式 Euler 离散：

$$
h_m^{\ell+1}(x)
=
h_m^\ell(x)
+
\Delta\tau_\ell
\left[
P_m^\ell(x)
+
\mathcal{S}_m^\ell(x)
+
\sum_{n\neq m}
\mathcal{C}_{n\rightarrow m}^\ell(x)
\right].
$$

因此，固定层数只是连续深度动力学的一种数值求解方式，而不是理论上必须存在的离散层级。

后续可以替换为：

- 自适应 Neural ODE 求解器；
- implicit layer；
- equilibrium solver；
- adaptive computation time；
- event-driven continuous dynamics。

---

# 8. 数值求积与计算效率

理论积分需要通过有限节点近似：

$$
\int_{\Omega}f(y)\,dy
\approx
\sum_{j=1}^{N}w_jf(y_j).
$$

这里的 \(y_j\) 是数值求积点，而不是 token。为了防止模型重新退化成固定序列，需要在训练中：

- 随机改变求积点数量；
- 随机改变求积位置；
- 使用不同分辨率和采样率；
- 对不同求积方案施加输出一致性约束；
- 避免持久性的固定 latent index。

## 8.1 局部—全局核分解

完整的全局积分可能产生 \(O(N^2)\) 复杂度。可以将核分解为：

$$
K(x,y)
=
K_{\mathrm{local}}(x,y)
+
K_{\mathrm{global}}(x,y).
$$

全局核使用低秩近似：

$$
K_{\mathrm{global}}(x,y)
\approx
\sum_{r=1}^{R}
a_r(x,h(x))
b_r(y,h(y)).
$$

于是：

$$
\mathcal{K}_{\mathrm{global}}h(x)
=
\sum_{r=1}^{R}
a_r(x,h(x))
\int
b_r(y,h(y))
Vh(y)\,dy.
$$

若每个位置访问 \(k\) 个局部点，全局秩为 \(R\)，复杂度约为：

$$
O(Nk+NR).
$$

其中 \(R\) 是数值近似秩，不代表语义实体数量。

## 8.2 多尺度数值表示

隐场可以使用多尺度基函数近似：

$$
h_m(x)
=
\sum_{\ell=0}^{L}
\sum_r
c_{\ell r}^{m}
\phi_{\ell r}^{m}(x).
$$

基函数可以采用：

- Fourier basis；
- wavelet basis；
- learned basis；
- local kernel basis。

多尺度只是一种数值表示方式，不预设"低层是字符、高层是句子"或"低层是纹理、高层是对象"。

---

# 9. 指令与条件输入

用户问题、系统提示和任务说明本身也作为文本场输入：

$$
h_{\mathrm{instruction}}^0(x).
$$

它们与其他文本、图像、音频或视频输入通过同样的跨模态算子交互。

模型不需要额外定义：

- task token；
- instruction embedding；
- 固定 prompt slot；
- classification token。

任务由输入场的内容和整体动力学共同决定。

---

# 10. 输出场构造

主干计算完成后得到：

$$
\mathcal{H}(T)
=
\{h_m(\cdot,T)\}_{m=1}^{M}.
$$

模型不能先将其平均池化为一个向量再输出，而应定义输出域 \(\Omega_o\) 和输出算子：

$$
\mathcal{R}_o:
\prod_m\mathcal{H}_m
\rightarrow
\mathcal{H}_o.
$$

输出场为：

$$
z_o(u)
=
b_o(u)
+
\sum_m
\int_{\Omega_m}
K_{m\rightarrow o}^{\mathrm{read}}
\left(
u,x,h_m(x,T)
\right)
V_{m\rightarrow o}h_m(x,T)\,dx.
$$

其中：

$$
z_o:\Omega_o\rightarrow\mathbb{R}^{d_o}.
$$

不同任务通过选择不同的输出域和边界解码器完成。

---

# 11. 不同任务的输出方式

## 11.1 分类

每个标签的分数由完整函数场上的泛函计算：

$$
s_k
=
\sum_m
\int_{\Omega_m}
\rho_{k,m}
\left(
x,h_m(x,T)
\right)\,dx.
$$

最终：

$$
p(y=k)=\operatorname{softmax}(s)_k.
$$

分类头读取整个隐场，但不会先压缩成一个固定全局 embedding。

## 11.2 图像分割与定位

输出域为：

$$
\Omega_o=[0,1]^2.
$$

在任意坐标 \((x,y)\) 上查询：

$$
p(c\mid x,y)
=
D_{\mathrm{seg}}
\left(
z_o(x,y)
\right).
$$

因此，模型可以在不同分辨率下输出分割结果。

## 11.3 图像生成

先构造图像输出隐场：

$$
z_{\mathrm{img}}:
[0,1]^2\rightarrow\mathbb{R}^{d_o}.
$$

再使用条件流生成：

$$
\frac{\partial z_{\mathrm{img}}(u,t)}{\partial t}
=
v_{\theta,\mathrm{img}}
\left(
z_{\mathrm{img}}(\cdot,t),
u,t,
\mathcal{H}(T)
\right).
$$

从噪声场开始：

$$
z_{\mathrm{img}}(\cdot,0)\sim p_0,
$$

演化得到：

$$
z_{\mathrm{img}}(\cdot,1).
$$

最终通过坐标解码器生成像素：

$$
RGB(x,y)
=
D_{\mathrm{img}}
\left(
z_{\mathrm{img}}(x,y,1)
\right).
$$

图像分辨率只决定最终查询坐标数量，不改变主干状态定义。

## 11.4 音频与视频生成

音频输出：

$$
a(t)
=
D_{\mathrm{audio}}
\left(
z_{\mathrm{audio}}(t)
\right).
$$

视频输出：

$$
RGB(x,y,t)
=
D_{\mathrm{video}}
\left(
z_{\mathrm{video}}(x,y,t)
\right).
$$

模型可以改变音频采样率、视频帧率和图像分辨率，而不改变主干参数。

---

# 12. 文本输出

文本最终必须落到离散字符，因此不能在物理上彻底消除输出离散性。合理目标是将离散化推迟到最后的边界接口。

## 12.1 输出长度

从完整隐场预测输出长度分布：

$$
p(L\mid\mathcal{H}(T)),
$$

或者预测连续终止函数：

$$
S(u)
=
P(L\geq u\mid\mathcal{H}(T)).
$$

由此确定输出域：

$$
\Omega_{\mathrm{text}}^o=[0,L].
$$

## 12.2 连续文本隐场

构造输出文本场：

$$
z_t:
[0,L]\rightarrow\mathbb{R}^{d_o}.
$$

采用条件流动力学：

$$
\frac{\partial z_t(u,s)}{\partial s}
=
v_{\theta,t}
\left(
u,
z_t(\cdot,s),
\mathcal{H}(T),
s
\right).
$$

输出位置之间通过输出域上的非局部算子联合交互，不需要逐 token 自回归地形成语义。

## 12.3 byte 边界解码

最终只在 byte 坐标上查询：

$$
\ell_j
=
D_{\mathrm{byte}}
\left(
z_t(j)
\right)
\in\mathbb{R}^{256}.
$$

然后：

$$
b_j
=
\arg\max_c \ell_{j,c}.
$$

byte 序列再被解析为 UTF-8 文本。

此时：

- 主干不以 token 为计算单位；
- 推理过程不要求 next-token recurrence；
- 输出离散化仅存在于最后的字符接口；
- 所有位置的语义状态在连续输出场中联合形成。

## 12.4 第一版的工程折中

第一版可采用：

$$
\text{continuous backbone}
\rightarrow
\text{text output field}
\rightarrow
\text{small byte transducer}.
$$

小型 byte transducer 可以暂时自回归，专门负责：

- 精确拼写；
- 数字；
- 代码；
- 标点；
- UTF-8 合法性；
- 终止符。

此时 token 或 byte 仅是输出编码协议，不再承担主干推理。

---

# 13. 训练目标

总训练目标可以写成：

$$
\mathcal{L}
=
\mathcal{L}_{\mathrm{task}}
+
\lambda_d\mathcal{L}_{\mathrm{disc}}
+
\lambda_m\mathcal{L}_{\mathrm{mask}}
+
\lambda_c\mathcal{L}_{\mathrm{cross}}
+
\lambda_g\mathcal{L}_{\mathrm{gen}}
+
\lambda_s\mathcal{L}_{\mathrm{stable}}.
$$

## 13.1 离散不变性

对同一个底层输入构造不同采样版本：

$$
X_m^{G_1},
\qquad
X_m^{G_2}.
$$

例如：

- 不同图像分辨率；
- 不同音频采样率；
- 不同视频帧率；
- 不同文本采样方式；
- 不同求积点数量和位置。

在共同探测坐标集合 \(Q\) 上约束：

$$
\mathcal{L}_{\mathrm{disc}}
=
\frac{1}{|Q|}
\sum_{x\in Q}
\left\|
h_{m,G_1}(x)
-
h_{m,G_2}(x)
\right\|^2.
$$

该损失要求模型学习底层函数，而不是依赖某一种离散方式。

## 13.2 连续区域遮挡

不遮挡 token，而是在原生坐标域中采样随机区域：

$$
M\subset\Omega_m.
$$

要求模型根据可见区域恢复：

$$
\widehat{X}_m|_M
=
D_m
\left(
\mathcal{H}(\Omega_m\setminus M)
\right).
$$

文字遮挡连续字符区间，图像遮挡任意区域，音频遮挡时间段，视频遮挡时空区域。

## 13.3 跨模态条件预测

例如从图像预测文本场：

$$
\widehat{h}_t
=
\mathcal{P}_{i\rightarrow t}(h_i),
$$

从文本和局部图像预测完整图像场：

$$
\widehat{h}_i
=
\mathcal{P}_{t\rightarrow i}
(h_t,h_i^{\mathrm{visible}}).
$$

函数级损失为：

$$
\mathcal{L}_{\mathrm{cross}}
=
\int_{\Omega_t}
\|
\widehat{h}_t(x)-h_t(x)
\|^2dx
+
\int_{\Omega_i}
\|
\widehat{h}_i(y)-h_i(y)
\|^2dy.
$$

全局 embedding 对比学习可以作为辅助目标，但不应成为主要监督。

## 13.4 生成目标

使用 flow matching 训练输出场的条件向量场：

$$
\mathcal{L}_{\mathrm{flow}}
=
\mathbb{E}_{t,z_t}
\left[
\|
v_\theta(z_t,t,\mathcal{H})
-
u_t(z_t)
\|^2
\right].
$$

文字、图像、音频和视频可以使用不同的输出流，但共享主干函数状态。

## 13.5 数值稳定性

约束不同积分精度下的输出一致性：

$$
\mathcal{L}_{\mathrm{stable}}
=
\left\|
F_{\theta,N}(X)
-
F_{\theta,2N}(X)
\right\|^2.
$$

还可以约束：

- 核函数范数；
- 算子 Lipschitz 常数；
- 隐场高频能量；
- ODE 求解残差；
- 长深度动力学稳定性。

---

# 14. 推荐训练阶段

## 阶段一：单模态连续提升

分别训练：

- raw byte \(\rightarrow\) text field；
- pixel samples \(\rightarrow\) image field；
- waveform samples \(\rightarrow\) audio field；
- video samples \(\rightarrow\) video field。

主要目标：

- 任意坐标重建；
- 多分辨率一致性；
- 连续区域补全；
- 高频细节保持。

## 阶段二：单模态算子预训练

在每种模态内部训练：

- 去噪；
- 连续区域补全；
- 未来区域预测；
- 多尺度重建；
- 不同采样网格下的一致性。

## 阶段三：多模态耦合

加入：

- 图像条件文本场预测；
- 文本条件图像场预测；
- 音频—文本条件预测；
- 视频—音频时空耦合；
- 跨模态区域补全。

## 阶段四：指令与任务训练

训练：

- VQA；
- 图文问答；
- 图像生成；
- 文本生成；
- 定位与分割；
- 音频理解；
- 任意模态到任意模态转换。

---

# 15. 第一版可落地系统

完整的文字—图像—音频—视频生成模型训练成本过高，第一版应优先验证"模型是否真正摆脱固定 token/patch"。

## 15.1 输入

- UTF-8 raw bytes；
- RGB 原始像素；
- 不使用 BPE；
- 不使用固定视觉 patch；
- 训练时随机改变图像分辨率和求积节点。

## 15.2 提升层

- 一维文本 lifting kernel；
- 二维图像 lifting kernel；
- Fourier 或 SIREN 类坐标编码；
- 局部紧支撑核；
- 随机求积坐标。

## 15.3 主干

使用 8–12 个 residual operator blocks。每个 block 包含：

1. pointwise channel mixer；
2. local integral operator；
3. global low-rank integral operator；
4. text-to-image cross operator；
5. image-to-text cross operator；
6. residual update。

第一版暂不使用自适应 ODE solver，以减少训练不稳定性。残差块可以解释为连续动力学的显式 Euler 离散。

## 15.4 输出任务

优先选择能够验证函数场价值的任务：

- 图文检索；
- VQA；
- referring expression grounding；
- 图像分割；
- OCR 与文档理解；
- image captioning。

其中：

- 检索使用场到场匹配泛函；
- VQA 使用完整隐场读取头；
- grounding 与分割输出二维函数场；
- caption 使用连续文本输出场和小型 byte decoder。

---

# 16. 完整数据流

```text
UTF-8 bytes + 1D coordinates
              │
              ▼
    Vector-valued observation measure
              │
              ▼
      Continuous text lifting
              │
              ▼
         Text field h_t(x)
              │
              │              RGB samples + 2D coordinates
              │                          │
              │                          ▼
              │              Vector-valued observation measure
              │                          │
              │                          ▼
              │                Continuous image lifting
              │                          │
              │                          ▼
              │                 Image field h_i(x,y)
              │                          │
              └──────────────┬───────────┘
                             ▼
          Coupled multimodal operator dynamics
          ┌──────────────────────────────────┐
          │ pointwise channel transformation │
          │ local integral operators         │
          │ global low-rank operators        │
          │ cross-domain integral operators  │
          │ residual / continuous-depth flow │
          └──────────────────────────────────┘
                             │
                             ▼
              Final multimodal field state
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
         Text output      Image output    Task output
            field            field         functional
              │              │              │
              ▼              ▼              ▼
         Byte decoder   Coordinate decoder  Labels / scores
              │              │
              ▼              ▼
            Text           Image / mask
```

---

# 17. 关键理论性质

## 17.1 Discretization invariance

对于同一个底层连续信号 \(x\)，在两个离散网格 \(G_1,G_2\) 下：

$$
\left\|
\mathcal{I}_{G_1\rightarrow G_2}
F_\theta(P_{G_1}x)
-
F_\theta(P_{G_2}x)
\right\|
\leq\epsilon.
$$

其中：

- \(P_G\) 表示在网格 \(G\) 上采样；
- \(\mathcal{I}\) 表示不同网格之间的插值；
- \(F_\theta\) 的参数与网格无关。

实验上应验证：

- 不同文本编码粒度下输出稳定；
- 不同图像分辨率下输出稳定；
- 不同音频采样率下输出稳定；
- 不同求积点数量下输出收敛；
- 训练时未见分辨率下仍然可以推理。

## 17.2 Operator consistency

若求积节点数 \(N\) 增加，则数值输出应收敛到同一个连续算子：

$$
F_{\theta,N}(x)
\rightarrow
F_\theta(x),
\qquad
N\rightarrow\infty.
$$

## 17.3 Modality-native equivariance

坐标变换不应被错误地解释为语义变化。对于适当的变换群 \(g\)：

$$
F_\theta(T_gx)
\approx
T_gF_\theta(x).
$$

例如：

- 图像平移对应图像场平移；
- 音频时间平移对应音频场平移；
- 文本长度归一化不应改变顺序语义；
- 视频时间采样变化不应改变事件内容。

---

# 18. 与现有架构的本质区别

| 架构 | 基本状态 | 语义单位 | 跨模态交互 |
|---|---|---|---|
| Transformer | 有限 token 矩阵 | token / patch | 离散 attention |
| Perceiver | 固定 latent array | latent slots | cross-attention |
| Slot-based model | 对象槽集合 | 显式对象假设 | slot interaction |
| 单一 embedding | 单个向量 | 全局摘要 | 向量相似度 |
| 本架构 | 原生域上的函数集合 | 完全涌现 | 函数空间积分算子 |

该方案不是将 token 换成更大的连续 embedding，也不是将 token 换成动态实体，而是取消"模型必须围绕有限语义单元组织状态"这一前提。

---

# 19. 主要风险

## 19.1 文本的精确离散结构

文本包含拼写、数字、代码、符号和严格顺序。纯连续场可能产生语义正确但表面形式错误的问题。

解决方向：

- 保留 byte-level 边界解码器；
- 加入可逆文本重建目标；
- 对数字、代码和引用加入精确复制通道；
- 将输出离散性限制在最终接口，而不是强行消除。

## 19.2 连续积分计算成本

高分辨率图像和长文本上的完整积分成本很高。

解决方向：

- 局部紧支撑核；
- 低秩全局核；
- Fourier 或 wavelet 算子；
- Monte Carlo quadrature；
- adaptive mesh refinement；
- 多尺度数值求解。

## 19.3 模型重新退化为隐式 token

即使架构采用连续场，固定求积节点、固定基函数索引或固定 latent coefficient 也可能重新承担 token 的角色。

需要通过以下实验检验：

- 随机改变求积节点；
- 测试未见节点数量；
- 测试未见分辨率；
- 对基函数数量进行外推；
- 检查节点置换或重采样后的稳定性。

## 19.4 训练稳定性

连续算子、跨模态积分和生成流同时训练可能产生：

- 核函数爆炸；
- 高频噪声；
- ODE stiffness；
- 跨模态场坍缩；
- 输出场不可解码。

第一版应使用有限 residual operator blocks，逐步引入连续深度和 flow-based 输出。

---

# 20. 研究贡献的潜在表述

一个完整论文可以围绕以下三点展开：

1. **Token-independent multimodal representation**  
   将文本、图像和其他模态表示为原生连续域上的函数，不使用固定 token、patch、slot 或统一 latent sequence。

2. **Coupled multimodal neural operator**  
   通过模态内和跨模态积分算子学习函数空间中的非局部交互，使语义结构作为动力学模式自行涌现。

3. **Discretization-invariant learning and decoding**  
   将离散性限制在输入观测和输出接口，通过多分辨率训练、随机求积和一致性目标，使模型对采样方式、分辨率和积分精度保持稳定。

---

# 21. 一句话总结

> 将大模型从"有限 token 序列上的 Transformer"改写为"原生多模态函数场之间的耦合神经算子"：输入是坐标化观测，主干是连续场动力学，语义结构由表征学习涌现，输出先形成函数场，最终只在接口处离散为文字、像素或音频采样。
