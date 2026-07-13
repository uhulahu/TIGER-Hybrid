## 1. B 和 C 的直接对比

这里把两种 RQ-VAE SID 对齐：

- **B**：训练时 L3 使用 Sinkhorn；导出时保留原始 nearest assignment；碰撞用 extra token。`rqvae/ckpt/Beauty/Jul-09-2026_02-06-03/sid_evaluation_extraOnly.txt`
- **C**：训练时 L3 使用 Sinkhorn；导出时再对碰撞组做 Sinkhorn 重分配；剩余碰撞用 extra token。`rqvae/ckpt/Beauty/Jul-09-2026_02-06-03/sid_evaluation.txt`

| 指标                                  | B：extra-only | C：post-hoc Sinkhorn |
| ------------------------------------- | ------------- | -------------------- |
| L1 groups                             | 56            | 56                   |
| L2 groups                             | 5673          | 5673                 |
| L3 full-prefix groups                 | 11292         | 12059                |
| excess collision items                | 809           | 42                   |
| 三层碰撞率                            | 6.69%         | 0.35%                |
| L3 max bucket                         | 19            | 3                    |
| H(L3)H(L3)H(L3)                       | 7.9863        | 7.8526               |
| H(L3∣L1,L2)H(L3\mid L1,L2)H(L3∣L1,L2) | 1.4213        | 1.5678               |
| H(L1,L2,L3)H(L1,L2,L3)H(L1,L2,L3)     | 13.4091       | 13.5556              |

L1、L2 完全不变，所有差异都来自导出阶段的 L3 重分配。这是一个非常干净的对照。

post-hoc Sinkhorn 净增加了：

12059−11292=76712059-11292=76712059−11292=767

个不同的三层 SID，基本消除了绝大部分碰撞。

但“碰撞变少”不等于“SID 变得更适合生成”。

------

# 2. 最关键的数学解释：Sinkhorn 只是把信息从 extra token 搬到了 L3

你的 catalog 一共有 12101 个 item。如果最终通过 extra token 保证每个 item 的完整 SID 唯一，那么 item-uniform 条件下，完整 SID 的总熵近似为：

H(C1,C2,C3,D)=log⁡212101≈13.562H(C_1,C_2,C_3,D)=\log_2 12101\approx 13.562H(C1,C2,C3,D)=log212101≈13.562

其中 DDD 是 extra token。

对于 B：

H(C1,C2,C3)=13.4091H(C_1,C_2,C_3)=13.4091H(C1,C2,C3)=13.4091

所以 extra token 需要承担的剩余身份信息约为：

H(D∣C1,C2,C3)≈13.562−13.4091≈0.153H(D\mid C_1,C_2,C_3) \approx 13.562-13.4091 \approx 0.153H(D∣C1,C2,C3)≈13.562−13.4091≈0.153

对于 C：

H(C1,C2,C3)=13.5556H(C_1,C_2,C_3)=13.5556H(C1,C2,C3)=13.5556

extra token 只需承担：

H(D∣C1,C2,C3)≈13.562−13.5556≈0.006H(D\mid C_1,C_2,C_3) \approx 13.562-13.5556 \approx 0.006H(D∣C1,C2,C3)≈13.562−13.5556≈0.006

看从 L2 往后总共需要区分多少信息：

### B：extra-only

H(C3,D∣C1,C2)=1.4213+0.153≈1.574H(C_3,D\mid C_1,C_2) = 1.4213+0.153 \approx 1.574H(C3,D∣C1,C2)=1.4213+0.153≈1.574

### C：post-hoc Sinkhorn

H(C3,D∣C1,C2)=1.5678+0.006≈1.574H(C_3,D\mid C_1,C_2) = 1.5678+0.006 \approx 1.574H(C3,D∣C1,C2)=1.5678+0.006≈1.574

几乎完全相同。

这说明：

> **post-hoc Sinkhorn 没有减少 L2 之后的总预测不确定性，只是把大约 0.146 bit 的身份区分信息从 extra token 提前搬到了 L3。**

这正是你所怀疑的事情。

------

# 3. 为什么“提前搬到 L3”可能反而不好？

在 B 中，职责划分是：

```
L1、L2、L3：自然量化得到的内容语义路径
extra token：量化器无法区分时，用来做 item identity 消歧
```

在 C 中则变成：

```
L1、L2：自然内容路径
L3：
  一部分是自然 residual 区分
  一部分是 post-hoc Sinkhorn 为了唯一性强制制造的区分
extra token：只解决极少数剩余碰撞
```

因此 C 的 L3 同时包含两类信号：

1. **几何支持的语义区分**：两个 residual 本来就更接近不同 code；
2. **唯一性驱动的身份区分**：两个 residual 本来最接近同一个 code，只是为了去重被迫分开。

生成模型却无法知道二者的区别，它会把所有 L3 token 都当成同一种“细粒度语义标签”学习。

相比之下，B 明确告诉模型：

> 前三层相同，说明量化意义上它们确实无法区分；最后一层只是桶内身份编号。

这种职责分离更干净。

------

# 4. 你的另一项推测也得到了支持：post-hoc Sinkhorn 弱化了层次依赖

可以算 L3 与前缀的互信息：

I((L1,L2);L3)=H(L3)−H(L3∣L1,L2)I((L1,L2);L3) = H(L3)-H(L3\mid L1,L2)I((L1,L2);L3)=H(L3)−H(L3∣L1,L2)

B：

7.9863−1.4213=6.56507.9863-1.4213=6.56507.9863−1.4213=6.5650

C：

7.8526−1.5678=6.28487.8526-1.5678=6.28487.8526−1.5678=6.2848

post-hoc Sinkhorn 后，互信息下降了约：

6.5650−6.2848=0.2802 bits6.5650-6.2848=0.2802\text{ bits}6.5650−6.2848=0.2802 bits

这意味着：

> **C 中的 L3 与前两层前缀之间的统计耦合更弱，L3 更像一个额外的独立区分标签，而不是严格依附于前缀的 residual refinement。**

这很符合“层次语义被身份标签污染”的解释。

与此同时：

H(L3∣L1,L2):1.4213→1.5678H(L3\mid L1,L2):1.4213\rightarrow1.5678H(L3∣L1,L2):1.4213→1.5678

其有效条件分支数从：

21.4213≈2.682^{1.4213}\approx2.6821.4213≈2.68

增加到：

21.5678≈2.962^{1.5678}\approx2.9621.5678≈2.96

也就是说，在给定同一个 L1L2 前缀后，模型需要在 L3 做更多选择。但增加的选择并不一定源于真实的 residual 可分性，而可能只是 post-hoc 去重。

------

# 5. 一个很有意思的现象：C 的 L3 反而没那么均匀

B：

H(L3)=7.9863H(L3)=7.9863H(L3)=7.9863

接近 8 bit 的理论最大值，说明训练期 L3 Sinkhorn 已经把全局 marginal usage 做得极其均匀。

C：

H(L3)=7.8526H(L3)=7.8526H(L3)=7.8526

反而下降了。

所以导出期 Sinkhorn虽然降低了 full SID collision，却没有让全局 L3 分布更平衡。原因很可能是：

> 它是在每个局部碰撞组内做唯一化，而不是对全体 item 做全局平衡。

于是结果是：

- L3 code 仍全部使用；
- full tuple 数变多；
- 但某些 L3 code 的全局频率更高；
- 条件分支更多；
- marginal entropy 反而下降。

因此它的作用应该准确描述为：

> **local collision resolution，而不是 global code balancing。**

# 6. Purity 的上升基本是机械结果，不代表语义变好了

B：

```
L3 Groups = 11292
Purity = 0.9850
BrandPurity = 0.9734
```

C：

```
L3 Groups = 12059
Purity = 0.9998
BrandPurity = 0.9992
```

这个提升主要来自 bucket 被拆成了 singleton：

```
B：p95 bucket size = 2，max = 19
C：p95 bucket size = 1，max = 3
```

一个 group 里只剩一个 item 时，category purity 和 brand purity天然等于 1。因此不能把 C 的 0.9998 理解为“L3 语义更纯”。

更有参考价值的是 NMI：

```
B Cat.NMI = 0.2851
C Cat.NMI = 0.2867
```

几乎没有变化。

这说明 post-hoc Sinkhorn 的主要效果是**拆分身份冲突**，而不是改善类目语义结构。

------

# 7. 训练期 Sinkhorn已经完成了它能完成的事情

B 的结果：

```
L3 utilization = 256/256
H(L3) = 7.9863 / 8
```

说明训练阶段 L3 Sinkhorn 的 global balancing 已经非常成功。L3 没有任何 codebook collapse，边际分布近乎完美均匀。

但仍然存在 6.69% 的三层碰撞，这并不矛盾：

> 边际上 256 个 L3 code 使用均匀，不代表在每个固定的 L1L2 prefix 内，每个 item 都能获得唯一 L3 code。

例如不同 prefix 可以复用同一个 L3 code，这完全合理；同一个 prefix 内也可能有若干 item 最近 residual 相同或相近，从而选中同一个 L3 code。

所以训练 Sinkhorn解决的是：

P(C3)P(C_3)P(C3)

的全局均衡。

post-hoc collision Sinkhorn解决的是：

P(C3∣C1,C2)P(C_3\mid C_1,C_2)P(C3∣C1,C2)

下的局部唯一性。

两者目标确实不同。

------

# 8. B 和 RQ-KMeans 的碰撞负担其实非常接近

这个结果还有一个非常重要的横向信息。

| 指标                     | RQ-VAE B | RQ-KMeans |
| ------------------------ | -------- | --------- |
| L3 groups                | 11292    | 11224     |
| 三层碰撞率               | 6.69%    | 7.25%     |
| 三层 joint entropy       | 13.4091  | 13.4011   |
| extra token 承担的剩余熵 | 约 0.153 | 约 0.161  |

两者的原始三层 collision burden 几乎一样。

因此，如果最终 B 的下游 TIGER 仍然落后于 RQ-KMeans，那么可以相当有力地排除：

> “RQ-KMeans 更好只是因为碰撞更少或 extra token 更容易预测。”

因为它们的 raw collision 和 extra-token 负担基本相同。

那时差异更可能来自：

- RQ-KMeans 更忠实地保留 Qwen embedding 几何；
- RQ-VAE 的 encoder 重组了原始内容空间；
- RQ-VAE 的 L1 只有 56 个有效簇，L2 路由压力过大；
- 两者前三层 label tree 对用户历史的可预测性不同。

这会是非常干净的结论。

------

# 10. 现在这个实验应当怎样解读？

B 的 SID 评估已经证明：

> **post-hoc Sinkhorn 的本质不是降低总身份预测难度，而是在固定总信息量的情况下，改变信息在 L3 与 extra token 之间的分配。**

可以把两者概括成：

### B：late identity disambiguation

[c1,c2,c3]⏟natural semantic quantization+[d]⏟identity\underbrace{[c_1,c_2,c_3]}_{\text{natural semantic quantization}} + \underbrace{[d]}_{\text{identity}}natural semantic quantization[c1,c2,c3]+identity[d]

### C：early identity injection

[c1,c2,c3′]⏟semantic + forced identity+[d]⏟rare residual collision\underbrace{[c_1,c_2,c_3']}_{\text{semantic + forced identity}} + \underbrace{[d]}_{\text{rare residual collision}}semantic + forced identity[c1,c2,c3′]+rare residual collision[d]

你此前的理论判断是成立的，而且现在有了非常明确的熵分解支持。