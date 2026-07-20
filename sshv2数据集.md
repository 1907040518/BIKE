# STHV2 子集验证说明

## 1. 这次要验证什么

这次目标是先做一个低成本 sanity check：

- 使用 Something-Something V2 的小规模分层子集。
- 训练集控制在 1 万以内。
- 验证集控制在 4000 以内。
- 先把 `attribute_guided_fusion.enable` 关掉，跑 2 个 epoch。
- 如果验证精度明显回到正常水平，例如远高于当前 clean 实验的约 5%，甚至接近或超过 20%，就可以基本确认问题主要来自属性引导融合模块及其训练/验证目标不一致。

## 2. 我做了什么

我在 `gmk/BIKE/lists/sthv2/subsets/` 下创建了一个可复现的分层抽样子集。

生成文件如下：

| 文件 | 作用 | 行数 |
|---|---|---:|
| `gmk/BIKE/lists/sthv2/subsets/train_rgb_stratified_10k_seed1024.txt` | 训练子集 list | 10000 |
| `gmk/BIKE/lists/sthv2/subsets/val_rgb_stratified_4k_seed1024.txt` | 验证子集 list | 4000 |
| `gmk/BIKE/lists/sthv2/subsets/sthv2_stratified_subset_seed1024_summary.csv` | 每类原始数量、子集数量、比例对比 | 349 |

同时创建了两个配置文件：

| 配置文件 | 用途 |
|---|---|
| `gmk/BIKE/configs/sthv2/sthv2_pre_fix_B16_subset10k_val4k_fusion_off.yaml` | 推荐先跑；属性引导融合关闭，epochs=2 |
| `gmk/BIKE/configs/sthv2/sthv2_pre_fix_B16_subset10k_val4k_fusion_on.yaml` | 对照实验；属性引导融合开启，epochs=2 |

其中 `fusion_off` 配置的关键改动：

```yaml
train_list: '/home/neimedia/gmk/BIKE/lists/sthv2/subsets/train_rgb_stratified_10k_seed1024.txt'
val_list: '/home/neimedia/gmk/BIKE/lists/sthv2/subsets/val_rgb_stratified_4k_seed1024.txt'
attribute_guided_fusion:
    enable: False
solver:
    epochs: 2
```

`fusion_on` 配置除 `attribute_guided_fusion.enable: True` 外，其余保持一致，用于验证是否一开融合就复现“loss 很低但验证精度很差”的现象。

## 3. 子集是怎么抽的

抽样方式是按类别分层抽样，不是简单随机抽样。

原因：STHV2 有 174 类，类别分布并不完全均匀。如果直接随机抽 1 万训练样本、4000 验证样本，某些小类可能数量太少甚至缺失，这样验证结果对完整数据集的代表性会变差。

我采用的策略：

1. 按 label 把全量 `train_rgb.txt` 和 `val_rgb.txt` 分组。
2. 每个类别至少保留 1 个样本，保证 174 类都覆盖。
3. 剩余名额按该类别在全量 split 中的占比分配。
4. 使用固定随机种子 `1024`，保证以后重新生成能得到一致结果。
5. 抽样后按原始 list 行号排序，尽量减少 dataloader 行为变化。

抽样结果：

- 训练集：全量 168913，子集 10000，覆盖 174 类。
  - 每类最少 5 个，最多 194 个。
- 验证集：全量 24777，子集 4000，覆盖 174 类。
  - 每类最少 2 个，最多 86 个。

这种抽法比纯随机更适合快速判断模型是否“整体正常”，因为它同时保留了：

- 全部类别覆盖；
- 原始类别比例；
- 固定 seed 可复现；
- 子集足够小，训练和验证都更快。

## 4. 推荐怎么跑

先跑 fusion off：

```bash
cd /home/neimedia/gmk/BIKE
bash scripts/run_train_23.sh configs/sthv2/sthv2_pre_fix_B16_subset10k_val4k_fusion_off.yaml
```

如果显存或 GPU 数量不合适，也可以参考已有脚本 `scripts/run_train_45.sh`，用 2 张卡：

```bash
cd /home/neimedia/gmk/BIKE
bash scripts/run_train_45.sh configs/sthv2/sthv2_pre_fix_B16_subset10k_val4k_fusion_off.yaml
```

然后再跑 fusion on 做对照：

```bash
cd /home/neimedia/gmk/BIKE
bash scripts/run_train_23.sh configs/sthv2/sthv2_pre_fix_B16_subset10k_val4k_fusion_on.yaml
```

建议对比两个实验的：

- epoch 1 后的训练 loss；
- epoch 1 后的 `Prec@1` / `Prec@5`；
- loss 是否很快降到特别低；
- fusion on 是否再次出现验证精度接近 5% 的现象。

## 5. 如何解读结果

### 情况 A：fusion off 的验证精度明显正常

如果 `fusion_off` 在这个小子集上验证精度明显高于 `clean` 的 5%，例如达到十几或二十以上，而 `fusion_on` 仍很低，则说明问题基本锁定在属性引导融合。

这时不是数据集路径或标签文件出了大问题，而是训练和验证的打分方式有偏差。

### 情况 B：fusion off 也很低

如果 `fusion_off` 也只有 5% 左右，就需要继续检查：

- 子集 list 是否能正常读取视频；
- label 是否和 `sthv2_labels.csv` 对齐；
- 当前代码是否还有别的改动；
- 预训练权重是否正常加载；
- dataloader 返回格式是否和训练脚本一致。

### 情况 C：fusion off 正常，fusion on 低且 loss 很小

这是最符合当前怀疑的结果。说明 fusion 模块让训练 loss 变得容易下降，但没有形成可泛化的类别判别能力。

## 6. 为什么“训练只用 GT 文本”会导致 loss 小但验证精度低

当前训练时，代码会根据真实标签取对应文本：

```python
indices = list_id.to(device, dtype=torch.long)
text_inputs = classes[indices]
image_embedding, cls_embedding, text_embedding, logit_scale = model(images, residuals, mvs, text_inputs, return_token=True)
```

意思是：每个训练样本都直接拿到了它自己的正确类别文本。

如果属性融合开启，模型的视觉特征会被这段正确文本调制。也就是说，训练时模型看到的是：

> 视频 + 正确类别文本 → 融合后的视觉特征 → 和正确文本做对比学习

这个任务会比真正分类简单很多。因为正确答案的语义已经作为条件输入给了模型。

但验证时不是这样。验证时模型并不知道真实类别，只能对 174 个候选类别逐个打分：

```python
for cls_idx in range(n_class):
    cls_text_tokens = text_features[cls_idx:cls_idx+1].expand(b, -1, -1)
    fused_feats, _, _ = clip_model.attribute_guided_fusion(..., cls_text_tokens, ...)
    cls_sim = video_head(fused_feats, ...)
```

这意味着同一个视频，在验证时会被 174 个不同类别文本分别调制出 174 份不同的视觉特征。

问题在于：这些分数不一定可比。

正常 CLIP 分类通常是：

> 固定视频特征 V，分别和 174 个固定类别文本 T0...T173 算相似度。

这时 174 个类别分数是可比的，因为视频特征是同一个。

但属性引导融合现在更像：

> 用类别 0 文本生成视频特征 V0，再算 S0；
> 用类别 1 文本生成视频特征 V1，再算 S1；
> ...
> 用类别 173 文本生成视频特征 V173，再算 S173。

此时每个类别分数对应的视频特征都不同，分数之间不再是同一个空间下的公平比较。所以可能出现训练 loss 很低，但验证分类不准。

## 7. 如果想保留属性融合，应该怎么改思路

### 方案 1：训练也按验证方式做 174 类 logits

训练时不要只给 GT 文本，而是对所有类别都计算分数，得到 `[B, 174]` 的 logits，然后用交叉熵：

```python
logits = compute_all_class_logits(video, all_class_texts)
loss = cross_entropy(logits, gt_label)
```

这样训练和验证目标一致：训练时也要求模型在 174 类里选正确类别，而不是只做 batch 内 GT 文本配对。

优点：最严格、最一致。

缺点：如果每个类别都跑一次属性融合，计算量会明显增加。

### 方案 2：属性文本只影响文本 embedding，不改变每个候选类别下的视频特征

也可以保留属性 prompt，但不要让验证时的视频特征随候选类别变化。

也就是说：

- 视频编码只生成一份固定视频特征；
- 属性信息用于增强类别文本特征；
- 然后用固定视频特征和 174 个文本特征算相似度。

这样分数仍然可比。

形式上更像：

```text
video -> V
class text + attributes -> T0...T173
score_i = sim(V, Ti)
```

而不是：

```text
video + T0 -> V0 -> score_0
video + T1 -> V1 -> score_1
...
```

### 方案 3：属性融合只生成全局、类别无关的视频增强

如果一定要用属性引导视频融合，可以让它使用一个类别无关的全局属性上下文，而不是每个候选类别一份上下文。这样验证时每个视频仍然只生成一份视频特征。

## 8. 这次最建议先做的实验

先跑：

```bash
cd /home/neimedia/gmk/BIKE
bash scripts/run_train_23.sh configs/sthv2/sthv2_pre_fix_B16_subset10k_val4k_fusion_off.yaml
```

如果它明显好于当前 clean 的 5%，再跑：

```bash
cd /home/neimedia/gmk/BIKE
bash scripts/run_train_23.sh configs/sthv2/sthv2_pre_fix_B16_subset10k_val4k_fusion_on.yaml
```

只要两者形成明显差距，就可以不用等完整数据集训练完，先确认 fusion 的训练/验证目标不一致问题。
