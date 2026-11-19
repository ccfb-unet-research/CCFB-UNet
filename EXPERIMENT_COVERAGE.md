# 审稿风险—实验—产物对应表

| 待解决问题 | 本增量包中的解决方案 | 论文中应使用的产物 |
|---|---|---|
| 缺少专用 DG 基线 | 同一 PVT/数据/增强/损失/优化器/14k 预算下增加 RobustNet-PVT、SAN-SAW-PVT、BlindNet-PVT、Rein-PVT、DPMFormer-PVT；前两项为经典方法，后三项为 2024–2025 方法 | `evaluation/summary/summary_mean_sd.tsv`、`macro_seed.tsv` |
| DG 方法的官方实现与主干不一致 | 主表明确标注 “PVT adaptation (our reimplementation)”；VFM 官方仓库另设 exploratory track，不与公平主表混算 | `official_track/README.md`、`official_repositories/COMMITS.tsv` |
| ImageNet 预训练混杂 | PVT-control scratch/pretrained 与 CCFB scratch/pretrained 构成 2×2 对照；三种原始分割基线按统一 scratch 协议重训 | `fig_pretraining_control.*`、`paired_tests_holm.tsv` |
| 原 DDCNet/ResUNet++/UltraSeg 的 28k 协议不一致 | 新增三份 14k 公平配置，统一 batch、LR、增强、Dice 权重和评估规则；原 28k 结果只可作为历史结果，不用于结构因果结论 | `fair_ddcnet_scratch.py` 等三份配置 |
| 单随机种子 | 所有新增实验使用 0/42/1234/2026/3407 五个固定种子 | `macro_seed.tsv`、`fig_multiseed_macro_dice.*` |
| 缺少显著性/效应量 | 配对符号翻转检验、95% CI、Cohen dz、Holm 多重校正；另有种子→图像两级 bootstrap | `paired_tests_holm.tsv`、`hierarchical_bootstrap.tsv` |
| 只报告区域重叠指标 | 新增 Dice、IoU、Precision、Recall、HD95、ASSD、Boundary-F；明示空掩膜规则 | 每个数据集的 `per_image.tsv` 和 `summary.json` |
| 测试集重复样本与可能泄漏 | SHA-256 精确审计、pHash 近重复候选、测试集重复簇；同时报告原始官方分数与 SHA 簇等权分数 | `data_audit/*`、`summary_mean_sd.tsv` 两种 protocol |
| FD/MGD/GLS 贡献不完整 | 在已有 PVT-control 与 full 的基础上补齐 6 个组合，形成完整 2^3 因子消融 | 六份 `ablation_*.py` 及其五域输出 |
| ColonDB 机制解释缺少证据 | 按病灶面积、对比度、边界复杂度分层，比较 full、GLS-only 与 PVT control，避免只凭可视化下结论 | `difficulty_summary.tsv`、`fig_colondb_lesion_size_mechanism.*` |
| 缺少效率与可复现性 | 参数量、可训练参数、FLOPs、batch-1 延迟/FPS、峰值显存；保存 checkpoint SHA-256 与嵌入配置 | `efficiency/*`、`provenance/*` |
| 缺少外部验证 | 提供 PolypGen/EndoCV 等外部图像/标签目录的无训练测试入口；没有数据时不生成或虚构结果 | `external_datasets.example.tsv`、`run_external_evaluation.sh` |
| 缺少定性失败分析 | `SAVE_MASKS=1` 时为 seed 0 保存预测掩膜，可用于挑选 ColonDB/ETIS 困难、失败及成功案例 | `evaluation/seed_0/*/*/masks/` |

## 论文报告边界

1. 五个新增 DG 方法是针对本项目二分类息肉分割与 PVT 控制主干的可复现适配实现，不是原作者城市街景代码直接运行的结果。
2. 表名建议写为 **Matched-backbone DG reimplementations**，方法名写成 `RobustNet-PVT (reimpl.)` 等。
3. Rein 与 DPMFormer 的官方版本依赖更大的视觉/视觉语言基础模型，若运行官方 track，必须单列 **foundation-model exploratory comparison**，不得与参数量和预训练资源不同的公平主表混合排名。
4. 只有置信区间、校正后 p 值和效应量共同支持时才使用“显著优于”；否则使用“取得更高均值”或“表现相当”。
5. 外部验证没有真实数据和运行结果前，不得在论文中声称已经完成。
