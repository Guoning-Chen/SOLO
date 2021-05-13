# 孙康论文

## 模型记号

`R<n>M<i><o>`

其中：

- `<n>`主干网络ResNet的深度，取50、101
- `<i>`和`<o>`语义分割分支特征图的输入输出尺寸
- 例如：R101M16-32代表ResNet为101层，语义分割分支的输入16，输出 32

## 训练参数

- 输入尺寸: 768x576
- 优化器: SGD
- batch: 4
- lr: 5e-4
- momentum: 0.9
- weight decay: 1e-4
- epoch: 36
- 其他技巧：
1. 学习率梯度衰减
2. early stopping
3. 接续训练
4. 性能提升不明显时冻结主干网络，专注训练检测和分割性能

# 训练命令行

`python tools/train.py ${CONFIG_FILE} --work_dir ${YOUR_WORK_DIR}`
例如：
`python tools/train.py configs/mask_rcnn_r101_fpn_1x.py --work_dir mask_rcnn_r101_fpn_1x`