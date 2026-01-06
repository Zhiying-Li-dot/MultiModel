# PVTT Benchmark Dataset

Product Video Template Transfer 评估数据集。

## 数据结构

```
pvtt-benchmark/
├── videos/          # 模板视频
│   ├── jewelry/     # 首饰类
│   ├── home/        # 家居类
│   ├── beauty/      # 美妆类
│   ├── fashion/     # 服饰类
│   └── electronics/ # 电子类
│
├── images/          # 产品图片
│   └── {category}/
│       ├── {ID}_source.jpg  # 模板中产品
│       └── {ID}_target.jpg  # 目标产品
│
└── annotations/
    ├── metadata.json        # 主标注文件
    └── shots/               # 镜头级标注
        └── {ID}_shots.json
```

## 样本 ID 规则

| 前缀 | 品类 |
|------|------|
| JEW | 首饰 |
| HOME | 家居 |
| BEA | 美妆 |
| FASH | 服饰 |
| ELEC | 电子 |

## 镜头类型

| 类型 | 难度 |
|------|------|
| pure_product | Easy |
| product_closeup | Medium |
| interaction | Hard |
| wearing | Expert |

## 使用

详见 `docs/dataset-design.md`。
