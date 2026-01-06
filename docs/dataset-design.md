# PVTT 数据集设计

## 一、数据集概述

### 目标
构建一个用于 Product Video Template Transfer 任务的评估数据集：
- **规模**：100+ 样本
- **用途**：Benchmark 评估，不用于训练
- **格式**：(Template Video, Product Image, Annotations)

### 核心原则
1. **聚焦可行场景**：优先收集"纯产品镜头"类型视频
2. **覆盖多品类**：首饰、家居、美妆等
3. **标注详尽**：支持多维度评估

---

## 二、数据结构

### 目录结构
```
data/
├── pvtt-benchmark/
│   ├── videos/                    # 模板视频
│   │   ├── jewelry/               # 首饰类
│   │   │   ├── JEW001.mp4
│   │   │   └── ...
│   │   ├── home/                  # 家居类
│   │   └── beauty/                # 美妆类
│   │
│   ├── images/                    # 产品图片
│   │   ├── jewelry/
│   │   │   ├── JEW001_source.jpg  # 模板视频中的产品图
│   │   │   ├── JEW001_target.jpg  # 目标产品图
│   │   │   └── ...
│   │   └── ...
│   │
│   ├── annotations/               # 标注文件
│   │   ├── metadata.json          # 主标注文件
│   │   └── shots/                 # 镜头级标注
│   │       ├── JEW001_shots.json
│   │       └── ...
│   │
│   └── README.md                  # 数据集说明
```

### 样本 ID 命名规则
- 格式：`{CATEGORY}{NUMBER}`
- 示例：`JEW001`, `HOME012`, `BEA003`

| 前缀 | 品类 |
|------|------|
| JEW | 首饰 (Jewelry) |
| HOME | 家居 (Home Decor) |
| BEA | 美妆 (Beauty) |
| FASH | 服饰 (Fashion) |
| ELEC | 电子 (Electronics) |

---

## 三、标注 Schema

### 主标注文件 (metadata.json)

```json
{
  "version": "1.0",
  "created": "2026-01-05",
  "samples": [
    {
      "id": "JEW001",
      "category": "jewelry",
      "subcategory": "bracelet",

      "template_video": {
        "path": "videos/jewelry/JEW001.mp4",
        "duration_sec": 6.2,
        "resolution": "832x480",
        "fps": 24,
        "total_frames": 149,
        "source": "etsy",
        "source_url": "https://..."
      },

      "source_product": {
        "image_path": "images/jewelry/JEW001_source.jpg",
        "description": "Two personalized couple bracelets, one silver and one black",
        "attributes": {
          "material": "silver/leather",
          "color": ["silver", "black"],
          "style": "minimalist"
        }
      },

      "target_product": {
        "image_path": "images/jewelry/JEW001_target.jpg",
        "description": "A gold charm necklace with colorful gemstone pendants",
        "attributes": {
          "material": "gold",
          "color": ["gold", "multicolor"],
          "style": "elegant"
        }
      },

      "prompts": {
        "source": "Two personalized couple bracelets, one silver and one black, placed on a purple silk fabric with decorative stones.",
        "target": "A gold charm necklace with colorful gemstone pendants placed on a purple silk fabric with decorative stones."
      },

      "difficulty": "medium",
      "shot_type": "pure_product",
      "notes": "Single continuous shot, no camera cuts"
    }
  ]
}
```

### 镜头级标注 (shots/*.json)

```json
{
  "video_id": "JEW002",
  "total_shots": 4,
  "shots": [
    {
      "shot_id": 1,
      "start_frame": 0,
      "end_frame": 148,
      "start_time": 0.0,
      "end_time": 6.2,
      "type": "pure_product",
      "description": "Bracelets on purple silk fabric",
      "camera_motion": "static_with_slight_push",
      "contains_human": false,
      "product_visibility": "full",
      "difficulty": "easy"
    },
    {
      "shot_id": 2,
      "start_frame": 149,
      "end_frame": 185,
      "type": "product_closeup",
      "description": "Bracelet detail closeup",
      "contains_human": false,
      "difficulty": "easy"
    }
  ]
}
```

---

## 四、镜头类型定义

| 类型 | 英文 | 特点 | 难度 | 数量目标 |
|------|------|------|------|----------|
| 纯产品镜头 | pure_product | 只有产品，无人物 | ⭐ Easy | 40+ |
| 产品特写 | product_closeup | 手持/展示细节 | ⭐⭐ Medium | 30+ |
| 交互镜头 | interaction | 人手操作产品 | ⭐⭐⭐ Hard | 20+ |
| 佩戴镜头 | wearing | 产品在人身上 | ⭐⭐⭐⭐ Expert | 10+ |

### Phase 1 优先级（聚焦 Easy/Medium）

- **优先收集**：纯产品镜头、产品特写
- **暂缓收集**：交互镜头、佩戴镜头

---

## 五、产品品类

### 品类规划

| 品类 | 子类 | 目标数量 | 优先级 |
|------|------|----------|--------|
| **首饰** | 项链、手链、戒指、耳环 | 30 | P0 |
| **家居** | 枕套、装饰品、花瓶 | 25 | P0 |
| **美妆** | 口红、香水、护肤品 | 20 | P1 |
| **服饰** | 包包、鞋子、配饰 | 15 | P1 |
| **电子** | 手机壳、耳机、手表 | 10 | P2 |

### 品类选择标准

1. **视频可获取性**：平台上有足够多的产品视频
2. **产品形态多样性**：同品类内有形态差异（如项链 vs 手链）
3. **迁移挑战性**：不同形态产品间的迁移有研究价值

---

## 六、数据来源

### 主要来源

| 来源 | 类型 | 优势 | 劣势 |
|------|------|------|------|
| **Etsy** | 手工艺品电商 | 高质量产品视频，创意类多 | 规模有限 |
| **Amazon** | 综合电商 | 品类全，量大 | 视频质量参差 |
| **Alibaba/1688** | B2B 电商 | 工厂级产品视频 | 需要登录 |
| **小红书** | 社交电商 | UGC 内容，真实场景 | 版权问题 |
| **Pexels/Unsplash** | 免费素材 | 无版权问题 | 非电商风格 |

### 收集策略

1. **Etsy 为主**：首饰、家居类
2. **Amazon 补充**：美妆、电子类
3. **免费素材**：通用产品展示视频

### 版权考虑

- 数据集仅用于**学术研究**
- 标注来源 URL，支持追溯
- 不公开原始视频，只提供处理后的版本或 URL

---

## 七、标注流程

### 自动化部分

1. **视频元数据**：ffprobe 自动提取
2. **镜头检测**：PySceneDetect 自动分割
3. **产品描述**：VLM 自动生成初始描述

### 人工标注

1. **校验镜头边界**：确认自动检测结果
2. **分类镜头类型**：pure_product / closeup / interaction / wearing
3. **撰写 prompts**：source/target prompt
4. **评估难度**：easy / medium / hard / expert

### 质量控制

- 每个样本由 2 人独立标注
- 不一致时由第 3 人仲裁
- 定期随机抽查

---

## 八、评估指标

### 自动指标

| 指标 | 描述 | 工具 |
|------|------|------|
| CLIP-I | 生成视频与目标产品图的 CLIP 相似度 | CLIP |
| CLIP-T | 生成视频与目标 prompt 的 CLIP 相似度 | CLIP |
| FVD | Fréchet Video Distance | I3D |
| SSIM | 结构相似度（与模板） | scikit-image |
| 光流一致性 | 运动模式保持度 | RAFT |

### 人工评估

| 维度 | 描述 | 评分 |
|------|------|------|
| 产品保真度 | 生成的产品是否像目标产品 | 1-5 |
| 风格一致性 | 是否保持模板的视觉风格 | 1-5 |
| 运动一致性 | 是否保持模板的运镜/节奏 | 1-5 |
| 时序平滑度 | 是否有闪烁/抖动 | 1-5 |
| 整体质量 | 作为产品视频的可用性 | 1-5 |

---

## 九、时间规划

| 阶段 | 时间 | 任务 | 产出 |
|------|------|------|------|
| 设计完善 | 1月 W2 | 完善 schema，搭建工具 | 标注工具 |
| 试收集 | 1月 W3-4 | 收集 20 个样本，验证流程 | 20 样本 |
| 正式收集 | 2月 | 扩展到 60 样本 | 60 样本 |
| 补充收集 | 3月 | 补充到 100+ 样本 | 100+ 样本 |
| 质量检查 | 4月 W1-2 | 全面质量审核 | 最终数据集 |

---

## 十、下一步行动

- [ ] 创建 `data/pvtt-benchmark/` 目录结构
- [ ] 编写视频元数据提取脚本
- [ ] 编写镜头自动检测脚本
- [ ] 设计标注界面（或使用现有工具如 Label Studio）
- [ ] 从 Etsy 收集首批 10 个首饰类视频
