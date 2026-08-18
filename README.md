# teacher-uv-dl-py39-20260815

PyTorch 深度学习教程项目（Python 3.9 + uv 环境管理）。

## 项目结构

```
├── ch03/     # Tensor 基础教程（notebook）
├── ch07/     # 卷积神经网络（CNN）教程
├── common/   # 公共工具代码
├── data/     # 数据集（大文件已忽略，不入库）
├── main.py   # PyTorch 环境检查脚本
└── pyproject.toml
```

## 环境准备

使用 [uv](https://github.com/astral-sh/uv) 管理依赖：

```bash
uv sync
```

## 运行环境检查

```bash
uv run python main.py
```

## 备注

- `data/fashion-mnist_train.csv`、`data/train.csv` 等大文件因超过 GitHub 单文件 100MB 限制，未纳入版本控制，需要时请自行下载。
