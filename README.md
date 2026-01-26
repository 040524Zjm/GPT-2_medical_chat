# GPT-2 医疗健康咨询聊天机器人

基于 GPT-2 模型构建的医疗健康咨询聊天机器人，使用 Flask 提供 Web 界面。本项目实现了从数据预处理、模型训练到部署的完整流程，支持多轮对话和上下文理解。

**数据处理**：格式转换、张量转换，封装 DataSet 与 DataLoader 对象，适配模型输入规范；
**设计模型训练策略**，完成模型Train、Validate全流程，使用Top-P发散式生成、温度调节，对话history，加入惩罚系数，Warmup学习率调节，使用驱动云算力平台；
**人机交互实现**：开发模型预测模块，使用Flask框架开发API接口，实现机器人上线应用。

## ✨ 功能特点

- 🤖 **基于 GPT-2 模型**：使用 GPT-2 架构进行对话生成
- 💬 **多轮对话支持**：支持历史对话上下文理解（默认保留最近 3 轮）
- 🏥 **医疗健康场景**：专注于医疗健康咨询领域的对话生成
- 🌐 **Web 界面**：提供友好的 Flask Web 交互界面
- 📱 **命令行交互**：支持命令行模式进行快速测试
- 🎯 **智能采样**：使用 Top-K 和 Top-P (Nucleus) 采样策略
- 🔄 **重复惩罚**：内置重复惩罚机制，减少重复生成
- 🚀 **GPU 加速**：支持 CUDA 加速训练和推理

## 📋 环境要求

- **Python**: 3.7+
- **PyTorch**: 1.8+（推荐使用 CUDA 版本以支持 GPU 加速）
- **CUDA**: 11.0+（可选，用于 GPU 加速）
- **内存**: 建议 8GB+ RAM
- **显存**: 如果使用 GPU，建议 4GB+ VRAM

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone <your-repo-url>
cd GPT2_Chatbot
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

**注意**：如果需要使用 GPU 加速，请安装 CUDA 版本的 PyTorch：

- 访问 [PyTorch 官网](https://pytorch.org/get-started/previous-versions/) 查看对应版本的安装命令
- 或使用以下命令（根据你的 CUDA 版本调整，以 CUDA 11.8 为例）：
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. 下载模型文件

由于模型文件较大，无法直接包含在 Git 仓库中。你需要：

1. **下载预训练模型**：
   - 将训练好的模型文件放置在 `save_model/epoch97/` 目录下
   - 模型目录应包含：
     - `config.json` - 模型配置文件
     - `pytorch_model.bin` 或 `model.safetensors` - 模型权重文件

2. **模型文件结构**：
```
save_model/
└── epoch97/
    ├── config.json
    └── pytorch_model.bin  # 或 model.safetensors
```

**提示**：如果模型文件在其他位置，可以：
- 使用 Git LFS 上传大文件
- 上传到云存储（如 Google Drive、百度网盘等）并提供下载链接
- 在 Releases 中提供模型文件下载

### 4. 验证文件结构

确保以下文件存在：
- ✅ `vocab/vocab.txt` - 词汇表文件（已包含在仓库中）
- ✅ `config/config.json` - 模型配置文件（已包含在仓库中）
- ✅ `save_model/epoch97/` - 模型文件目录（需要下载）

## 💻 使用方法

### Web 界面运行

启动 Flask Web 应用：

```bash
python app.py
```

然后在浏览器中访问 `http://localhost:5000`

**注意**：默认运行在调试模式（`debug=True`），生产环境请修改 `app.py` 中的配置。

### 命令行交互运行

使用命令行模式进行交互：

```bash
python interact.py
```

输入 `quit` 或 `exit` 退出程序。

## 📁 项目结构

```
GPT2_Chatbot/
├── app.py                 # Flask Web 应用入口
├── flask_predict.py       # Flask 预测模块（模型加载和推理）
├── interact.py            # 命令行交互脚本
├── train.py              # 模型训练脚本
├── parameter_config.py   # 参数配置文件
├── function_tools.py     # 工具函数（损失计算、准确率计算等）
├── requirements.txt      # Python 依赖
├── config/
│   └── config.json       # GPT-2 模型配置文件
├── vocab/
│   ├── vocab.txt         # 词汇表文件
│   └── vocab2.txt        # 备用词汇表
├── templates/
│   ├── index.html        # Web 界面模板
│   └── index1.html       # 备用界面模板
├── data/                 # 数据目录
│   ├── medical_train.pkl # 训练数据（预处理后）
│   ├── medical_train.txt # 训练数据（原始文本）
│   ├── medical_valid.pkl # 验证数据（预处理后）
│   └── medical_valid.txt # 验证数据（原始文本）
├── data_preprocess/      # 数据预处理模块
│   ├── dataloader.py     # 数据加载器
│   ├── dataset.py        # 数据集类
│   └── preprocess.py     # 数据预处理脚本
└── save_model/           # 模型保存目录（需要下载模型文件）
    └── epoch97/
        ├── config.json
        └── pytorch_model.bin
```

## ⚙️ 配置说明

主要配置在 `parameter_config.py` 中，主要参数说明：

### 设备配置
- `device`: 自动检测使用 CPU 或 CUDA（GPU）

### 路径配置
- `vocab_path`: 词汇表路径（默认 `./vocab/vocab.txt`）
- `train_path`: 训练数据路径（默认 `data/medical_train.pkl`）
- `valid_path`: 验证数据路径（默认 `data/medical_valid.pkl`）
- `config_json`: 模型配置文件路径（默认 `config/config.json`）
- `save_model_path`: 模型保存路径（默认 `save_model2`）

### 生成参数
- `max_history_len`: 历史对话最大长度（默认 3），控制模型记住多少轮历史对话
- `max_len`: 生成回复的最大长度（默认 300），超过此长度会截断
- `repetition_penalty`: 重复惩罚参数（默认 10.0），值越大越能减少重复生成
- `topk`: Top-K 采样参数（默认 4），只从概率最高的 k 个词中选择
- `topp`: Top-P (Nucleus) 采样参数（默认 0.2），累积概率阈值

### 训练参数
- `batch_size`: 批次大小（默认 8）
- `epochs`: 训练轮数（默认 4）
- `lr`: 学习率（默认 2.6e-5）
- `warmup_steps`: 学习率预热步数（默认 100）
- `max_grad_norm`: 梯度裁剪阈值（默认 2.0）
- `gradient_accumulation_steps`: 梯度累积步数（默认 1）

## 🔧 数据格式

### 训练数据格式

训练数据应为文本文件，格式如下：

```
用户1: 你好
机器人1: 您好，有什么可以帮助您的吗？

用户2: 我最近有点头疼
机器人2: 头疼的原因有很多，建议您多休息，如果持续不缓解，建议就医检查。
```

- 每个对话由多轮组成，用空行分隔不同对话
- 每轮对话格式：`用户: 内容` 或 `机器人: 内容`
- 支持 Windows (`\r\n`) 和 Linux (`\n`) 换行符

### 数据预处理

运行数据预处理脚本将文本数据转换为模型可用的格式：

```bash
cd data_preprocess
python preprocess.py
```

预处理后的数据会保存为 `.pkl` 格式，包含 tokenized 的对话序列。

## 🎓 训练模型

如果需要训练自己的模型：

### 1. 准备训练数据

将训练数据按照上述格式放置在 `data/` 目录下，例如：
- `data/medical_train.txt` - 训练集
- `data/medical_valid.txt` - 验证集

### 2. 运行数据预处理

```bash
python data_preprocess/preprocess.py
```

**注意**：需要修改 `preprocess.py` 中的文件路径。

### 3. 配置训练参数

在 `parameter_config.py` 中调整训练参数：
- 修改 `train_path` 和 `valid_path` 指向你的数据文件
- 根据你的硬件配置调整 `batch_size`
- 设置 `epochs` 和 `save_model_path`

### 4. 开始训练

```bash
python train.py
```

训练过程中会：
- 自动保存每 10 个 epoch 的模型
- 保存验证集损失最低的模型到 `min_ppl_model_bj/` 目录
- 显示训练损失、准确率和学习率等信息

### 5. 使用训练好的模型

训练完成后，修改 `flask_predict.py` 或 `interact.py` 中的 `model_path` 指向你的模型目录。

## 🔍 技术细节

### 模型架构
- **基础模型**: GPT-2 (GPT2LMHeadModel)
- **分词器**: BERT Tokenizer (BertTokenizerFast)
- **特殊标记**: `[CLS]`（对话开始）、`[SEP]`（分隔符）、`[PAD]`（填充）

### 生成策略
- **Top-K 采样**: 只从概率最高的 k 个词中选择下一个词
- **Top-P (Nucleus) 采样**: 从累积概率达到 p 的最小词集合中选择
- **重复惩罚**: 对已生成的词降低其生成概率，减少重复

### 对话格式
模型输入格式：`[CLS]utterance1[SEP]utterance2[SEP]utterance3[SEP]...`

### 训练优化
- **学习率预热**: 使用线性预热策略，稳定训练初期
- **梯度裁剪**: 防止梯度爆炸
- **梯度累积**: 支持小批次训练，模拟大批次效果

## 🐛 常见问题

### Q: 运行时提示找不到模型文件？
A: 请确保已下载模型文件并放置在 `save_model/epoch97/` 目录下，或修改 `flask_predict.py` 中的 `model_path` 变量。

### Q: 如何修改模型路径？
A: 在 `flask_predict.py` 中修改 `model_path` 变量：
```python
model_path = os.path.join(BASE_DIR, 'save_model/epoch97')
```

### Q: 如何使用 GPU？
A: 
1. 确保安装了 CUDA 版本的 PyTorch
2. 代码会自动检测并使用 GPU（如果可用）
3. 可以通过 `os.environ["CUDA_VISIBLE_DEVICES"] = '0'` 指定使用的 GPU

### Q: 生成的回复质量不佳？
A: 可以尝试以下方法：
- 增加 `repetition_penalty` 减少重复（如改为 15.0）
- 调整 `topk`（如改为 8）和 `topp`（如改为 0.9）参数
- 增加 `max_history_len` 让模型记住更多上下文
- 使用更好的训练数据重新训练模型
- 增加训练轮数 `epochs`

### Q: 内存不足怎么办？
A: 
- 减小 `batch_size`
- 减小 `max_len` 和 `max_history_len`
- 使用梯度累积 (`gradient_accumulation_steps`)
- 使用 CPU 训练（虽然速度较慢）

### Q: 训练速度很慢？
A: 
- 使用 GPU 加速（安装 CUDA 版本的 PyTorch）
- 增加 `batch_size`（如果内存允许）
- 减少 `gradient_accumulation_steps`
- 使用混合精度训练（需要额外配置）

### Q: 如何修改 Web 界面端口？
A: 在 `app.py` 中修改：
```python
app.run(debug=True, port=5000)  # 修改 port 参数
```

## 📊 性能优化建议

1. **推理优化**：
   - 使用 GPU 进行推理
   - 批量处理多个请求（需要修改代码）
   - 使用模型量化减少内存占用

2. **训练优化**：
   - 使用多 GPU 训练（修改 `CUDA_VISIBLE_DEVICES`）
   - 使用混合精度训练
   - 使用梯度检查点节省显存

3. **数据优化**：
   - 预处理数据并保存为 `.pkl` 格式
   - 使用多进程数据加载（修改 `dataloader.py`）

## 📝 更新日志

### v1.0.0
- 初始版本发布
- 支持 Web 界面和命令行交互
- 支持模型训练和推理
- 实现多轮对话功能

## 📄 许可证

[添加你的许可证信息]

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

贡献指南：
1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📮 联系方式

[添加你的联系方式]

## 🙏 致谢

- [Hugging Face Transformers](https://github.com/huggingface/transformers) - 提供 GPT-2 模型实现
- [Flask](https://flask.palletsprojects.com/) - Web 框架
- [PyTorch](https://pytorch.org/) - 深度学习框架
