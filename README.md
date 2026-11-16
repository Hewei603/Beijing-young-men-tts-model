# Beijing Young Men TTS

基于 **Qwen3-TTS 1.7B** 的个人普通话男声微调模型。输入文字，直接生成固定声线的音频，不需要再提供参考录音。

This repository releases a single-speaker Qwen3-TTS fine-tune, its complete inference checkpoint, download utilities, and speech examples.

- **声线名称**：`user_speaker`
- **模型类型**：`custom_voice`，完整推理检查点，非 LoRA / adapter
- **基础模型**：[Qwen3-TTS-12Hz-1.7B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base)
- **授权**：[Apache-2.0](LICENSE)
- **权重下载**：[v1.0.0 Release](https://github.com/Hewei603/Beijing-young-men-tts-model/releases/tag/v1.0.0)

## 试听

以下都是微调模型实际生成的声音，不是训练原始录音。

| 示例 | 文本 |
|---|---|
| [约 10 秒讲解样音](examples/narration.wav) | 这样，人工就可以把时间放在需要判断的事情上，重复整理信息的工作交给工具来做。我们接着看一下实际操作。 |
| [短句](examples/short.wav) | 我们先打开工作台。 |

## 下载后怎样运行？

`git clone` 只取得代码。完整权重和语音解码器约 **4.52 GB**，保存在 GitHub Release，由下载器校验并自动组装。无需 Git LFS，也不需要 GitHub 或 Hugging Face token。

建议使用 **Python 3.12** 和独立虚拟环境。先按设备安装 PyTorch，再安装本仓库依赖。

### CPU 运行

```bash
git clone https://github.com/Hewei603/Beijing-young-men-tts-model.git
cd Beijing-young-men-tts-model
python -m venv .venv
```

激活虚拟环境：Linux 运行 `source .venv/bin/activate`，Windows PowerShell 运行 `./.venv/Scripts/Activate.ps1`。如果 PowerShell 阻止激活，可直接用 `.venv\Scripts\python.exe` 替代下面的 `python`。

```bash
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt

python download_model.py
python infer.py --mode cpu --text "我们先打开工作台。" --output output.wav
```

CPU 入口以 BF16 加载主模型，声音解码器使用 CPU FP32。速度取决于硬件，建议先用短句确认环境。

### NVIDIA 显卡运行

在独立环境中，将上面的 PyTorch 安装命令换成 CUDA 12.8 构建：

```bash
python -m pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python download_model.py

python infer.py --mode balanced --text "我们先打开工作台。" --output output.wav
```

`balanced` 使用 CPU 预载、GPU BF16 主生成模块和 CPU FP32 解码器。这沿用本模型既有的低显存运行方式，不会把全部组件一次性装进显卡。请使用支持 BF16 的 NVIDIA GPU。

### 朗读文件、设置停顿

稿件使用 UTF-8 编码。以下例子在完整句之间插入 0.4 秒静音，并按自然句分段，保持单次生成的长度可控：

```bash
python infer.py --mode balanced --text-file narration.txt --pause 0.4 --output narration.wav
```

```bash
python download_model.py --verify-only
python infer.py --help
```

更换模型保存位置时，同时设置两个入口的路径：

```bash
python download_model.py --output /path/to/model
python infer.py --model-dir /path/to/model --mode cpu --text "你好，这是语音合成示例。" --output sample.wav
```

## 资源与文件

主权重约 3.83 GB，声音 tokenizer 约 0.68 GB。下载器会将主权重的三个 Release 分片校验、拼接成标准的 `model.safetensors`，不改变原权重。

默认下载目录：

```text
model/
  config.json
  generation_config.json
  model.safetensors
  tokenizer_config.json
  vocab.json
  merges.txt
  preprocessor_config.json
  speech_tokenizer/
    config.json
    configuration.json
    model.safetensors
    preprocessor_config.json
  LICENSE
  NOTICE
```

运行还需要 PyTorch、其他依赖和生成时的内存。权重文件大小不等于运行内存。推理入口会检查可用 RAM / VRAM，资源不足时给出具体原因。先跑短句，再处理长稿。

下载中断后重新执行同一命令，下载器会复用已校验文件或继续有效断点。现有文件不匹配时会报出路径，请先确认该目录内容，再选择新的模型目录。SHA-256、文件大小和 Release 分片对应关系见 [model-manifest.json](model-manifest.json)。

## 模型来源与训练概况

| 项目 | 内容 |
|---|---|
| 基础模型修订 | `fd4b254389122332181a7c3db7f27e918eec64e3` |
| 微调方式 | 单说话人监督微调，完整 BF16 推理检查点 |
| 数据划分 | 122 条训练、22 条验证、16 条测试 |
| 训练轮数 | 3 轮，选用 epoch 3 |
| speaker | `user_speaker` |
| 配置类型 | `custom_voice` |
| 推理接口 | `generate_custom_voice` |
| 主权重 SHA-256 | `c03fa20db0b545e2546bca808753c9ce40dbfc5f7435cea8f81d7e13e2867678` |

导出时将固定说话人向量写入对应 codec embedding，省略生成固定声线时不需要的 speaker encoder。完整推理不依赖本人的原始参考录音。语音解码器和文本 tokenizer 随权重一起提供。

发布不包含原始录音、训练数据集、优化器状态或本机配置。`Beijing` 是仓库名称，不代表已经完成北京方言能力评测。当前声线主要用于普通话朗读与讲解，数字、英文和专业词的发音建议在具体稿件中检查。

更完整的说明见 [模型卡](MODEL_CARD.md)。已生成样音便于了解当前音色，不代表所有稿件的自然度、速度或口音效果。

## Python 调用

完成下载和依赖安装以后，可以通过标准 Qwen3-TTS API 使用这个检查点：

```python
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained(
    "./model",
    device_map="cpu",
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
    local_files_only=True,
)
model.model.speech_tokenizer.model.to(
    device="cpu", dtype=torch.float32
)
model.model.speech_tokenizer.device = torch.device("cpu")

with torch.inference_mode():
    wavs, sample_rate = model.generate_custom_voice(
        text="我们先打开工作台。",
        language="Chinese",
        speaker="user_speaker",
        instruct="",
        max_new_tokens=384,
    )
sf.write("output.wav", wavs[0], sample_rate)
```

需要分句、停顿与资源检查时，使用 `infer.py`。自定义脚本也应按实际音轨检查输出，不要把达到生成上限的片段当作完整句。

## 许可证与上游

本微调检查点与本仓库原创发布工具按 Apache-2.0 发布。Qwen3-TTS 的原有权利和归因保留，见 [LICENSE](LICENSE) 与 [NOTICE](NOTICE)。

- [Qwen3-TTS 上游实现](https://github.com/QwenLM/Qwen3-TTS)
- [官方微调说明](https://github.com/QwenLM/Qwen3-TTS/tree/main/finetuning)
- [基础模型](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base)
