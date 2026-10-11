# AI-System

AI-System 是一个本地优先的个人 AI 记忆系统，为本地大语言模型提供记忆检索、证据门控、答案校验和自动记忆写入能力。

## 一、架构

- `memory_proxy.py`：HTTP API 主入口。
- `memory/`：记忆模块及本地记忆数据。
- `memory/memory_search.py`：记忆检索。
- `memory/answer_evidence_gate.py`：证据门控。
- `memory/answer_validator.py`：答案校验。
- `memory/memory_writer_v2.py`：自动记忆写入及索引同步。
- `tests/`：自动化测试。

v1.0 首选入口为 `memory_proxy.py`。其他入口的支持状态尚待进一步确认。

## 二、运行环境

当前默认配置：

- Python：Python 3
- 本地模型服务：LM Studio
- 上游 API：`http://127.0.0.1:1234/v1/chat/completions`
- 模型：`openai/gpt-oss-20b`
- 记忆代理：`http://127.0.0.1:8766`

启动代理前，应先启动 LM Studio，并加载配置对应的模型。

## 三、启动服务

```bash
cd ~/AI-System
python3 memory_proxy.py

```
