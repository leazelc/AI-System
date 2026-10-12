# AI-System

本地优先的个人 AI 记忆系统，为本地大语言模型提供记忆检索、证据门控、答案校验和自动记忆写入。

## 架构

- `memory_proxy.py`：HTTP API 代理入口。
- `memory/`：记忆管理、检索、校验及相关模块。
- `memory/memory_search.py`：记忆检索。
- `memory/answer_evidence_gate.py`：证据门控。
- `memory/answer_validator.py`：答案校验。
- `memory/memory_writer_v2.py`：记忆写入及索引同步。
- `tests/`：自动化测试。

## 默认环境

- 系统：macOS
- Python：Python 3
- 本地模型服务：LM Studio
- 上游 API：`http://127.0.0.1:1234/v1/chat/completions`
- 默认模型：`openai/gpt-oss-20b`
- 记忆代理：`http://127.0.0.1:8766`

以上为默认配置，其他操作系统、模型和模型服务的兼容性尚待验证。

## 启动

1. 启动 LM Studio，加载模型并启动本地 API 服务。
2. 在终端执行：

   `cd ~/AI-System`

   `python3 memory_proxy.py`

3. 根据终端输出确认代理启动成功。
4. 按 `Control+C` 停止代理。

## API 测试

在代理运行时，打开另一个终端执行：

`curl -sS http://127.0.0.1:8766/v1/models`

## 故障排查

- 代理无法启动：检查 Python 环境、项目路径和终端报错。
- 无法连接模型：检查 LM Studio、模型加载状态及上游 API 地址。
- 记忆异常：检查日志；确认原因前不要删除记忆文件或索引。

## 隐私与项目状态

项目以本地运行和本地数据管理为设计方向。实际数据存储、日志内容和网络访问范围以当前实现及配置为准。项目正在进行 v1.0 可靠性与交付准备，支持范围以实际验证结果为准。
