#!/bin/bash
set -e

PROJECT_DIR="$HOME/AI-System"
cd "$PROJECT_DIR"

echo "================================"
echo "AI-System v1.0.0"
echo "================================"

if ! command -v python3 >/dev/null 2>&1; then
    echo "错误：未找到 Python 3。"
    exit 1
fi

if ! curl --silent --fail --max-time 3 \
    http://127.0.0.1:1234/v1/models >/dev/null; then
    echo "错误：LM Studio API 暂不可用。"
    echo "请启动 LM Studio，并开启本地 API 服务。"
    exit 1
fi

echo "Python：$(python3 --version)"
echo "LM Studio API：可连接"
echo "正在启动 Memory Proxy..."
echo "按 Control+C 停止服务。"
echo

exec python3 "$PROJECT_DIR/memory_proxy.py"
