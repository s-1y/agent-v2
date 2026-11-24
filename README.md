# Agent

给一句任务，它会在一个目录里读代码、改代码。入口是 `Agents.v3`。

grace 上可以直接读 `/home/syy/work/agent-v2`。先复制到自己的目录再配环境，不要改共享的那一份。那里没有你的密钥。

## 环境

用 Python 3.12。在仓库根目录：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install "openhands-sdk==1.45.0" "openhands-tools==1.45.0"
pip install "browser-use==0.13.14"
pip install "mcp==2.3.0" "anyio>=4.14" "click>=8.4.2"
```

Windows 上激活环境用 `.venv\Scripts\activate`。

这几行会让 pip 打出「版本对不上」的提示，装完可以继续用，不要改回它建议的版本。`openhands-tools` 1.45 还在用旧的浏览器接口，所以 `browser-use` 钉在 `0.13.14`。这个旧包会把 `mcp`、`anyio`、`click` 拉低，OpenHands 又会导入失败，所以最后一行把它们装回去。冲突只在这个虚拟环境里。

看网页的阶段需要 Playwright 的 Chrome，不要用 snap 里的 `chromium-browser`。没有浏览器时，读文件和改代码仍然可以做。

```bash
python -m playwright install chromium
```

## 密钥

```bash
cp .env.example .env
```

`.env` 不要提交。程序只在变量还没设过的时候读它。至少写 `LLM_API_KEY`。用千问、百炼时写成：

```bash
LLM_API_KEY=sk-...
LLM_MODEL=openai/qwen3.8-flash
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

走官方 OpenAI 时只填密钥，模型不写则默认 `gpt-5.5`，地址留空。

## 跑

在仓库根目录、环境已激活时：

```bash
python -m Agents.v3.run --brief "要做什么"
```

`--brief` 是那句任务。`--repeat` 是做几版，默认 1。`--workspace` 指到已有项目；不写就用 `workspace/v3/<运行号>/`。`--viz` 打开 `http://127.0.0.1:8765/`。程序在远程机器上时，把 8765 转到自己的电脑再打开。
