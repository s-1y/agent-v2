# Agent V2

这是一套会自己分工的编程代理。你给它一句任务和一个工作区目录，它会在这个目录里读代码、改代码。同一次运行里的节点共用这一次的会话；下一次运行即使还用同一个工作区，也会另开一套会话，不会接着上次的节点往下聊。工作区里的项目文件每次都照常可读、可改。

## 准备 Python 环境

用 Python 3.12。下面按 conda 来配，环境名用 `meta-agent`。

```bash
conda create -n meta-agent python=3.12 -y
conda activate meta-agent
pip install "openhands-sdk==1.45.0" "openhands-tools==1.45.0"
```

这两个包装的是对话、终端、文件编辑和浏览器。版本要成对，都用 1.45.0。

自己干活的阶段会打开本机 Chromium 看页面。Ubuntu 22.04 上安装：

```bash
sudo apt install -y chromium-browser
```

装好后命令 `chromium-browser` 或 `chromium` 至少有一个能找到。没有浏览器时，这一阶段的浏览器工具起不来，读文件和改代码仍然可以做。

## 配置模型

在仓库根目录复制一份环境文件：

```bash
cp .env.example .env
```

`.env` 不要提交。程序启动时会读它，已经写在系统环境里的同名变量不会被覆盖。需要这三项：

| 变量 | 作用 |
| --- | --- |
| `LLM_API_KEY` | 模型密钥。没写时会再看 `DASHSCOPE_API_KEY`。两个都空会直接报错。 |
| `LLM_MODEL` | 模型名。不写时默认是 `gpt-5.5`。 |
| `LLM_BASE_URL` | 兼容接口地址。用官方默认地址时可以留空。 |

走阿里云百炼时，`.env` 写成这样（密钥换成你自己的）：

```bash
LLM_API_KEY=sk-...
LLM_MODEL=openai/qwen3.8-flash
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_DROP_PARAMS=true
```

`LLM_DROP_PARAMS=true` 是给这种兼容接口用的。客户端会带上一些对方不认识的参数，设上之后这些参数会被丢掉，请求才能通过。

## 跑一次

在仓库根目录执行，这样 `python -m Agents.v2` 才能找到包：

```bash
conda activate meta-agent
python -m Agents.v2.run --brief "把这个小游戏的界面改得能玩"
```

不写 `--workspace` 时，工作区是仓库里的 `workspace/v2/<运行号>/`。要把改动落到已有项目上，把那个目录指过去：

```bash
python -m Agents.v2.run --brief "..." --workspace /path/to/project --viz
```

`--viz` 会在本机打开工作树页面，地址是 `http://127.0.0.1:8765/`。这台机器如果是远程的，用 Cursor 的 Ports 面板把 8765 转到自己的电脑再打开。

跑完之后：

- 日志和树在 `logging/v2/<运行号>.log`、`.tree.json`
- 这次运行的节点会话在工作区的 `.meta_agent/runs/<运行号>/conversations/`
