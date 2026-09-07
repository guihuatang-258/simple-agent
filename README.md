# simple-agent

用 LangChain v1 最新 API（`create_agent`）搭的简易工具调用 Agent。

## 能力

- 模型循环 + 工具调用：`langchain.agents.create_agent`
- 多轮记忆：`InMemorySaver` + `thread_id`
- 工具调用示例：当前时间、天气（本地演示数据）、高德官方 MCP 周边门店查询
- 千问默认走 DashScope 原生 `Generation.call`（与 Dify 官方插件同一条口）
- 其它厂商仍可用 OpenAI 兼容接口

## 启动

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt -i https://pypi.org/simple
copy .env.example .env
```

国内镜像可能还没有 LangChain 1.4 稳定版，建议用上面的官方源。当前已验证：`langchain==1.4.0`、`langchain-openai==1.6.0`。

编辑 `.env`，千问填 `DASHSCOPE_API_KEY`（或沿用 `OPENAI_API_KEY`），`MODEL_PROVIDER=dashscope`。

如需根据用户地址查询周边门店，还需申请高德“Web 服务 API”类型的 Key，并在 `.env` 中设置 `AMAP_MAPS_API_KEY`。程序通过 `npx -y @amap/amap-maps-mcp-server` 启动高德官方 MCP Server，因此还需安装 Node.js 22.14 或更高版本。门店搜索支持品牌或类型关键词，例如“星巴克”“便利店”；未配置 Key 时会跳过地图工具，不影响其它能力。

其它兼容接口把 `MODEL_PROVIDER` 改成 `openai` 并填 `OPENAI_BASE_URL`。

每次用户提问默认最多执行 10 个工具、调用模型 6 次，可在 `.env` 调整：

```env
AGENT_MAX_TOOL_CALLS=10
AGENT_MAX_LOOPS=6
```

多个并行工具调用会分别计数。`AGENT_MAX_LOOPS` 统计模型调用：首次进入模型为
第 1 次，每轮工具执行结束后再次进入模型会再加 1；下一条用户消息重新计数。
达到工具上限后，额外工具会被阻止，模型可以利用已有结果继续回答；达到循环
上限后，当前运行直接结束。这两个值必须是大于等于 1 的整数。

腾讯位置服务 MCP 可与高德同时启用。在 `.env` 中设置 `TENCENT_MAPS_API_KEY`，
该 Key 需开通 WebServiceAPI 且拥有相关接口配额。腾讯通过远程 Streamable HTTP
地址 `https://mcp.map.qq.com/mcp` 接入，无需 Node.js；加载工具时即会联网握手。
参考[腾讯官方接入说明](https://lbs.qq.com/service/MCPServer/MCPServerGuide/userGuide)。

腾讯工具动态发现并以 `tencent_` 前缀注册，避免与高德工具重名。可设置
`TENCENT_MCP_FORMAT=0`（默认，语义化文本）或 `1`（原始 JSON）。Key 由程序
按官方要求放入连接 URL，请勿在日志中输出完整连接配置。
未配置某一家 Key 时跳过该服务；配置后连接失败会报错。纯聊天模式不连接地图 MCP。

```powershell
python main.py "用腾讯地图查深圳市南山区腾讯滨海大厦附近的星巴克"
```

带工具：

默认工具在 `agent.py` 的 `DEFAULT_TOOLS` 列表中选择。高德与腾讯分别为
`load_amap_store_tools`、`load_tencent_tools`，删除哪一项就不会连接对应服务。
也可以在 Python 中通过 `chat` 的 `tools` 参数指定（加载函数不要加括号）：

```python
from main import chat
from mcp_tools import load_amap_store_tools, load_tencent_tools
from tools import get_current_time

chat(tools=[get_current_time, load_amap_store_tools])  # 只选高德
# chat(tools=[get_current_time, load_tencent_tools])  # 只选腾讯
# chat(tools=[load_amap_store_tools, load_tencent_tools])  # 两者都选
# chat(tools=[])  # 不加载任何工具
```

每个 MCP 加载入口代表一组远程工具，加载后才展开为 `BaseTool` 列表传给
`create_agent(tools=...)`。直接调用 `build_agent(tools=...)` 时需传已加载的工具，
不能传加载函数。

```powershell
python main.py
python main.py 北京现在几点，再算一下 23*17
```

纯聊天（不绑工具，用来对比首 token）：

```powershell
python main_chat.py
python main_chat.py 你好
```
