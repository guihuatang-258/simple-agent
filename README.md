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

带工具：

```powershell
python main.py
python main.py 北京现在几点，再算一下 23*17
```

纯聊天（不绑工具，用来对比首 token）：

```powershell
python main_chat.py
python main_chat.py 你好
```
