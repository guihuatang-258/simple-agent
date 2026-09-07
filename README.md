# simple-agent

面向租车客服 POC 的显式 LangGraph 工作流。

## 能力

- LLM 入口分类：网点查询、转人工、其他 FAQ、结束会话
- 多轮状态：`InMemorySaver` + `thread_id`
- 高德官方 MCP 地址解析
- POC 网点服务：预设网点距离排序、营业判断、合规旺季话术
- 规则话术：Excel 前三个类别共 11 条，每条支持多条正则
- 受限兜底：本地召回 Top-3 候选，LLM 只选规则 ID，代码返回标准话术
- 千问默认走 DashScope 原生 `Generation.call`，其它厂商可用 OpenAI 兼容接口

## 启动

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt -i https://pypi.org/simple
copy .env.example .env
```

国内镜像可能还没有 LangChain 1.4 稳定版，建议使用官方源。当前已验证
`langchain==1.4.0`、`langchain-openai==1.6.0`。

编辑 `.env`，千问填 `DASHSCOPE_API_KEY`（或沿用 `OPENAI_API_KEY`），并设置
`MODEL_PROVIDER=dashscope`。其它兼容接口把 `MODEL_PROVIDER` 改成 `openai`，
并填写 `OPENAI_BASE_URL`。

网点查询需要高德“Web 服务 API”类型的 Key，并在 `.env` 中设置
`AMAP_MAPS_API_KEY`。程序通过 `npx -y @amap/amap-maps-mcp-server` 启动高德
官方 MCP Server，因此还需安装 Node.js 22.14 或更高版本。未配置 Key 时会跳过
地图工具，规则话术和转人工流程仍可使用。

## 对话路由

每条用户消息都会从 Graph 的 `START` 重新进入：

```text
START -> LLM intent router
  ├─ branch_query  -> branch tools -> END
  ├─ human_handoff -> handoff tool -> END
  ├─ faq           -> regex rule / bounded fallback -> END
  └─ goodbye       -> END
```

`route` 每轮先让 LLM 只判断 `branch_query`、`human_handoff`、`faq` 或 `goodbye`，
同时提取网点查询地址或转人工确认状态。入口只返回结构化 JSON，不生成客服答案。

只有 `faq` 才进入规则匹配。正则命中后不再调用回答模型，直接返回
`data/dialogue_rules.json` 中的标准口径。当前前三个类别来自“网点查询营销话术.xlsx”
的“营销话术”页：租车条件、要素查询、车况与服务。

正则未命中时，本地字符相似度和关键词检索只召回最多三条候选主题，候选答案不
发送给模型。LLM 只能返回一个候选规则 ID 或 `null`；代码校验 ID 后读取标准口径。
没有可靠匹配时，系统说明当前支持范围并询问是否转人工。

`pending_intent` 保存等待地址或等待转人工确认的跨轮状态。用户说“再见”后，
Graph 设置 `conversation_ended=true`，CLI 结束当前会话。

命令行每轮回答后会打印实际 LangGraph 执行路径，并附带路由决策、回复来源和
命中的规则 ID，例如：

```text
[执行路径] START -> route(decision=rule, rule_id=rental-process-concern) -> rule(source=regex_rule, rule_id=rental-process-concern) -> END
```

## 网点和营销数据

POC 租车网点保存在 `data/branches.json`。入口判定为网点查询后，Graph 直接调用
高德地址解析工具和本地网点推荐工具，完成 GCJ-02 坐标解析、距离排序和营业判断。
营销开关及审核话术保存在 `data/marketing_policy.json`。

两个示例网点尚未提供联系电话，因此对应字段为 `null`，程序不会生成虚构号码。
`request_human_handoff` 当前只生成带请求 ID 的 POC 转接事件，并且要求用户明确
选择转人工；它尚未连接真实呼叫中心或在线客服平台。

## 地图工具选择

默认地图服务在 `agent.py` 的 `DEFAULT_TOOLS` 中选择。高德与腾讯分别为
`load_amap_store_tools`、`load_tencent_tools`。也可以通过 `chat` 的 `tools` 参数
指定，加载函数不要加括号：

```python
from main import chat
from mcp_tools import load_amap_store_tools, load_tencent_tools

chat(tools=[load_amap_store_tools])
# chat(tools=[load_tencent_tools])
# chat(tools=[load_amap_store_tools, load_tencent_tools])
# chat(tools=[])
```

腾讯位置服务需要在 `.env` 中设置 `TENCENT_MAPS_API_KEY`。该 Key 需开通
WebServiceAPI 并具有相关接口配额。可设置 `TENCENT_MCP_FORMAT=0`（语义化文本）
或 `1`（原始 JSON）。

## 运行示例

```powershell
python main.py
python main.py "天津南站附近哪个网点最近？"
python main.py "异地还车费为什么这么贵？"
```

纯聊天模式不加载地图和客服规则，可用于比较模型首 token：

```powershell
python main_chat.py
python main_chat.py 你好
```
