# simple-agent

面向租车客服 POC 的显式 LangGraph 工作流。

## 能力

- LLM 入口分类：网点查询、转人工、其他 FAQ、结束会话
- 多轮状态：`InMemorySaver` + `thread_id`
- 高德 Web Service 地址解析，单次请求并带进程内缓存
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
`AMAP_MAPS_API_KEY`。Graph 在进程内直接调用高德 `place/text`，不启动 MCP Server，
也不要求安装 Node.js。未配置 Key 或地图服务异常时，网点节点会询问是否转人工；
规则话术和其它流程仍可使用。连接和读取超时可分别通过
`AMAP_CONNECT_TIMEOUT`、`AMAP_READ_TIMEOUT` 调整，默认是 3 秒和 5 秒。

## 对话路由

每条用户消息都会从 Graph 的 `START` 重新进入：

```text
START -> LLM intent router
  ├─ branch_query  -> AMap Web Service -> local branch policy -> END
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

静态 Graph 可以通过独立脚本导出。默认通过 Mermaid 服务生成 PNG；Mermaid 文本
输出不需要联网：

```powershell
python visualize_graph.py
python visualize_graph.py --format mermaid
python visualize_graph.py --format mermaid -o customer-service-graph.mmd
```

终端 ASCII 模式使用 `python visualize_graph.py --format ascii`，需要额外安装
`grandalf`。

## 网点和营销数据

POC 租车网点保存在 `data/branches.json`。入口判定为网点查询后，Graph 直接调用
高德 `place/text`，根据首条结果的 `typecode` 和坐标完成地理粒度判断，再由本地
函数完成距离排序和营业判断。每次定位最多发送一次地图请求；同一进程内重复地址
会命中最多 128 条的 LRU 缓存。行政区级或更粗的地址不会直接计算距离，而是继续
询问道路、门牌或附近地标。
营销开关及审核话术保存在 `data/marketing_policy.json`。

两个示例网点尚未提供联系电话，因此对应字段为 `null`，程序不会生成虚构号码。
`request_human_handoff` 当前只生成带请求 ID 的 POC 转接事件，并且要求用户明确
选择转人工；它尚未连接真实呼叫中心或在线客服平台。

## 地图调用性能

`branch` 节点复用 `AMapWebServiceClient`，HTTP 请求在线程池中执行，不阻塞
LangGraph 事件循环；客户端按线程复用 `requests.Session` 的连接池。命令行的
执行路径会打印地图提供方、判断级别、接口耗时和缓存状态，例如：

```text
branch(source=branch_workflow, map=amap_web_service, level=poi, map_ms=185, cache=miss)
```

`mcp_tools.py` 和 `scripts/amap_tool_smoke.py` 仅保留为独立实验脚本，不参与 Graph
默认启动和网点查询链路。

高德 MCP 的三个工具可以通过独立脚本分别测试；输出包含调用参数、耗时和结果，
不会打印 API Key：

```powershell
python -m scripts.amap_tool_smoke schema
python -m scripts.amap_tool_smoke geo --address "天津南站" --city "天津"
python -m scripts.amap_tool_smoke around --location "117.050646,39.050010" --keywords "停车场" --radius 1000
python -m scripts.amap_tool_smoke detail --id "周边搜索返回的POI ID"
python -m scripts.amap_tool_smoke all --address "天津南站" --city "天津" --keywords "停车场"
```

不启动 MCP Server、直接调用高德 Web Service REST API 时，使用：

```powershell
python -m scripts.amap_rest_smoke text --keywords "北京大学" --types 141201 --region "北京市"
python -m scripts.amap_rest_smoke text --keywords "太平洋" --geocode-fallback
python -m scripts.amap_rest_smoke geo --address "天津南站" --city "天津"
python -m scripts.amap_rest_smoke around --location "117.050646,39.050010" --keywords "停车场" --radius 1000
python -m scripts.amap_rest_smoke detail --id "搜索接口返回的POI ID" --show-fields business
python -m scripts.amap_rest_smoke all --address "天津南站" --city "天津" --keywords "停车场"
```

REST 脚本同样从 `.env` 读取 `AMAP_MAPS_API_KEY`，并且不会在请求日志或错误信息中
打印 Key。`--timeout` 需要写在子命令之前，例如
`python -m scripts.amap_rest_smoke --timeout 30 text --keywords "北京大学"`。

`text` 关键词检索默认只调用一次 `place/text`。程序先对照第一条结果的省、市、区
字段，避免纯行政区名称被搜索联想成具体场所；然后从 POI 的 `typecode` 判断地理
粒度。`190101` 到 `190109` 映射国家至村组级，`1903xx`、`1904xx` 视为区级
以下的精确位置，其余带唯一 ID 和坐标的普通地点视为 POI 点位。
无法稳定判断的自然地名、城市中心等 `190xxx` 返回 `unknown`；需要高德官方
`level` 时可加 `--geocode-fallback`，仅在 `unknown` 时补调 `geocode/geo`。
级别相对市级、区级的关系使用 `broader`（范围更大）、`same` 或 `finer`
（范围更小、更精细）表示，通用判断实现在 `scripts/geo_level.py`。

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
