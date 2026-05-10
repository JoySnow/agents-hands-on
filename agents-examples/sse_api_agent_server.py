import json
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

# 这里导入我们之前写好的超级 Agent 状态机
# 假设你在 agent.py 里定义了 app = workflow.compile()
from agent_react_router_with_rag_subgraph import app as agent_app

# ==========================================
# 1. 初始化 FastAPI 实例
# ==========================================
api = FastAPI(title="Enterprise Agentic RAG API", version="1.0")

# 🌟 新增跨域配置
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 生产环境请务必改为真实的前端域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 定义前端请求的 JSON 结构体
class ChatRequest(BaseModel):
    message: str
    session_id: str = "default_user_123" # 生产中用来隔离不同用户的记忆

# ==========================================
# 2. 核心黑魔法：SSE 异步生成器 (Async Generator)
# ==========================================
# async def sse_event_generator(user_message: str):
#     """
#     这是一个极度硬核的流式拦截器。
#     它会监听 LangGraph 在底层跑时的每一次“心跳”，并将其转化为 SSE 规范的字符串推送给前端。
#     """
#     # 告诉前端：引擎已经启动了
#     yield f"data: {json.dumps({'type': 'status', 'content': '🚀 Agent 正在启动...'})}\n\n"

#     # 使用 V2 版本的事件流 API，开始异步执行有向无环图
#     async_stream = agent_app.astream_events(
#         {"messages": [HumanMessage(content=user_message)]},
#         version="v2"
#     )

#     async for event in async_stream:
#         kind = event["event"]

#         # --- 拦截器 1：捕获大模型生成的每一个字 (Token Streaming) ---
#         if kind == "on_chat_model_stream":
#             chunk = event["data"]["chunk"]
#             if chunk.content:
#                 # SSE 协议极其严格：必须是 data: 你的数据 \n\n 格式
#                 yield f"data: {json.dumps({'type': 'token', 'content': chunk.content})}\n\n"

#         # --- 拦截器 2：捕获工具调用开始 (让前端 UI 显示“正在查询知识库...”) ---
#         elif kind == "on_tool_start":
#             tool_name = event["name"]
#             yield f"data: {json.dumps({'type': 'tool_start', 'tool': tool_name})}\n\n"

#         # --- 拦截器 3：捕获工具调用结束 ---
#         elif kind == "on_tool_end":
#             tool_name = event["name"]
#             yield f"data: {json.dumps({'type': 'tool_end', 'tool': tool_name})}\n\n"

#     # 引擎完全走到 END 节点，告诉前端连接可以断开了
#     yield f"data: {json.dumps({'type': 'done'})}\n\n"

async def sse_event_generator(user_message: str):
    # 修复 1：状态提示加入 ensure_ascii=False
    yield f"data: {json.dumps({'type': 'status', 'content': '🚀 Agent 正在启动...'}, ensure_ascii=False)}\n\n"

    async_stream = agent_app.astream_events(
        {"messages": [HumanMessage(content=user_message)]},
        version="v2"
    )
    # 我们关心的、需要在前端展示进度条的核心业务节点名称
    # 请确保这些名字与你 workflow.add_node() 里定义的名字一致！
    target_nodes = ["rag_tool_node", "rewrite", "retriever", "database_tool_node", "rag_search", "db_query"]

    async for event in async_stream:
        kind = event["event"]
        name = event["name"] # 获取当前正在执行的组件名称

        # --- 拦截器 1：大模型吐字 ---
        if kind == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            if chunk.content:
                # 修复 2：Token 推送加入 ensure_ascii=False
                yield f"data: {json.dumps({'type': 'token', 'content': chunk.content}, ensure_ascii=False)}\n\n" #.encode("utf-8")

        # --- 拦截器 2：捕获 官方 Tool 的启动 ---
        elif kind in set(["on_tool_start", "on_tool_end"]):
            yield f"data: {json.dumps({'type': 'tool_start', 'tool': name}, ensure_ascii=False)}\n\n" #.encode("utf-8")

        # --- 🌟 拦截器 3：捕获 自定义 Node 的启动 (全视之眼) ---
        elif kind in set(["on_chain_start", "on_chain_end"]):
            # 过滤掉杂音，只推送我们关心的核心节点
            if name in target_nodes:
                yield f"data: {json.dumps({'type': 'tool_start', 'tool': name}, ensure_ascii=False)}\n\n" #.encode("utf-8")

    yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"


# ==========================================
# 3. 暴露 RESTful 路由
# ==========================================
@api.post("/v1/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    print(f"📥 收到新请求，用户会话: {request.session_id}")

    # 返回 StreamingResponse，将媒体类型强制设置为 text/event-stream
    return StreamingResponse(
        sse_event_generator(request.message),
        media_type="text/event-stream"  # ; charset=utf-8"
    )

if __name__ == "__main__":
    # 启动 ASGI 服务器
    print("🌐 启动 API 网关: http://127.0.0.1:8000")
    uvicorn.run(api, host="127.0.0.1", port=8000)


# Test SSE API with curl
"""
$ curl -X POST "http://127.0.0.1:8000/v1/chat/stream" \
     -H "Content-Type: application/json" \
     -d '{"message": "帮我查一下订单 20260509 的状态. 公司的报销找谁签字?"}'
data: {"type": "status", "content": "🚀 Agent 正在启动..."}

data: {"type": "tool_start", "tool": "LangGraph"}

data: {"type": "tool_start", "tool": "llm_orchestrator"}

data: {"type": "tool_start", "tool": "tool_calling_router"}

data: {"type": "tool_start", "tool": "tool_calling_router"}

data: {"type": "tool_start", "tool": "llm_orchestrator"}

data: {"type": "tool_start", "tool": "database_tool_node"}

data: {"type": "tool_start", "tool": "rag_tool_node"}

data: {"type": "tool_start", "tool": "LangGraph"}

data: {"type": "tool_start", "tool": "rewrite"}

data: {"type": "tool_start", "tool": "database_tool_node"}

data: {"type": "tool_start", "tool": "rewrite"}

data: {"type": "tool_start", "tool": "retrieve"}

data: {"type": "tool_start", "tool": "retrieve"}

data: {"type": "tool_start", "tool": "LangGraph"}

data: {"type": "tool_start", "tool": "rag_tool_node"}

data: {"type": "tool_start", "tool": "llm_orchestrator"}

data: {"type": "token", "content": "订单"}

data: {"type": "token", "content": " 2"}

data: {"type": "token", "content": "0"}

data: {"type": "token", "content": "2"}

data: {"type": "token", "content": "6"}

data: {"type": "token", "content": "0"}

data: {"type": "token", "content": "5"}

data: {"type": "token", "content": "0"}

data: {"type": "token", "content": "9"}

data: {"type": "token", "content": " 当前"}

data: {"type": "token", "content": "状态"}

data: {"type": "token", "content": "为"}

data: {"type": "token", "content": "**"}

data: {"type": "token", "content": "正在"}

data: {"type": "token", "content": "派送"}

data: {"type": "token", "content": "中"}

data: {"type": "token", "content": "**。"}

data: {"type": "token", "content": "\n\n关于"}

data: {"type": "token", "content": "公司"}

data: {"type": "token", "content": "报销"}

data: {"type": "token", "content": "签字"}

data: {"type": "token", "content": "事宜"}

data: {"type": "token", "content": "，"}

data: {"type": "token", "content": "根据"}

data: {"type": "token", "content": "规章制度"}

data: {"type": "token", "content": "，"}

data: {"type": "token", "content": "报销"}

data: {"type": "token", "content": "需要"}

data: {"type": "token", "content": "**"}

data: {"type": "token", "content": "财务总监"}

data: {"type": "token", "content": "**"}

data: {"type": "token", "content": "签字"}

data: {"type": "token", "content": "。"}

data: {"type": "tool_start", "tool": "tool_calling_router"}

data: {"type": "tool_start", "tool": "tool_calling_router"}

data: {"type": "tool_start", "tool": "llm_orchestrator"}

data: {"type": "tool_start", "tool": "LangGraph"}

data: {"type": "done"}
"""