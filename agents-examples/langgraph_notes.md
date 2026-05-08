## LangGraph 理解&问答

LangGraph 借用了前端 React/Redux 或者后端 事件溯源（Event Sourcing） 的设计理念。

- workflow = StateGraph(State) : 一个关系图, 它只是一堆路由规则和节点定义的集合，本身不能运行。
- 引擎（Graph Engine）:
    - “运行时环境”, start on `app = workflow.compile(...)`
    - 它包含了一个事件循环 (Event Loop)、状态合并器 (Reducer) 和执行器池 (Executor Pool)，专门用来跑你画好的那张图。
- 节点（Node）：
    - 节点只负责计算，并返回一个**状态增量**（State Diff / Payload）。
    - Graph Engine 来做take return 做 Reducer合并
- Edge: a quick&simple calculation if conditional ones
- Self-pointing edge: No default one. And allow for adding for self-loop cases.

有向图内点与边的并发性：
 - 边与节点之前是严格Sequential阻塞的， which make senses。
 - say A -> B/C/D -> E, nodes B/C/D be run async. But still blocking at E on all B/C/D be finished.

State隔离级别:
- 核心机制：LangGraph 的 State 不是被“修改”的，而是通过“归约器（Reducer）”进行增量合并（Merge）的。
- State 并非天生在 app.invoke() 之间共享，它是靠持久化层拼装出来的, eg. memory_saver = MemorySaver().
- 图引擎本身是无状态 (Stateless) 的。
- 所谓的“跨 invoke 共享状态”，本质上是依靠唯一的 thread_id 在每次图开始前做了一次 DB.Load()，结束后做了一次 DB.Save()。


场景 - 多个并发请求带着相同的thread_id 涌入:
- 并发状态一致性（Concurrency State Consistency）
- 比如用户在前端疯狂点击发送，或者系统收到并发的 Webhook 回调.
- 开源版的 LangGraph 引擎本身在这方面非常“底层且纯粹”，它默认不管并发控制.

- 没特殊处理：
    - LangGraph 引擎会同时启动两个执行流。
    - both from Checkpoint V1, finished with Checkpoint V2 and Checkpoint V3, leads to 经典的“丢失更新（Lost Update）”幻读问题.
    - LangGraph Checkpointer 的底层机制：乐观并发控制 (Optimistic Concurrency Control, OCC)
    - 并发写入 (Write) 的冲突与合并, like git, but merge may failed.

- 官方怎么解决这个问题的？（生产环境的最佳实践）
    - 对相同 thread_id 的并发处理原则是：强制串行化（Strict Serialization / Actor Model）。
    - 方案 A：分布式锁 (Distributed Lock)
    - 方案 B：消息队列串行化 (Message Queue / Actor Model)
    - plus 请求防抖 (Debouncing): Drop/Reject duplicate incoming calls
