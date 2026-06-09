## Why single agent to multiple
解决痛点：
 - 上下文窗口与记忆 限制。
 - 长链条任务下，串行依赖，

优势：
 - 功能模块化分离-解耦
 - agent 任务复杂的下降，提高准确率
 - context 隔离，缓解单agent的上下文窗口限制

Q： 为什么选择 星形结构？？？

## how the guardrail works?
## how the permission works?
- two layer for fix:
    - one is in the HITL, accept the fix, then in AAP, the person with right permission to click the run button in AAP.
- mcp readonly
    - xxx