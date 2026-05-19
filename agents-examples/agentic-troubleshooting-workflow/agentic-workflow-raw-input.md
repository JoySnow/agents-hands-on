Design a following agentic workflow project with me:
this workflow got multi-agents with one agent as orchestrator with following sub-agents:
 - sub-agent "rhel agent" with rhel mcp, for rhel system actions;
 - sub-agent "satellite agent" with satellite mcp, for satellite actions;
 - sub-agent "insights agent" with insights mcp, for insights service actions;
 - sub-agent "aap agent" with aap mcp, for aap actions;
 - sub-agent "slack agent" for communicate with slack for message read and send;

The agentic workflow is started from a prod alert triggering, take the following use case as an example:
 - on a rhel system rhel-A, it is experience an high CPU usage issue of certain thread say "bagger";
 - some system monitor trigger an alert on this cpu high, a msg send to slack on this;
 - on the alert slack message, here is the agentic workflow comes in to action;
 - agent receive the slack notification, then do troubleshooting process based on the "rhel agent", "satellite agent", "insights agent", "aap agent", etc. analysis.
 - then it make a final fix proposal based on the analysis;
 - the agent requires a HITL with the fix proposal, though the slack message, and provide a accpet button and a reject button in slack conversation;
 - on the human confirm with accept button, the aap agent apply the fix run with aap job template;
 - then agent report back the fix result to slack;
 - if the human confirm with reject button and any human input hint(if any), the agent can make further analysis based on the human feedback and agent's self adjust, then raise another new fix proposal;

---

Now understand the workflow, and let's fill in the tech details for each detail part of this agentic workflow; adjust or enhance the workflow if needed.

