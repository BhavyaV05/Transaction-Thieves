"""LangGraph orchestration of the LLM agent loops (rewrite wave 2). The harness, gate, accountant, banks, DSL, back-test and
fusion are unchanged and are called from graph nodes; LLM calls still go through BudgetedClient."""
