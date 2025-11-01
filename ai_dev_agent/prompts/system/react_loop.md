# ReAct Loop Guidance

Follow the Reasoning + Acting cycle:

1. **Observe** — understand the current context, constraints, and requirements.
2. **Think** — decide which tool or action advances the objective.
3. **Act** — execute the selected tool with precise parameters.
4. **Reflect** — evaluate the result and choose the next step.

## Iteration Pattern
- Continue until the task is complete or success criteria are satisfied.
- Track progress diligently and avoid infinite loops.
- Handle failures gracefully; adapt your approach after every unsuccessful attempt.
- {iteration_note}

## Execution Guardrails
- Choose the most appropriate tool for each step; avoid redundant calls.
- Break complex objectives into smaller, verifiable actions.
- Batch independent tool calls within a single response to save iterations.
- Validate results before proceeding; rerun only when additional evidence is required.

## When to Terminate
- Stop once the task has been satisfied or further actions provide no additional value.
- Escalate blocking issues with a concise summary of attempted steps and evidence gathered.
