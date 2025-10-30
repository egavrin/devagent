# ReAct Loop Pattern

## Overview

The ReAct (Reasoning and Acting) loop is a structured approach for problem-solving that alternates between reasoning about the task and taking concrete actions.

## Loop Structure

```
1. OBSERVE: Analyze current state and context
2. THINK: Reason about what needs to be done
3. ACT: Take specific action(s)
4. REFLECT: Evaluate results and adjust approach
5. REPEAT: Continue until task is complete
```

## Implementation Pattern

### 1. OBSERVE Phase
- Read relevant files and documentation
- Understand existing patterns and conventions
- Identify constraints and requirements
- Check current test coverage and CI status

### 2. THINK Phase
- Break down the problem into steps
- Identify dependencies and risks
- Plan the implementation approach
- Consider edge cases and error conditions

### 3. ACT Phase
- Execute planned actions (read, write, edit, test)
- Make incremental, testable changes
- Run tests frequently
- Commit logical units of work

### 4. REFLECT Phase
- Check if tests pass
- Verify requirements are met
- Assess code quality
- Identify what worked and what didn't

### 5. REPEAT Decision
- If complete: Summarize results
- If incomplete: Adjust approach and continue
- If blocked: Identify blockers and seek clarification

## Self-Check Questions

At each phase, ask yourself:

**OBSERVE:**
- Do I understand the current codebase structure?
- Have I identified all relevant files and dependencies?

**THINK:**
- Is my approach the simplest solution?
- Have I considered all edge cases?
- Will this maintain backward compatibility?

**ACT:**
- Am I making the minimal necessary changes?
- Are my changes well-tested?
- Is the code readable and maintainable?

**REFLECT:**
- Did the tests pass?
- Is the code cleaner than before?
- Did I meet the requirements?

## Iteration Limits

- Aim to complete tasks within 10-15 iterations
- If stuck after 5 iterations on the same issue, try a different approach
- If no progress after 15 iterations, summarize blockers and seek guidance

## Example Loop

```
Iteration 1:
OBSERVE: Read test file to understand requirements
THINK: Tests expect a Calculator class with add() method
ACT: Create Calculator class with add() implementation
REFLECT: Tests pass, basic requirement met

Iteration 2:
OBSERVE: Check for additional test cases
THINK: Need to handle edge cases (negative numbers, overflow)
ACT: Add validation and error handling
REFLECT: All tests pass, implementation complete
```
