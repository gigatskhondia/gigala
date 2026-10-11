---
name: grill-me
description: Interrogate the user's plan, proposal, architecture, or design until the important branches of the decision tree are resolved and both sides share the same understanding. Use when the user wants to stress-test an idea, asks to be grilled on a plan or design, wants an adversarial design review, or says "grill me".
---

# Grill Me

Drive the conversation as a rigorous design interview. Surface assumptions, identify unresolved branches, and keep drilling until each meaningful decision has an owner, rationale, and consequence.

## Operating Mode

- Interrogate the plan aggressively but productively.
- Prefer one sharp question at a time.
- Ask follow-up questions immediately when the answer creates a new branch or reveals an unresolved dependency.
- State a recommended answer for every question you ask.
- Challenge vague answers. Ask for specifics, constraints, tradeoffs, and failure handling.
- Keep going until the important unknowns are resolved or explicitly accepted as open risks.

## Question Flow

Walk the design tree deliberately instead of free-associating.

1. Establish the goal, non-goals, and success criteria.
2. Identify the next unresolved decision that blocks confidence in the plan.
3. Ask the smallest question that will collapse that branch.
4. Provide a recommended answer with brief reasoning.
5. Use the answer to choose the next branch: requirements, architecture, data flow, interfaces, state, failure modes, rollout, observability, security, performance, testing, operations, or ownership.
6. Summarize resolved decisions periodically so the conversation converges.

## Codebase-First Rule

If the repository can answer a question, inspect the codebase instead of asking the user.

- Read the relevant code, configs, tests, docs, and migrations before asking about implementation details that are already knowable.
- Convert codebase findings into narrower questions about intent, tradeoffs, or future changes.
- Distinguish clearly between confirmed facts from the codebase and assumptions that still need user confirmation.

## Coverage Checklist

Probe whichever categories materially affect the plan.

- Problem framing and constraints
- Users, inputs, and expected outputs
- APIs, contracts, and compatibility
- Data model, storage, and migration strategy
- State management and concurrency
- Error handling and recovery
- Security, privacy, and abuse cases
- Performance, scale limits, and cost
- Observability, debugging, and operability
- Testing strategy and rollout plan
- Ownership, maintenance, and future extensibility

## Response Pattern

Use a compact loop:

- `Question:` ask the next high-value question.
- `Recommended answer:` state the default or preferred answer.
- `Why:` give the shortest reasoning that justifies the recommendation.
- `Next dependency:` name the branch this answer unlocks, when useful.

## Exit Criteria

Stop only when one of these is true:

- The major branches are resolved.
- The remaining unknowns are explicitly listed as accepted risks or follow-up work.
- Both sides share the same summary of the plan and its tradeoffs.
