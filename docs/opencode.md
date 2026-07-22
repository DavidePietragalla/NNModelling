# OpenCode workflow

NNModelling provides a project-local OpenCode team under `.opencode/agents/`.
The agents share the repository guidance in `AGENTS.md` and can load only the
skills committed under `.agents/skills/` and explicitly allowlisted in
`opencode.json`.

## One-time OpenAI setup

In OpenCode, run `/connect`, choose **OpenAI**, then choose
**ChatGPT Plus/Pro**. Credentials remain in OpenCode's user configuration and
must not be committed to this repository.

## Agent roster

| Role | Agent | Model |
| --- | --- | --- |
| Primary architect | `architect` | `openai/gpt-5.6-sol` |
| Alternative architect | `architect-deepseek` | `deepseek/deepseek-v4-pro` |
| Frontend implementer | `frontend-openai` | `openai/gpt-5.6-luna` |
| Frontend implementer | `frontend-deepseek` | `deepseek/deepseek-v4-flash` |
| Backend implementer | `backend-openai` | `openai/gpt-5.6-luna` |
| Backend implementer | `backend-deepseek` | `deepseek/deepseek-v4-flash` |
| Reviewer | `reviewer-openai` | `openai/gpt-5.6-terra` |
| Reviewer | `reviewer-deepseek` | `deepseek/deepseek-v4-pro` |
| UI/UX specialist | `designer` | `deepseek/deepseek-v4-flash` |
| Read-only explorer | `explorer` | `deepseek/deepseek-v4-flash` |

Primary agents are selected with OpenCode's agent switcher. Subagents can also
be mentioned directly with `@agent-name`.

## Selecting implementers and reviewers

Tell the architect which provider to use. The choice may be global or per task:

```text
Use OpenAI implementers and the DeepSeek reviewer.
```

```text
Use frontend-openai for the UI, backend-deepseek for Python, then both reviewers.
```

If a change requires implementation and no provider was selected, the
architect must ask before delegating. It must not silently choose a default.
The reviewer choice is independent; request OpenAI, DeepSeek, or both.

Older, completed design documents can still contain `@frontend`, `@backend`,
and `@reviewer`. These are historical role labels; the architects map them to a
current agent only after receiving your provider choice.

## Agentic loop

Both architects follow the same loop:

1. Frame the request, constraints, acceptance criteria, and affected areas.
2. Inspect the relevant code and load applicable skills.
3. Produce a proportionate implementation plan or design document.
4. Confirm any missing implementer/reviewer choice with the user.
5. Delegate one bounded task at a time to the selected implementer.
6. Have the implementer inspect, edit, test, and report evidence.
7. Delegate review to the selected reviewer or reviewers.
8. Route actionable findings back to the appropriate implementer.
9. Repeat implementation, validation, and review until the acceptance criteria
   pass or a genuine blocker requires user input.
10. Report the completed outcome, validation evidence, and remaining risks.

The loop does not authorize destructive actions, external writes, credential
changes, or work outside the user's requested scope.

## Skills

OpenCode discovers the repository skills directly from `.agents/skills/`; no
copy under `.opencode/skills/` is required. `opencode.json` denies every skill
by default and then allows exactly these project skills:

- `chrome-direct`
- `code-reviewer`
- `nnmodelling-mcp`
- `python-code-style`
- `python-testing-patterns`
- `pytorch-patterns`
- `svelte-core-bestpractices`
- `svelte5-best-practices`
- `typescript-advanced-types`
- `vitest`

Agents load a skill only when its description matches the task. Browser-backed
work must follow the routing rules in `AGENTS.md`.
