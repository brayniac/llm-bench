# Synthetic Multi-Turn Conversations

## Problem

Current synthetic mode generates independent single-turn prompts. There's no way to benchmark multi-turn conversational workloads with synthetic data — users must load ShareGPT-format files for that.

## Solution

Add `turns` and `turn_prompt_tokens` fields to `input.synthetic` config. When `turns > 1`, synthetic workloads become multi-turn conversations. Each turn's user prompt is generated synthetically with controlled token counts. The LLM generates real assistant responses.

## Design

### Config Changes

```toml
[input.synthetic]
prompt_tokens = 128
turns = 3                    # NEW: fixed turns per conversation
turn_prompt_tokens = 128     # NEW: token count per subsequent turn
```

| Field | Type | Default | Validation |
|-------|------|---------|------------|
| `turns` | `usize` | 1 | Must be >= 1 |
| `turn_prompt_tokens` | `Option<usize>` | `prompt_tokens` | Must be > 0 if set |

### SyntheticConfig Changes (`config.rs`)

Add two fields to `SyntheticConfig`:
- `pub turns: usize`
- `pub turn_prompt_tokens: Option<usize>`

In `Config::validate()`:
- Validate `turns >= 1`
- Default `turn_prompt_tokens` to `prompt_tokens` when unset

### SyntheticDataGenerator Changes (`synthetic.rs`)

Change `generate_workload()` to:
1. Sample `turns` synthetic prompts (first turn uses `prompt_tokens`, subsequent turns use `turn_prompt_tokens`)
2. Each turn gets a deterministic prefix `[synthetic-{turn_index}]` based on workload index + turn number
3. When `turns <= 1`, behavior is unchanged (returns `Workload::SingleTurn`)
4. When `turns > 1`, returns `Workload::MultiTurn(Conversation)` with all user_turns pre-populated

### Execution Flow

No changes to execution. Existing `execute_conversation()` path handles the work:
1. First turn: system prompt + synthetic user prompt → LLM streams response → appended to message history
2. Second turn: synthetic user prompt + message history → LLM streams response → appended
3. Continues until all turns complete or conversation fails

### Test Coverage

- Update existing `test_generate_workload` test to verify single-turn still works
- Add test for multi-turn generation: verify correct number of turns, correct token counts per turn
- Add test for deterministic generation with same seed
