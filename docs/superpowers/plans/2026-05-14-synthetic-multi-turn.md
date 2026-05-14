# Synthetic Multi-Turn Conversations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `turns` and `turn_prompt_tokens` config fields to generate multi-turn synthetic conversations where the LLM provides real responses and the harness generates fake user prompts per turn.

**Architecture:** Extend `SyntheticConfig` with turn fields. Modify `SyntheticDataGenerator::generate_workload()` to return `Workload::MultiTurn(Conversation)` when `turns > 1`, pre-populating `Conversation.user_turns` with synthetic prompts. Reuse the existing `execute_conversation()` path in `benchmark.rs` — no changes needed there.

**Tech Stack:** Rust, fake-rs (already a dependency), tiktoken-rs (via tokenizer module), serde/TOML for config.

---

### Task 1: Add `turns` and `turn_prompt_tokens` fields to `SyntheticConfig`

**Files:**
- Modify: `src/config.rs:96-123` (SyntheticConfig struct)
- Modify: `src/config.rs:483-544` (validate function, synthetic block)

Add two fields to `SyntheticConfig`:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyntheticConfig {
    pub prompt_tokens: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens_stdev: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens_min: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens_max: Option<usize>,
    #[serde(default = "default_add_prefix")]
    pub add_prefix: bool,
    #[serde(default = "default_common_prefix_sample_ratio")]
    pub common_prefix_sample_ratio: f64,
    #[serde(default = "default_common_prefix_tokens")]
    pub common_prefix_tokens: usize,
    // NEW FIELDS
    #[serde(default = "default_turns")]
    pub turns: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub turn_prompt_tokens: Option<usize>,
}

fn default_turns() -> usize {
    1
}
```

Add validation in the existing synthetic validation block (around line 543, after the common_prefix_tokens check):

```rust
// Validate turn fields
if synthetic.turns == 0 {
    anyhow::bail!("input.synthetic.turns must be greater than 0");
}
if let Some(turn_tokens) = synthetic.turn_prompt_tokens
    && turn_tokens == 0
{
    anyhow::bail!("input.synthetic.turn_prompt_tokens must be greater than 0 if specified");
}
if synthetic.turn_prompt_tokens.is_some()
    && synthetic.turn_prompt_tokens.unwrap() > synthetic.prompt_tokens_max.unwrap_or(synthetic.prompt_tokens)
{
    anyhow::bail!(
        "input.synthetic.turn_prompt_tokens ({}) cannot exceed prompt_tokens_max ({})",
        synthetic.turn_prompt_tokens.unwrap(),
        synthetic.prompt_tokens_max.unwrap_or(synthetic.prompt_tokens),
    );
}
```

- [ ] **Step 1: Add fields and defaults to `SyntheticConfig`**

Edit `src/config.rs` to add the two new fields and the `default_turns()` function after the existing default functions (around line 94, after `default_common_prefix_tokens`).

- [ ] **Step 2: Add validation for turn fields**

Add the validation block in `Config::validate()` after the existing common_prefix_tokens validation (around line 543).

- [ ] **Step 3: Run tests to ensure nothing broke**

Run: `cargo test --lib config::tests`
Expected: All existing tests pass (the new fields have defaults so existing configs still parse).

- [ ] **Step 4: Commit**

```bash
git add src/config.rs
git commit -m "feat(config): add turns and turn_prompt_tokens to SyntheticConfig

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 2: Generate multi-turn conversations in `SyntheticDataGenerator`

**Files:**
- Modify: `src/synthetic.rs:137-156` (`generate_workload` method)
- Modify: `src/synthetic.rs:166-249` (`generate_prompt` method — add turn_idx parameter)

Changes:

1. Change `generate_workload` signature to accept `turns` parameter and generate a `Conversation` when turns > 1:

Replace the existing `generate_workload` method (lines 138-156):

```rust
/// Generate a single workload with sampled token counts.
pub fn generate_workload(&mut self, index: usize, max_tokens: Option<u32>, turns: usize) -> Workload {
    // Determine turn_prompt_tokens for each turn
    let first_turn_tokens = self.prompt_dist.sample();
    let subsequent_turn_tokens: Vec<usize> = (0..turns.saturating_sub(1))
        .map(|_| self.prompt_dist.sample())
        .collect();

    let mut user_turns = Vec::with_capacity(turns);

    // First turn
    let prompt_tokens = first_turn_tokens;
    let use_common_prefix = if self.common_prefix_ratio > 0.0 {
        let ratio_index = (index as f64) / (1.0 / self.common_prefix_ratio);
        ratio_index.fract() < self.common_prefix_ratio
    } else {
        false
    };
    user_turns.push(self.generate_prompt_for_turn(prompt_tokens, index, 0, use_common_prefix, max_tokens));

    // Subsequent turns
    for (turn_idx, &turn_token_count) in subsequent_turn_tokens.iter().enumerate() {
        user_turns.push(self.generate_prompt_for_turn(
            turn_token_count, index, turn_idx + 1, false, max_tokens,
        ));
    }

    if user_turns.len() == 1 {
        Workload::SingleTurn(Prompt {
            prompt: user_turns.pop().unwrap(),
            max_tokens,
        })
    } else {
        Workload::MultiTurn(Conversation {
            system_prompt: None,
            user_turns,
            max_tokens,
        })
    }
}
```

2. Add a new `generate_prompt_for_turn` method that accepts a `turn_idx` parameter for prefix generation:

```rust
/// Generate prompt text for a specific turn with deterministic prefix.
fn generate_prompt_for_turn(
    &self,
    token_count: usize,
    index: usize,
    turn_idx: usize,
    use_common_prefix: bool,
    max_tokens: Option<u32>,
) -> String {
    // Prefix uses workload index + turn index for deterministic uniqueness
    let prefix = if use_common_prefix {
        if let Some(ref common_prefix) = self.common_prefix_text {
            if !common_prefix.is_empty() {
                common_prefix.clone()
            } else if self.add_prefix {
                format!("[synthetic-{}-t{}] ", index, turn_idx)
            } else {
                String::new()
            }
        } else if self.add_prefix {
            format!("[synthetic-{}-t{}] ", index, turn_idx)
        } else {
            String::new()
        }
    } else if self.add_prefix {
        format!("[synthetic-{}-t{}] ", index, turn_idx)
    } else {
        String::new()
    };

    // Calculate how many tokens we need after the prefix
    let prefix_tokens = self.tokenizer.count_tokens(&prefix);
    let remaining_tokens = if prefix_tokens >= token_count {
        return prefix[..self.truncate_to_tokens(&prefix, token_count)].to_string();
    } else {
        token_count - prefix_tokens
    };

    const AVG_CHARS_PER_TOKEN: usize = 5;
    const MARGIN_OF_SAFETY: f64 = 1.5;
    const MAX_ATTEMPTS: usize = 3;

    let mut attempts = 0;
    loop {
        attempts += 1;
        let num_chars = ((remaining_tokens * AVG_CHARS_PER_TOKEN) as f64
            * MARGIN_OF_SAFETY
            * attempts as f64) as usize;

        let mut rng = StdRng::seed_from_u64(self.seed + (index * 1000 + turn_idx) as u64);
        let mut text = String::new();
        while text.len() < num_chars {
            let sentence: String = Sentence(5..20).fake_with_rng(&mut rng);
            text.push_str(&sentence);
            text.push(' ');
        }
        let text = text[..num_chars.min(text.len())].to_string();
        let full_text = format!("{}{}", prefix, text);
        let token_count_actual = self.tokenizer.count_tokens(&full_text);

        if token_count_actual >= token_count {
            return full_text[..self.truncate_to_tokens(&full_text, token_count)].to_string();
        }
        if attempts >= MAX_ATTEMPTS {
            warn!(
                "Failed to generate {} tokens after {} attempts (got {}), using what we have",
                token_count, MAX_ATTEMPTS, token_count_actual
            );
            return full_text;
        }
    }
}
```

3. Update the `generate_synthetic_workloads` function signature:

```rust
pub fn generate_synthetic_workloads(
    config: &SyntheticConfig,
    tokenizer: Arc<Tokenizer>,
    sample_size: usize,
    seed: u64,
    max_tokens: Option<u32>,
) -> Result<Vec<Workload>> {
    let turns = config.turns.max(1);
    let generator = SyntheticDataGenerator::new(config, tokenizer, seed);
    let workloads: Vec<Workload> = (0..sample_size)
        .map(|i| generator.generate_workload(i, max_tokens, turns))
        .collect();
    Ok(workloads)
}
```

- [ ] **Step 1: Add `generate_prompt_for_turn` method and update `generate_workload`**

Replace the `generate_workload` method and add the new `generate_prompt_for_turn` method in `src/synthetic.rs`.

- [ ] **Step 2: Update `generate_synthetic_workloads` function**

Update the function to pass `turns` to `generate_workload`.

- [ ] **Step 3: Update existing tests**

Update `test_generate_workload` test (line 462) to pass the new `turns` parameter. The test uses `generator.generate_workload(0, Some(50))` — change to `generator.generate_workload(0, Some(50), 1)`.

- [ ] **Step 4: Run tests**

Run: `cargo test --lib synthetic::tests`
Expected: All synthetic tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/synthetic.rs
git commit -m "feat(synthetic): generate multi-turn conversations with configurable turns

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 3: Add test for multi-turn synthetic generation

**Files:**
- Modify: `src/synthetic.rs` (add test in existing test module)

Add a test that verifies multi-turn generation produces correct Conversation structure and token counts per turn:

```rust
#[test]
fn test_multi_turn_generation() {
    let tokenizer = Arc::new(Tokenizer::new("gpt-3.5-turbo").expect("Failed to create tokenizer"));
    let config = SyntheticConfig {
        prompt_tokens: 64,
        prompt_tokens_stdev: None,
        prompt_tokens_min: None,
        prompt_tokens_max: None,
        add_prefix: true,
        common_prefix_sample_ratio: 0.0,
        common_prefix_tokens: 0,
        turns: 3,
        turn_prompt_tokens: Some(48),
    };

    let generator = SyntheticDataGenerator::new(&config, tokenizer.clone(), 42);
    let workload = generator.generate_workload(0, Some(50), 3);

    match workload {
        Workload::MultiTurn(conv) => {
            assert_eq!(conv.user_turns.len(), 3, "Should have 3 user turns");
            assert_eq!(conv.max_tokens, Some(50));

            // First turn uses prompt_tokens (64)
            let first_tokens = tokenizer.count_tokens(&conv.user_turns[0]);
            assert!(
                (62..=68).contains(&first_tokens),
                "First turn token count {} should be close to 64",
                first_tokens
            );
            assert!(conv.user_turns[0].starts_with("[synthetic-0-t0]"), "First turn should have prefix");

            // Subsequent turns use turn_prompt_tokens (48)
            for (i, turn) in conv.user_turns.iter().skip(1).enumerate() {
                let turn_idx = i + 1;
                let turn_tokens = tokenizer.count_tokens(turn);
                assert!(
                    (46..=52).contains(&turn_tokens),
                    "Turn {} token count {} should be close to 48",
                    turn_idx,
                    turn_tokens
                );
                assert!(
                    turn.starts_with(&format!("[synthetic-0-t{}]", turn_idx)),
                    "Turn {} should have correct turn-specific prefix",
                    turn_idx
                );
            }
        }
        _ => panic!("Expected MultiTurn workload"),
    }
}

#[test]
fn test_multi_turn_deterministic() {
    let tokenizer = Arc::new(Tokenizer::new("gpt-3.5-turbo").expect("Failed to create tokenizer"));
    let config = SyntheticConfig {
        prompt_tokens: 50,
        prompt_tokens_stdev: None,
        prompt_tokens_min: None,
        prompt_tokens_max: None,
        add_prefix: true,
        common_prefix_sample_ratio: 0.0,
        common_prefix_tokens: 0,
        turns: 4,
        turn_prompt_tokens: None,
    };

    let workloads1 = generate_synthetic_workloads(&config, tokenizer.clone(), 5, 42, Some(100))
        .expect("Failed to generate workloads");
    let workloads2 = generate_synthetic_workloads(&config, tokenizer.clone(), 5, 42, Some(100))
        .expect("Failed to generate workloads");

    for (w1, w2) in workloads1.iter().zip(workloads2.iter()) {
        match (w1, w2) {
            (Workload::MultiTurn(c1), Workload::MultiTurn(c2)) => {
                assert_eq!(c1.user_turns.len(), c2.user_turns.len());
                for (t1, t2) in c1.user_turns.iter().zip(c2.user_turns.iter()) {
                    assert_eq!(t1, t2, "Same seed should produce identical turns");
                }
            }
            _ => panic!("Expected MultiTurn workloads"),
        }
    }
}
```

- [ ] **Step 1: Add multi-turn tests**

Add the two tests above to the `mod tests` block in `src/synthetic.rs`.

- [ ] **Step 2: Run all synthetic tests**

Run: `cargo test --lib synthetic::tests`
Expected: All tests pass including the two new ones.

- [ ] **Step 3: Commit**

```bash
git add src/synthetic.rs
git commit -m "test(synthetic): add multi-turn generation and determinism tests

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 4: Update example config with multi-turn synthetic example

**Files:**
- Modify: `examples/config.example.toml` (add synthetic multi-turn example)

Add an example in the `[input]` section comments (after the existing synthetic examples, around line 159):

```toml
#   example that generates multi-turn synthetic conversations (3 turns each):
#   [endpoint]
#   max_tokens = 512
#   [input]
#   file = "synthetic"
#   sample_size = 100
#   [input.synthetic]
#   prompt_tokens = 128
#   turns = 3
#   turn_prompt_tokens = 96
```

- [ ] **Step 1: Add multi-turn example to config doc comments**

Insert the example in the `[input]` file comments around line 159.

- [ ] **Step 2: Commit**

```bash
git add examples/config.example.toml
git commit -m "docs: add multi-turn synthetic config example

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Self-Review

**1. Spec coverage:**
- Config fields (`turns`, `turn_prompt_tokens`) → Task 1
- SyntheticConfig validation → Task 1
- Multi-turn generation in SyntheticDataGenerator → Task 2
- Reuse existing execute_conversation() → no code change needed (by design)
- System prompt support → already handled by execute_conversation(), Conversation has system_prompt field
- Tests → Task 3
- Config example → Task 4

**2. Placeholder scan:** No TBD/TODO markers. All code is shown explicitly.

**3. Type consistency:** `generate_workload` signature change from `(usize, Option<u32>)` to `(usize, Option<u32>, usize)` is propagated in `generate_synthetic_workloads`. All field names match.

**4. Scope check:** Focused — two config fields, one generator change, tests, docs. No execution path changes.
