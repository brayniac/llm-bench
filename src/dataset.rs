use anyhow::{Context, Result};
use arrow::array::{Array, ListArray, StringArray};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::io::Write;
use std::path::{Path, PathBuf};

/// Known dataset names that can be auto-downloaded from HuggingFace.
pub const KNOWN_DATASETS: &[(&str, &str)] = &[
    (
        "openorca",
        "Open-Orca/OpenOrca (~1 GB download, 1M GPT-4 prompts)",
    ),
    (
        "sharegpt",
        "ShareGPT Vicuna (~670 MB download, ~90K multi-turn conversations)",
    ),
    (
        "toolbench",
        "THUDM/ToolBench (~300 MB download, tool-use trajectories)",
    ),
    (
        "gpt4-tool-use",
        "openai/gpt-4-tool-use-v1.0 (~200 MB download, GPT-4 tool calling examples)",
    ),
    (
        "claude-distill",
        "Kassadin88/Claude-Distills (~800 MB download, Claude Sonnet/Opus distillation with system prompts)",
    ),
];

/// Resolve an input path: if it exists on disk, use it directly.
/// If it matches a known dataset name, download from HuggingFace and convert to JSONL.
pub async fn resolve_input(path: &Path) -> Result<PathBuf> {
    // If the path exists on disk, use it directly
    if path.exists() {
        return Ok(path.to_path_buf());
    }

    // Extract the name (strip directory components and extension)
    let name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .map(|s| s.to_lowercase())
        .unwrap_or_default();

    match name.as_str() {
        "openorca" => download_openorca().await,
        "sharegpt" => download_sharegpt().await,
        "toolbench" => download_toolbench().await,
        "gpt4-tool-use" => download_gpt4_tooluse().await,
        "claude-distill" => download_claude_distill().await,
        _ => {
            let known = KNOWN_DATASETS
                .iter()
                .map(|(name, desc)| format!("  {name:12} — {desc}"))
                .collect::<Vec<_>>()
                .join("\n");
            anyhow::bail!(
                "Input file '{}' not found.\n\nKnown datasets (auto-downloaded from HuggingFace):\n{}",
                path.display(),
                known
            );
        }
    }
}

fn cache_dir() -> Result<PathBuf> {
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .unwrap_or_else(|_| ".".to_string());
    let dir = PathBuf::from(home)
        .join(".cache")
        .join("llm-perf")
        .join("datasets");
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

async fn download_sharegpt() -> Result<PathBuf> {
    let cache = cache_dir()?;
    let cached_path = cache.join("sharegpt.jsonl");

    if cached_path.exists() {
        log::info!("Using cached ShareGPT dataset: {}", cached_path.display());
        return Ok(cached_path);
    }

    log::info!("Downloading ShareGPT dataset from HuggingFace (~670 MB)...");

    let api = hf_hub::api::tokio::Api::new()?;
    let repo = api.dataset("anon8231489123/ShareGPT_Vicuna_unfiltered".to_string());
    let path = repo
        .get("ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json")
        .await
        .context("Failed to download ShareGPT dataset")?;

    log::info!("Converting ShareGPT dataset to JSONL...");

    // Parse JSON array and convert to our JSONL format
    let data = std::fs::read_to_string(&path).context("Failed to read downloaded ShareGPT JSON")?;
    let entries: Vec<serde_json::Value> =
        serde_json::from_str(&data).context("Failed to parse ShareGPT JSON")?;

    let file = std::fs::File::create(&cached_path)?;
    let mut writer = std::io::BufWriter::new(file);
    let mut count = 0;

    for entry in &entries {
        if let Some(conversations) = entry.get("conversations").and_then(|c| c.as_array()) {
            if conversations.is_empty() {
                continue;
            }
            let line = serde_json::json!({ "conversations": conversations });
            serde_json::to_writer(&mut writer, &line)?;
            writeln!(&mut writer)?;
            count += 1;
        }
    }

    writer.flush()?;
    log::info!(
        "Cached {} ShareGPT conversations to {}",
        count,
        cached_path.display()
    );

    Ok(cached_path)
}

async fn download_openorca() -> Result<PathBuf> {
    let cache = cache_dir()?;
    let cached_path = cache.join("openorca.jsonl");

    if cached_path.exists() {
        log::info!("Using cached OpenOrca dataset: {}", cached_path.display());
        return Ok(cached_path);
    }

    log::info!("Downloading OpenOrca dataset from HuggingFace (GPT-4 subset, ~1 GB)...");

    let api = hf_hub::api::tokio::Api::new()?;
    let repo = api.dataset("Open-Orca/OpenOrca".to_string());
    let path = repo
        .get("1M-GPT4-Augmented.parquet")
        .await
        .context("Failed to download OpenOrca dataset")?;

    log::info!("Converting OpenOrca dataset to JSONL...");

    let file = std::fs::File::open(&path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let reader = builder.build()?;

    let out_file = std::fs::File::create(&cached_path)?;
    let mut writer = std::io::BufWriter::new(out_file);
    let mut count = 0;

    for batch in reader {
        let batch = batch?;

        let question_col = batch
            .column_by_name("question")
            .context("missing 'question' column in OpenOrca parquet")?;
        let questions = question_col
            .as_any()
            .downcast_ref::<StringArray>()
            .context("'question' column is not a string array")?;

        // system_prompt column is optional
        let system_col = batch.column_by_name("system_prompt");
        let systems = system_col.and_then(|c| c.as_any().downcast_ref::<StringArray>());

        for i in 0..batch.num_rows() {
            let question = questions.value(i);

            // Prepend system prompt if present and non-empty
            let prompt = if let Some(systems) = systems {
                let system = systems.value(i);
                if system.is_empty() {
                    question.to_string()
                } else {
                    format!("{}\n\n{}", system, question)
                }
            } else {
                question.to_string()
            };

            let line = serde_json::json!({ "prompt": prompt });
            serde_json::to_writer(&mut writer, &line)?;
            writeln!(&mut writer)?;
            count += 1;
        }
    }

    writer.flush()?;
    log::info!(
        "Cached {} OpenOrca prompts to {}",
        count,
        cached_path.display()
    );

    Ok(cached_path)
}

async fn download_toolbench() -> Result<PathBuf> {
    let cache = cache_dir()?;
    let cached_path = cache.join("toolbench.jsonl");

    if cached_path.exists() {
        log::info!("Using cached ToolBench dataset: {}", cached_path.display());
        return Ok(cached_path);
    }

    log::info!("Downloading ToolBench dataset from HuggingFace (~300 MB)...");

    let api = hf_hub::api::tokio::Api::new()?;
    let repo = api.dataset("THUDM/ToolBench".to_string());

    // Try parquet first, then JSONL
    let path = match repo.get("data/train.parquet").await {
        Ok(p) => p,
        Err(_) => {
            log::info!("Parquet not found, trying JSONL...");
            repo.get("data/train.jsonl").await?
        }
    };

    log::info!("Converting ToolBench dataset to JSONL...");

    // Check if it's parquet or jsonl
    if path.extension().map(|e| e == "parquet").unwrap_or(false) {
        let file = std::fs::File::open(&path)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
        let reader = builder.build()?;

        let out_file = std::fs::File::create(&cached_path)?;
        let mut writer = std::io::BufWriter::new(out_file);
        let mut count = 0;

        for batch in reader {
            let batch = batch?;

            // ToolBench format: instruction, tool_type, tool_response, response
            let instruction_col = batch.column_by_name("instruction");
            let tool_type_col = batch.column_by_name("tool_type");
            let tool_response_col = batch.column_by_name("tool_response");
            let response_col = batch.column_by_name("response");

            if let (Some(inst), Some(tool_t), Some(tool_r), Some(resp)) = (
                instruction_col,
                tool_type_col,
                tool_response_col,
                response_col,
            ) {
                let instructions = inst
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .context("instruction column is not a string array")?;
                let tool_types = tool_t
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .context("tool_type column is not a string array")?;
                let tool_responses = tool_r
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .context("tool_response column is not a string array")?;
                let responses = resp
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .context("response column is not a string array")?;

                for i in 0..batch.num_rows() {
                    let instruction = instructions.value(i);
                    let tool_type = tool_types.value(i);
                    let tool_response = tool_responses.value(i);
                    let response = responses.value(i);

                    // Create a ShareGPT-style conversation with system prompt
                    let conversation = serde_json::json!({
                        "conversations": [
                            {"from": "system", "value": "You are a helpful AI assistant. Use the provided tools to answer the user's question."},
                            {"from": "user", "value": format!("{} [Tool: {}]", instruction, tool_type)},
                            {"from": "assistant", "value": format!("{}\n\nTool output: {}", response, tool_response)}
                        ]
                    });

                    serde_json::to_writer(&mut writer, &conversation)?;
                    writeln!(&mut writer)?;
                    count += 1;
                }
            }
        }

        writer.flush()?;
        log::info!(
            "Cached {} ToolBench conversations to {}",
            count,
            cached_path.display()
        );
    } else {
        // JSONL format - read and convert
        let data = std::fs::read_to_string(&path).context("Failed to read ToolBench JSONL")?;
        let out_file = std::fs::File::create(&cached_path)?;
        let mut writer = std::io::BufWriter::new(out_file);
        let mut count = 0;

        for line in data.lines() {
            if line.trim().is_empty() {
                continue;
            }

            let entry: serde_json::Value = serde_json::from_str(line)?;
            let conversation = convert_toolbench_entry(&entry);

            serde_json::to_writer(&mut writer, &conversation)?;
            writeln!(&mut writer)?;
            count += 1;
        }

        writer.flush()?;
        log::info!(
            "Cached {} ToolBench conversations to {}",
            count,
            cached_path.display()
        );
    }

    Ok(cached_path)
}

async fn download_gpt4_tooluse() -> Result<PathBuf> {
    let cache = cache_dir()?;
    let cached_path = cache.join("gpt4-tool-use.jsonl");

    if cached_path.exists() {
        log::info!(
            "Using cached GPT-4 Tool Use dataset: {}",
            cached_path.display()
        );
        return Ok(cached_path);
    }

    log::info!("Downloading GPT-4 Tool Use dataset from HuggingFace (~200 MB)...");

    let api = hf_hub::api::tokio::Api::new()?;
    let repo = api.dataset("openai/gpt-4-tool-use-v1.0".to_string());

    // Try different file names
    let path = match repo.get("data/train-00000-of-00001.parquet").await {
        Ok(p) => p,
        Err(_) => match repo.get("data/validation-00000-of-00001.parquet").await {
            Ok(p) => p,
            Err(_) => {
                log::info!("Parquet not found, trying JSONL...");
                repo.get("data/train.jsonl").await?
            }
        },
    };

    log::info!("Converting GPT-4 Tool Use dataset to JSONL...");

    if path.extension().map(|e| e == "parquet").unwrap_or(false) {
        let file = std::fs::File::open(&path)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
        let reader = builder.build()?;

        let out_file = std::fs::File::create(&cached_path)?;
        let mut writer = std::io::BufWriter::new(out_file);
        let mut count = 0;

        for batch in reader {
            let batch = batch?;

            if let Some(messages_col) = batch.column_by_name("messages") {
                let messages_list = messages_col
                    .as_any()
                    .downcast_ref::<ListArray>()
                    .context("messages column is not a list array")?;

                for i in 0..batch.num_rows() {
                    let messages_array = messages_list.value(i);
                    let messages_strings = messages_array.as_any().downcast_ref::<StringArray>();

                    if let Some(messages) = messages_strings {
                        let mut conversations = vec![];
                        for j in 0..messages.len() {
                            let msg_str = messages.value(j);
                            if let Ok(msg) = serde_json::from_str::<serde_json::Value>(msg_str) {
                                if let Some(role) = msg.get("role").and_then(|r| r.as_str()) {
                                    let content =
                                        msg.get("content").and_then(|c| c.as_str()).unwrap_or("");
                                    conversations.push(serde_json::json!({
                                        "from": role,
                                        "value": content
                                    }));
                                }
                            }
                        }

                        if !conversations.is_empty() {
                            let entry = serde_json::json!({
                                "conversations": conversations
                            });
                            serde_json::to_writer(&mut writer, &entry)?;
                            writeln!(&mut writer)?;
                            count += 1;
                        }
                    }
                }
            }
        }

        writer.flush()?;
        log::info!(
            "Cached {} GPT-4 Tool Use conversations to {}",
            count,
            cached_path.display()
        );
    } else {
        let data = std::fs::read_to_string(&path).context("Failed to read GPT-4 Tool Use JSONL")?;
        let out_file = std::fs::File::create(&cached_path)?;
        let mut writer = std::io::BufWriter::new(out_file);
        let mut count = 0;

        for line in data.lines() {
            if line.trim().is_empty() {
                continue;
            }

            let entry: serde_json::Value = serde_json::from_str(line)?;
            let conversations = extract_messages(&entry);
            let conversation = serde_json::json!({
                "conversations": conversations
            });

            serde_json::to_writer(&mut writer, &conversation)?;
            writeln!(&mut writer)?;
            count += 1;
        }

        writer.flush()?;
        log::info!(
            "Cached {} GPT-4 Tool Use conversations to {}",
            count,
            cached_path.display()
        );
    }

    Ok(cached_path)
}

async fn download_claude_distill() -> Result<PathBuf> {
    let cache = cache_dir()?;
    let cached_path = cache.join("claude-distill.jsonl");

    if cached_path.exists() {
        log::info!(
            "Using cached Claude Distillation dataset: {}",
            cached_path.display()
        );
        return Ok(cached_path);
    }

    log::info!("Downloading Claude Distillation dataset from HuggingFace (~800 MB)...");

    let api = hf_hub::api::tokio::Api::new()?;
    let repo = api.dataset("Kassadin88/Claude-Distills".to_string());

    // Try parquet first, then JSONL
    let path = match repo.get("claude_distill.parquet").await {
        Ok(p) => p,
        Err(_) => {
            log::info!("Parquet not found, trying JSONL...");
            repo.get("claude_distill.jsonl").await?
        }
    };

    log::info!("Converting Claude Distillation dataset to JSONL...");

    if path.extension().map(|e| e == "parquet").unwrap_or(false) {
        let file = std::fs::File::open(&path)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
        let reader = builder.build()?;

        let out_file = std::fs::File::create(&cached_path)?;
        let mut writer = std::io::BufWriter::new(out_file);
        let mut count = 0;

        for batch in reader {
            let batch = batch?;

            if let Some(messages_col) = batch.column_by_name("messages") {
                let messages_list = messages_col
                    .as_any()
                    .downcast_ref::<ListArray>()
                    .context("messages column is not a list array")?;

                for i in 0..batch.num_rows() {
                    let messages_array = messages_list.value(i);
                    let messages_strings = messages_array.as_any().downcast_ref::<StringArray>();

                    if let Some(messages) = messages_strings {
                        let mut conversations = vec![];
                        for j in 0..messages.len() {
                            let msg_str = messages.value(j);
                            if let Ok(msg) = serde_json::from_str::<serde_json::Value>(msg_str) {
                                if let Some(role) = msg.get("role").and_then(|r| r.as_str()) {
                                    let content =
                                        msg.get("content").and_then(|c| c.as_str()).unwrap_or("");
                                    conversations.push(serde_json::json!({
                                        "from": role,
                                        "value": content
                                    }));
                                }
                            }
                        }

                        if !conversations.is_empty() {
                            let entry = serde_json::json!({
                                "conversations": conversations
                            });
                            serde_json::to_writer(&mut writer, &entry)?;
                            writeln!(&mut writer)?;
                            count += 1;
                        }
                    }
                }
            }
        }

        writer.flush()?;
        log::info!(
            "Cached {} Claude Distillation conversations to {}",
            count,
            cached_path.display()
        );
    } else {
        let data =
            std::fs::read_to_string(&path).context("Failed to read Claude Distillation JSONL")?;
        let out_file = std::fs::File::create(&cached_path)?;
        let mut writer = std::io::BufWriter::new(out_file);
        let mut count = 0;

        for line in data.lines() {
            if line.trim().is_empty() {
                continue;
            }

            let entry: serde_json::Value = serde_json::from_str(line)?;
            let conversations = extract_messages(&entry);
            let conversation = serde_json::json!({
                "conversations": conversations
            });

            serde_json::to_writer(&mut writer, &conversation)?;
            writeln!(&mut writer)?;
            count += 1;
        }

        writer.flush()?;
        log::info!(
            "Cached {} Claude Distillation conversations to {}",
            count,
            cached_path.display()
        );
    }

    Ok(cached_path)
}

fn convert_toolbench_entry(entry: &serde_json::Value) -> serde_json::Value {
    let instruction = entry
        .get("instruction")
        .or_else(|| entry.get("query"))
        .or_else(|| entry.get("prompt"))
        .and_then(|v| v.as_str())
        .unwrap_or("");

    let response = entry
        .get("response")
        .or_else(|| entry.get("answer"))
        .or_else(|| entry.get("output"))
        .and_then(|v| v.as_str())
        .unwrap_or("");

    let tool_type = entry
        .get("tool_type")
        .or_else(|| entry.get("tool"))
        .and_then(|v| v.as_str())
        .unwrap_or("");

    let tool_response = entry
        .get("tool_response")
        .or_else(|| entry.get("tool_output"))
        .and_then(|v| v.as_str())
        .unwrap_or("");

    let conversations = if !tool_type.is_empty() {
        vec![
            serde_json::json!({"from": "system", "value": "You are a helpful AI assistant. Use the provided tools to answer the user's question."}),
            serde_json::json!({"from": "user", "value": format!("{} [Tool: {}]", instruction, tool_type)}),
            serde_json::json!({"from": "assistant", "value": format!("{}\n\nTool output: {}", response, tool_response)}),
        ]
    } else {
        vec![
            serde_json::json!({"from": "system", "value": "You are a helpful AI assistant."}),
            serde_json::json!({"from": "user", "value": instruction}),
            serde_json::json!({"from": "assistant", "value": response}),
        ]
    };

    serde_json::json!({"conversations": conversations})
}

fn extract_messages(entry: &serde_json::Value) -> Vec<serde_json::Value> {
    if let Some(messages) = entry.get("messages").and_then(|m| m.as_array()) {
        return messages
            .iter()
            .filter_map(|m| {
                let role = m.get("role")?.as_str()?;
                let content = m.get("content")?.as_str()?;
                Some(serde_json::json!({"from": role, "value": content}))
            })
            .collect();
    }

    if let Some(conv) = entry.get("conversation").and_then(|c| c.as_array()) {
        return conv
            .iter()
            .filter_map(|m| {
                let role = m.get("role").or_else(|| m.get("from"))?.as_str()?;
                let content = m.get("content").or_else(|| m.get("value"))?.as_str()?;
                Some(serde_json::json!({"from": role, "value": content}))
            })
            .collect();
    }

    Vec::new()
}
