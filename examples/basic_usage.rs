use async_trait::async_trait;
use llm_weaver::{
	loom::Loom, types::WrapperRole, BoundedU8, Config, ContextMessage, Llm, LlmConfig, Result,
	TapestryChestHandler, TapestryFragment, TapestryId,
};
use serde::{Deserialize, Serialize};
use std::fmt::Display;

// Define a simple TapestryId
#[derive(Debug, Clone)]
struct MyTapestryId(String);

impl TapestryId for MyTapestryId {
	fn base_key(&self) -> String {
		format!("my_tapestry:{}", self.0)
	}
}

// Define a simple LLM
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq)]
struct MyLlm;

#[async_trait]
impl Llm<MyConfig> for MyLlm {
	type Tokens = u16;
	type Parameters = ();
	type Request = String;
	type Response = String;

	fn max_context_length(&self) -> Self::Tokens {
		1000
	}

	fn name(&self) -> &'static str {
		"MyLlm"
	}

	fn alias(&self) -> &'static str {
		"MyLlm"
	}

	fn count_tokens(content: &str) -> Result<Self::Tokens> {
		Ok(content.len() as u16)
	}

	async fn prompt(
		&self,
		_is_summarizing: bool,
		_prompt_tokens: Self::Tokens,
		msgs: Vec<Self::Request>,
		_params: &Self::Parameters,
		_max_tokens: Self::Tokens,
	) -> Result<Self::Response> {
		// Simple concatenation of messages for demonstration
		Ok(msgs.join(" "))
	}

	fn compute_cost(&self, prompt_tokens: Self::Tokens, response_tokens: Self::Tokens) -> f64 {
		(prompt_tokens + response_tokens) as f64 * 0.01
	}
}

// Define a simple storage handler
#[derive(Debug, Default)]
struct MyChest;

#[async_trait]
impl TapestryChestHandler<MyConfig> for MyChest {
	type Error = std::io::Error;

	fn new() -> Self {
		Self
	}

	async fn save_tapestry_fragment<TID: TapestryId>(
		&self,
		_tapestry_id: &TID,
		_tapestry_fragment: TapestryFragment<MyConfig>,
		_increment: bool,
	) -> Result<u64> {
		Ok(0)
	}

	async fn get_tapestry_fragment<TID: TapestryId>(
		&self,
		_tapestry_id: TID,
		_instance: Option<u64>,
	) -> Result<Option<TapestryFragment<MyConfig>>> {
		Ok(None)
	}

	// Implement other required methods...
}

// Define the configuration
#[derive(Debug, Clone, Default)]
struct MyConfig;

impl Config for MyConfig {
	const TOKEN_THRESHOLD_PERCENTILE: BoundedU8<0, 100> = BoundedU8::new(85).unwrap();
	const MINIMUM_RESPONSE_LENGTH: u64 = 10;

	type PromptModel = MyLlm;
	type SummaryModel = MyLlm;
	type Chest = MyChest;

	fn convert_prompt_tokens_to_summary_model_tokens(
		tokens: <Self::PromptModel as Llm<Self>>::Tokens,
	) -> <Self::SummaryModel as Llm<Self>>::Tokens {
		tokens
	}
}

#[tokio::main]
async fn main() -> Result<()> {
	// Create a Loom instance
	let loom = Loom::<MyConfig>::new();

	// Create LlmConfig instances
	let prompt_llm_config = LlmConfig { model: MyLlm, params: () };
	let summary_llm_config = LlmConfig { model: MyLlm, params: () };

	// Create a TapestryId
	let tapestry_id = MyTapestryId("example_conversation".to_string());

	// Define instructions and messages
	let instructions = "You are a helpful assistant.".to_string();
	let messages = vec![ContextMessage::new(
		WrapperRole::from("user"),
		"Hello, how are you?".to_string(),
		None,
		chrono::Utc::now().to_rfc3339(),
	)];

	// Use the Loom to generate a response
	let (response, fragment_id, was_summarized) = loom
		.weave(prompt_llm_config, summary_llm_config, tapestry_id, instructions, messages)
		.await?;

	println!("Response: {}", response);
	println!("Fragment ID: {}", fragment_id);
	println!("Was summarized: {}", was_summarized);

	Ok(())
}
