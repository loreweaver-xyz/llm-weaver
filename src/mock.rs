use std::fmt::Formatter;

use async_trait::async_trait;
use serde::de::DeserializeOwned;
use tiktoken_rs::p50k_base;

use crate::*;

use self::types::StorageError;

pub struct MockChest;

#[async_trait]
impl TapestryChestHandler<MockConfig> for MockChest {
	type Error = StorageError;

	fn new() -> Self {
		Self {}
	}

	async fn save_tapestry_fragment<TID: TapestryId>(
		&self,
		_tapestry_id: &TID,
		_tapestry_fragment: TapestryFragment<MockConfig>,
		_increment: bool,
	) -> crate::Result<u64> {
		Ok(0)
	}

	async fn save_tapestry_metadata<TID: TapestryId, M: Debug + Clone + Send + Sync>(
		&self,
		_tapestry_id: TID,
		_metadata: M,
	) -> crate::Result<()> {
		Ok(())
	}

	async fn get_tapestry<TID: TapestryId>(&self, _tapestry_id: TID) -> crate::Result<Option<u16>> {
		Ok(Some(0))
	}

	async fn get_tapestry_fragment<TID: TapestryId>(
		&self,
		_tapestry_id: TID,
		_instance: Option<u64>,
	) -> crate::Result<Option<TapestryFragment<MockConfig>>> {
		Ok(Some(TapestryFragment { context_tokens: 0, context_messages: vec![] }))
	}

	async fn get_tapestry_metadata<TID: TapestryId, M: DeserializeOwned>(
		&self,
		_tapestry_id: TID,
	) -> crate::Result<Option<M>> {
		Ok(Some(serde_json::from_str("{}").unwrap()))
	}

	async fn delete_tapestry<TID: TapestryId>(&self, _tapestry_id: TID) -> crate::Result<()> {
		Ok(())
	}

	async fn delete_tapestry_fragment<TID: TapestryId>(
		&self,
		_tapestry_id: TID,
		_instance: Option<u64>,
	) -> crate::Result<()> {
		Ok(())
	}
}

#[derive(Debug, Clone)]
pub struct MockTapestryId;
impl TapestryId for MockTapestryId {
	fn base_key(&self) -> String {
		"test".to_string()
	}
}

#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub struct MockConfig;
impl Config for MockConfig {
	const TOKEN_THRESHOLD_PERCENTILE: BoundedU8<0, 100> = BoundedU8::new(70).unwrap();
	const MINIMUM_RESPONSE_LENGTH: u64 = 300;

	type PromptModel = MockLlm;
	type SummaryModel = MockLlm;
	type Chest = MockChest;

	fn convert_prompt_tokens_to_summary_model_tokens(
		tokens: PromptModelTokens<Self>,
	) -> SummaryModelTokens<Self> {
		tokens
	}
}

#[derive(Default, Debug, Copy, Clone, PartialEq, Eq)]
pub struct MockLlm;

#[async_trait]
impl Llm<MockConfig> for MockLlm {
	type Tokens = u16;
	type Parameters = ();
	type Request = MockLlmRequest;
	type Response = MockLlmResponse;

	fn count_tokens(content: &str) -> Result<Self::Tokens> {
		let bpe = p50k_base().unwrap();
		let tokens = bpe.encode_with_special_tokens(&content.to_string());

		tokens.len().try_into().map_err(|_| {
			LoomError::from(WeaveError::BadConfig(format!(
				"Number of tokens exceeds max tokens for model: {}",
				content
			)))
			.into()
		})
	}

	fn name(&self) -> &'static str {
		"TestLlm"
	}

	fn alias(&self) -> &'static str {
		"TestLlm"
	}

	async fn prompt(
		&self,
		_is_summarize: bool,
		_prompt_tokens: Self::Tokens,
		_msgs: Vec<Self::Request>,
		_params: &Self::Parameters,
		_max_tokens: Self::Tokens,
	) -> Result<Self::Response> {
		Ok(MockLlmResponse {})
	}

	fn max_context_length(&self) -> Self::Tokens {
		1000
	}

	fn convert_tokens_to_words(&self, tokens: Self::Tokens) -> Self::Tokens {
		tokens
	}

	fn ctx_msgs_to_prompt_requests(
		&self,
		msgs: &[ContextMessage<MockConfig>],
	) -> Vec<Self::Request> {
		msgs.iter().map(|msg| MockLlmRequest::from(msg.clone())).collect()
	}

	fn compute_cost(&self, prompt_tokens: Self::Tokens, response_tokens: Self::Tokens) -> f64 {
		(prompt_tokens + response_tokens) as f64
	}
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MockLlmRequest {
	pub id: u32,
	pub msg: String,
}

impl MockLlmRequest {
	pub fn new(id: u32, msg: String) -> Self {
		Self { id, msg }
	}
}

impl Display for MockLlmRequest {
	fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
		write!(f, "{}", self.msg)
	}
}

impl From<ContextMessage<MockConfig>> for MockLlmRequest {
	fn from(_msg: ContextMessage<MockConfig>) -> Self {
		Self {
			id: 0,                      // or some logic to assign an ID
			msg: "default".to_string(), // or use data from _msg
		}
	}
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MockLlmResponse;

impl Display for MockLlmResponse {
	fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
		write!(f, "TestLlmResponse")
	}
}

impl From<Option<String>> for MockLlmResponse {
	fn from(_msg: Option<String>) -> Self {
		Self {}
	}
}

impl From<MockLlmResponse> for Option<String> {
	fn from(_msg: MockLlmResponse) -> Self {
		Some("TestLlmResponse".to_string())
	}
}
