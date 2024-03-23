use std::fmt::Formatter;

use async_trait::async_trait;
use serde::de::DeserializeOwned;
use tiktoken_rs::p50k_base;

use crate::*;

use self::types::StorageError;

pub struct TestChest;

#[async_trait]
impl TapestryChestHandler<TestApp> for TestChest {
	type Error = StorageError;

	async fn save_tapestry_fragment<TID: TapestryId>(
		_tapestry_id: &TID,
		_tapestry_fragment: TapestryFragment<TestApp>,
		_increment: bool,
	) -> crate::Result<u64> {
		Ok(0)
	}

	async fn save_tapestry_metadata<
		TID: TapestryId,
		M: ToRedisArgs + Debug + Clone + Send + Sync,
	>(
		_tapestry_id: TID,
		_metadata: M,
	) -> crate::Result<()> {
		Ok(())
	}

	async fn get_tapestry<TID: TapestryId>(_tapestry_id: TID) -> crate::Result<Option<u16>> {
		Ok(Some(0))
	}

	async fn get_tapestry_fragment<TID: TapestryId>(
		_tapestry_id: TID,
		_instance: Option<u64>,
	) -> crate::Result<Option<TapestryFragment<TestApp>>> {
		Ok(Some(TapestryFragment { context_tokens: 0, context_messages: vec![] }))
	}

	async fn get_tapestry_metadata<TID: TapestryId, M: DeserializeOwned>(
		_tapestry_id: TID,
	) -> crate::Result<Option<M>> {
		Ok(Some(serde_json::from_str("{}").unwrap()))
	}

	async fn delete_tapestry<TID: TapestryId>(_tapestry_id: TID) -> crate::Result<()> {
		Ok(())
	}

	async fn delete_tapestry_fragment<TID: TapestryId>(
		_tapestry_id: TID,
		_instance: Option<u64>,
	) -> crate::Result<()> {
		Ok(())
	}
}

#[derive(Debug, Clone)]
pub struct TestTapestryId;
impl TapestryId for TestTapestryId {
	fn base_key(&self) -> String {
		"test".to_string()
	}
}

#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub struct TestApp;
impl Config for TestApp {
	const TOKEN_THRESHOLD_PERCENTILE: BoundedU8<0, 100> = BoundedU8::new(70).unwrap();
	type PromptModel = TestLlm;
	type SummaryModel = TestLlm;
	type Chest = TestChest;

	fn convert_prompt_tokens_to_summary_model_tokens(
		tokens: PromptModelTokens<Self>,
	) -> SummaryModelTokens<Self> {
		tokens
	}
}

impl<T: Config> Loom<T> for TestApp {}

#[derive(Default, Debug, Copy, Clone, PartialEq, Eq)]
pub struct TestLlm;

#[async_trait]
impl Llm<TestApp> for TestLlm {
	type Tokens = u16;
	type Parameters = ();
	type Request = TestLlmRequest;
	type Response = TestLlmResponse;

	fn count_tokens(content: String) -> Result<Self::Tokens> {
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

	async fn prompt(
		&self,
		_is_summarize: bool,
		_prompt_tokens: Self::Tokens,
		_msgs: Vec<Self::Request>,
		_params: &Self::Parameters,
		_max_tokens: Self::Tokens,
	) -> Result<Self::Response> {
		Ok(TestLlmResponse {})
	}

	fn max_context_length(&self) -> Self::Tokens {
		1000
	}

	fn convert_tokens_to_words(&self, tokens: Self::Tokens) -> Self::Tokens {
		tokens
	}

	fn ctx_msgs_to_prompt_requests(&self, msgs: &[ContextMessage<TestApp>]) -> Vec<Self::Request> {
		msgs.iter().map(|msg| TestLlmRequest::from(msg.clone())).collect()
	}
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TestLlmRequest {
	pub id: u32,
	pub msg: String,
}

impl TestLlmRequest {
	pub fn new(id: u32, msg: String) -> Self {
		Self { id, msg }
	}
}

impl Display for TestLlmRequest {
	fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
		write!(f, "{}", self.msg)
	}
}

impl From<ContextMessage<TestApp>> for TestLlmRequest {
	fn from(_msg: ContextMessage<TestApp>) -> Self {
		Self {
			id: 0,                      // or some logic to assign an ID
			msg: "default".to_string(), // or use data from _msg
		}
	}
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TestLlmResponse;

impl Display for TestLlmResponse {
	fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
		write!(f, "TestLlmResponse")
	}
}

impl From<Option<String>> for TestLlmResponse {
	fn from(_msg: Option<String>) -> Self {
		Self {}
	}
}

impl From<TestLlmResponse> for Option<String> {
	fn from(_msg: TestLlmResponse) -> Self {
		Some("TestLlmResponse".to_string())
	}
}
