use std::fmt::Formatter;

// test VecPromptMsgsDeque methods
use super::*;

#[derive(Default, Debug, Clone, PartialEq, Eq)]
struct TestConfig;
impl Config for TestConfig {
	type PromptModel = TestLlm;
	type SummaryModel = TestLlm;

	// implement all Config methods
	fn convert_prompt_tokens_to_summary_model_tokens(
		_tokens: PromptModelTokens<Self>,
	) -> SummaryModelTokens<Self> {
		unimplemented!()
	}
}
#[derive(Default, Debug, Copy, Clone, PartialEq, Eq)]
struct TestLlm;
impl Llm<TestConfig> for TestLlm {
	type Tokens = u8;
	type Parameters = ();
	type Request = TestLlmRequest;
	type Response = TestLlmResponse;

	fn count_tokens(_msg: String) -> Result<Self::Tokens> {
		Ok(4)
	}

	fn name(&self) -> &'static str {
		"TestLlm"
	}

	fn prompt<'life0, 'life1, 'async_trait>(
		&'life0 self,
		_is_summarize: bool,
		_prompt_tokens: Self::Tokens,
		_msgs: Vec<Self::Request>,
		_params: &'life1 Self::Parameters,
		_max_tokens: Self::Tokens,
	) -> core::pin::Pin<
		Box<
			dyn core::future::Future<Output = Result<Self::Response>>
				+ core::marker::Send
				+ 'async_trait,
		>,
	>
	where
		'life0: 'async_trait,
		'life1: 'async_trait,
		Self: 'async_trait,
	{
		unimplemented!()
	}

	fn max_context_length(&self) -> Self::Tokens {
		4
	}

	fn get_max_token_limit(&self) -> Self::Tokens {
		4
	}

	fn convert_tokens_to_words(&self, tokens: Self::Tokens) -> Self::Tokens {
		tokens
	}

	fn ctx_msgs_to_prompt_requests(
		&self,
		msgs: &[ContextMessage<TestConfig>],
	) -> Vec<Self::Request> {
		msgs.iter().map(|msg| TestLlmRequest::from(msg.clone())).collect()
	}
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TestLlmRequest {
	id: u32,
	msg: String,
}

impl TestLlmRequest {
	fn new(id: u32, msg: String) -> Self {
		Self { id, msg }
	}
}

impl Display for TestLlmRequest {
	fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
		write!(f, "TestLlmRequest(id: {}, msg: '{}')", self.id, self.msg)
	}
}

impl From<ContextMessage<TestConfig>> for TestLlmRequest {
	fn from(_msg: ContextMessage<TestConfig>) -> Self {
		Self {
			id: 0,                      // or some logic to assign an ID
			msg: "default".to_string(), // or use data from _msg
		}
	}
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TestLlmResponse;

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

#[test]
fn vec_prompt_msgs_deque_append() {
	let msgs = vec![
		TestLlmRequest::new(1, "Hello".to_string()),
		TestLlmRequest::new(2, "World".to_string()),
	];
	let mut vec_prompt_msgs_deque = VecPromptMsgsDeque::<TestConfig, TestLlm>::new();

	vec_prompt_msgs_deque.append(&mut msgs.into());
	assert_eq!(vec_prompt_msgs_deque.tokens, 8);
	assert_eq!(vec_prompt_msgs_deque.inner.len(), 2);
	assert_eq!(vec_prompt_msgs_deque.inner.front().unwrap().id, 1);
	assert_eq!(vec_prompt_msgs_deque.inner.front().unwrap().msg, "Hello");
}

#[test]
fn vec_prompt_msgs_deque_new() {
	let deque = VecPromptMsgsDeque::<TestConfig, TestLlm>::new();
	assert!(deque.inner.is_empty());
	assert_eq!(deque.tokens, 0);
}

#[test]
fn vec_prompt_msgs_deque_with_capacity() {
	let deque = VecPromptMsgsDeque::<TestConfig, TestLlm>::with_capacity(10);
	assert!(deque.inner.capacity() >= 10);
	assert_eq!(deque.tokens, 0);
}

#[test]
fn vec_prompt_msgs_deque_push_front() {
	let mut deque = VecPromptMsgsDeque::<TestConfig, TestLlm>::new();
	let request = TestLlmRequest::new(1, "Hello".to_string());

	deque.push_front(request.clone());
	assert_eq!(deque.tokens, 4);
	assert_eq!(deque.inner.front(), Some(&request));
}

#[test]
fn vec_prompt_msgs_deque_truncate() {
	let mut deque = VecPromptMsgsDeque::<TestConfig, TestLlm>::new();
	deque.extend(vec![
		TestLlmRequest::new(1, "First message".to_string()),
		TestLlmRequest::new(2, "Second message".to_string()),
		TestLlmRequest::new(3, "Third message".to_string()),
	]);
	assert_eq!(deque.tokens, 12); // Assuming each request counts as 4 tokens

	deque.truncate(1);
	assert_eq!(deque.tokens, 4); // Tokens should be 4 after truncating to 1 request
	assert_eq!(deque.inner.len(), 1);
	assert_eq!(
		deque.inner.front(),
		Some(&TestLlmRequest { id: 1, msg: "First message".to_string() })
	);
}

#[test]
fn vec_prompt_msgs_deque_extend() {
	let mut deque = VecPromptMsgsDeque::<TestConfig, TestLlm>::new();
	let requests = vec![
		TestLlmRequest::new(4, "Extend".to_string()),
		TestLlmRequest::new(5, "Test".to_string()),
	];

	deque.extend(requests.clone());
	assert_eq!(deque.tokens, 8);
	assert_eq!(deque.inner.len(), 2);
	assert_eq!(deque.inner[0].id, 4);
	assert_eq!(deque.inner[0].msg, "Extend");
	assert_eq!(deque.inner[1].id, 5);
	assert_eq!(deque.inner[1].msg, "Test");
}

#[test]
fn vec_prompt_msgs_deque_into_vec() {
	let mut deque = VecPromptMsgsDeque::<TestConfig, TestLlm>::new();
	let request1 = TestLlmRequest::new(1, "Message 1".to_string());
	let request2 = TestLlmRequest::new(2, "Message 2".to_string());

	deque.extend(vec![request1.clone(), request2.clone()]);

	let vec = deque.into_vec();
	assert_eq!(vec.len(), 2);
	assert_eq!(vec[0], request1);
	assert_eq!(vec[1], request2);
}
