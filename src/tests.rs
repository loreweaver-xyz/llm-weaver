use async_openai::types::Role;

use crate::mock::{TestApp, TestLlm, TestLlmRequest, TestTapestryId};

use super::*;

#[tokio::test]
async fn prompt() {
	assert!(TestApp::weave(
		LlmConfig::<TestApp, TestLlm> { model: TestLlm, params: () },
		LlmConfig::<TestApp, TestLlm> { model: TestLlm, params: () },
		TestTapestryId,
		"instructions".to_string(),
		vec![ContextMessage::<TestApp>::new(
			WrapperRole::Role(Role::Assistant),
			"Hello".to_string(),
			None,
			"time".to_string()
		)],
	)
	.await
	.is_ok());
}

#[test]
fn vec_prompt_msgs_deque_extend() {
	let mut deque = VecPromptMsgsDeque::<TestApp, TestLlm>::new();

	let msg1 = TestLlmRequest::new(1, "Hello".to_string());
	let msg2 = TestLlmRequest::new(2, "World".to_string());
	let requests = vec![msg1.clone(), msg2.clone()];

	let msg1_token_count =
		<TestApp as Config>::PromptModel::count_tokens(&msg1.msg).expect("Token count failed");
	let msg2_token_count =
		<TestApp as Config>::PromptModel::count_tokens(&msg2.msg).expect("Token count failed");

	deque.extend(requests.clone());
	assert_eq!(deque.tokens, msg1_token_count + msg2_token_count);
	assert_eq!(deque.inner.len(), requests.len());
	assert_eq!(deque.inner[0], msg1);
	assert_eq!(deque.inner[1], msg2);
}

#[test]
fn vec_prompt_msgs_deque_append() {
	let msg1 = TestLlmRequest::new(1, "Hello".to_string());
	let msg2 = TestLlmRequest::new(2, "World".to_string());
	let mut msgs = std::collections::VecDeque::new();
	msgs.push_back(msg1.clone());
	msgs.push_back(msg2.clone());
	let mut vec_prompt_msgs_deque = VecPromptMsgsDeque::<TestApp, TestLlm>::new();

	let msg1_token_count =
		<TestApp as Config>::PromptModel::count_tokens(&msg1.msg).expect("Token count failed");
	let msg2_token_count =
		<TestApp as Config>::PromptModel::count_tokens(&msg2.msg).expect("Token count failed");

	vec_prompt_msgs_deque.append(&mut msgs);
	assert_eq!(vec_prompt_msgs_deque.tokens, msg1_token_count + msg2_token_count);
	assert_eq!(vec_prompt_msgs_deque.inner.len(), 2);
	assert_eq!(vec_prompt_msgs_deque.inner.front(), Some(&msg1));
}

#[test]
fn vec_prompt_msgs_deque_new() {
	let deque = VecPromptMsgsDeque::<TestApp, TestLlm>::new();
	assert!(deque.inner.is_empty());
	assert_eq!(deque.tokens, 0);
}

#[test]
fn vec_prompt_msgs_deque_with_capacity() {
	let deque = VecPromptMsgsDeque::<TestApp, TestLlm>::with_capacity(10);
	assert!(deque.inner.capacity() >= 10);
	assert_eq!(deque.tokens, 0);
}

#[test]
fn vec_prompt_msgs_deque_push_front() {
	let mut deque = VecPromptMsgsDeque::<TestApp, TestLlm>::new();
	let request = TestLlmRequest::new(1, "Hello".to_string());

	let request_token_count =
		<TestApp as Config>::PromptModel::count_tokens(&request.msg).expect("Token count failed");

	deque.push_front(request.clone());
	assert_eq!(deque.tokens, request_token_count);
	assert_eq!(deque.inner.front(), Some(&request));
}

#[test]
fn vec_prompt_msgs_deque_push_back() {
	let mut deque = VecPromptMsgsDeque::<TestApp, TestLlm>::new();
	let request = TestLlmRequest::new(1, "Hello".to_string());

	let request_token_count =
		<TestApp as Config>::PromptModel::count_tokens(&request.msg).expect("Token count failed");

	deque.push_back(request.clone());
	assert_eq!(deque.tokens, request_token_count);
	assert_eq!(deque.inner.back(), Some(&request));
}

#[test]
fn vec_prompt_msgs_deque_truncate() {
	let mut deque = VecPromptMsgsDeque::<TestApp, TestLlm>::new();
	let msg1 = TestLlmRequest::new(1, "First message".to_string());
	let msg2 = TestLlmRequest::new(2, "Second message".to_string());
	let msg3 = TestLlmRequest::new(3, "Third message".to_string());
	let requests = vec![msg1.clone(), msg2.clone(), msg3.clone()];

	deque.extend(requests.clone());

	let total_token_count = requests
		.iter()
		.map(|r| <TestApp as Config>::PromptModel::count_tokens(&r.msg).unwrap())
		.sum::<u16>();

	assert_eq!(deque.tokens, total_token_count);

	deque.truncate(1);
	let msg1_token_count =
		<TestApp as Config>::PromptModel::count_tokens(&msg1.msg).expect("Token count failed");
	assert_eq!(deque.tokens, msg1_token_count);
	assert_eq!(deque.inner.len(), 1);
	assert_eq!(deque.inner.front(), Some(&msg1));
}

#[test]
fn vec_prompt_msgs_deque_into_vec() {
	let mut deque = VecPromptMsgsDeque::<TestApp, TestLlm>::new();
	let request1 = TestLlmRequest::new(1, "Message 1".to_string());
	let request2 = TestLlmRequest::new(2, "Message 2".to_string());

	deque.extend(vec![request1.clone(), request2.clone()]);

	let vec = deque.into_vec();
	assert_eq!(vec.len(), 2);
	assert_eq!(vec[0], request1);
	assert_eq!(vec[1], request2);
}
