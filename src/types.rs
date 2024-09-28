use std::{collections::VecDeque, error::Error};

use async_openai::types::Role;
use num_traits::{CheckedAdd, FromPrimitive, SaturatingAdd};
use serde::{Deserialize, Serialize};
use tracing::error;

use crate::{Config, Llm};

pub type PromptModelTokens<T> = <<T as Config>::PromptModel as Llm<T>>::Tokens;
pub type SummaryModelTokens<T> = <<T as Config>::SummaryModel as Llm<T>>::Tokens;
pub type PromptModelRequest<T> = <<T as Config>::PromptModel as Llm<T>>::Request;

/// Base type for all configuration parameters.
pub type F32 = f32;

pub const SYSTEM_ROLE: &str = "system";
pub const ASSISTANT_ROLE: &str = "assistant";
pub const USER_ROLE: &str = "user";
pub const FUNCTION_ROLE: &str = "function";

/// Wrapped [`Role`] for custom implementations.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum WrapperRole {
	Role(Role),
}

impl Default for WrapperRole {
	fn default() -> Self {
		Self::Role(Role::User)
	}
}

impl From<&str> for WrapperRole {
	fn from(role: &str) -> Self {
		match role {
			SYSTEM_ROLE => Self::Role(Role::System),
			ASSISTANT_ROLE => Self::Role(Role::Assistant),
			USER_ROLE => Self::Role(Role::User),
			FUNCTION_ROLE => Self::Role(Role::Function),
			_ => panic!(
				"Invalid role: {} \n Valid roles: {} | {} | {} | {}",
				role, SYSTEM_ROLE, ASSISTANT_ROLE, USER_ROLE, FUNCTION_ROLE
			),
		}
	}
}

impl From<WrapperRole> for String {
	fn from(role: WrapperRole) -> Self {
		match role {
			WrapperRole::Role(Role::System) => SYSTEM_ROLE.to_string(),
			WrapperRole::Role(Role::Assistant) => ASSISTANT_ROLE.to_string(),
			WrapperRole::Role(Role::User) => USER_ROLE.to_string(),
			WrapperRole::Role(Role::Function) => FUNCTION_ROLE.to_string(),
			WrapperRole::Role(_) => panic!("Invalid role"),
		}
	}
}

#[derive(Debug, thiserror::Error)]
pub enum LoomError {
	#[error("LLM error: {0}")]
	Llm(#[from] Box<dyn Error + Send + Sync>),
	#[error("Storage error: {0}")]
	Storage(#[from] StorageError),
	#[error("Unknown error: {0}")]
	UnknownError(String),
}

impl LoomError {
	/// Wraps any error type that implements `std::error::Error + Send + Sync + 'static` into a
	/// `LoomError::Llm` variant.
	///
	/// This function provides a convenient way to convert custom error types into `LoomError`,
	/// allowing for unified error handling within the Loom framework.
	///
	/// # Arguments
	///
	/// * `error` - Any error type that implements `std::error::Error + Send + Sync + 'static`.
	///
	/// # Returns
	///
	/// Returns a `LoomError::Llm` variant containing the boxed input error.
	///
	/// # Example
	///
	/// ```ignore
	/// use your_crate::{LoomError, MyCustomError};
	///
	/// let custom_error = MyCustomError::SomeVariant;
	/// let loom_error = LoomError::from_error(custom_error);
	/// ```
	///
	/// # Type Parameters
	///
	/// * `E`: The type of the error being wrapped. It must implement `std::error::Error + Send +
	///   Sync + 'static`.
	pub fn from_error<E>(error: E) -> Self
	where
		E: Error + Send + Sync + 'static,
	{
		LoomError::Llm(Box::new(error))
	}
}

#[derive(Debug, thiserror::Error)]
pub enum WeaveError {
	#[error("Exceeds max prompt tokens")]
	MaxCompletionTokensIsZero,
	#[error("Bad configuration: {0}")]
	BadConfig(String),
	#[error("Not enough credits to cover cost")]
	NotEnoughCredits,
	#[error("Unknown error: {0}")]
	Unknown(String),
}

#[derive(Debug, thiserror::Error)]
pub enum StorageError {
	#[cfg(feature = "rocksdb")]
	#[error("RocksDb error: {0}")]
	RocksDb(rocksdb::Error),
	#[error("Parsing error")]
	Parsing,
	#[error("Not found")]
	NotFound,
	#[error("Failed to read instance count")]
	FailedToReadInstanceCount,
	#[error("Database error: {0}")]
	DatabaseError(String),
	#[error("Serialization error: {0}")]
	SerializationError(String),
	#[error("Deserialization error: {0}")]
	DeserializationError(String),
	#[error("Internal error: {0}")]
	InternalError(String),
}

/// A helper struct to manage the prompt messages in a deque while keeping track of the tokens
/// added or removed.
pub struct VecPromptMsgsDeque<T: Config, L: Llm<T>> {
	pub tokens: <L as Llm<T>>::Tokens,
	pub inner: VecDeque<<L as Llm<T>>::Request>,
}

impl<T: Config, L: Llm<T>> VecPromptMsgsDeque<T, L> {
	pub fn new() -> Self {
		Self { tokens: L::Tokens::from_u8(0).unwrap(), inner: VecDeque::new() }
	}

	pub fn with_capacity(capacity: usize) -> Self {
		Self { tokens: L::Tokens::from_u8(0).unwrap(), inner: VecDeque::with_capacity(capacity) }
	}

	pub fn push_front(&mut self, msg_reqs: L::Request) {
		let tokens = L::count_tokens(&msg_reqs.to_string()).unwrap_or_default();
		self.tokens = self.tokens.saturating_add(&tokens);
		self.inner.push_front(msg_reqs);
	}

	pub fn push_back(&mut self, msg_reqs: L::Request) {
		let tokens = L::count_tokens(&msg_reqs.to_string()).unwrap_or_default();
		self.tokens = self.tokens.saturating_add(&tokens);
		self.inner.push_back(msg_reqs);
	}

	pub fn append(&mut self, msg_reqs: &mut VecDeque<L::Request>) {
		msg_reqs.iter().for_each(|msg_req| {
			let msg_tokens = L::count_tokens(&msg_req.to_string()).unwrap_or_default();
			self.tokens = self.tokens.saturating_add(&msg_tokens);
		});
		self.inner.append(msg_reqs);
	}

	pub fn truncate(&mut self, len: usize) {
		let mut tokens = L::Tokens::from_u8(0).unwrap();
		for msg_req in self.inner.iter().take(len) {
			let msg_tokens = L::count_tokens(&msg_req.to_string()).unwrap_or_default();
			tokens = tokens.saturating_add(&msg_tokens);
		}
		self.inner.truncate(len);
		self.tokens = tokens;
	}

	pub fn extend(&mut self, msg_reqs: Vec<L::Request>) {
		let mut tokens = L::Tokens::from_u8(0).unwrap();
		for msg_req in &msg_reqs {
			let msg_tokens = L::count_tokens(&msg_req.to_string()).unwrap_or_default();
			tokens = tokens.saturating_add(&msg_tokens);
		}
		self.inner.extend(msg_reqs);
		match self.tokens.checked_add(&tokens) {
			Some(v) => self.tokens = v,
			None => {
				error!("Token overflow");
			},
		}
	}

	pub fn into_vec(self) -> Vec<L::Request> {
		self.inner.into()
	}
}
