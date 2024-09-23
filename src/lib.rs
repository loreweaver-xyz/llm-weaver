//! Flexible library developed for creating and managing coherent narratives which leverage LLMs
//! (Large Language Models) to generate dynamic responses.
//!
//! Built based on [OpenAI's recommended tactics](https://platform.openai.com/docs/guides/gpt-best-practices/tactic-for-dialogue-applications-that-require-very-long-conversations-summarize-or-filter-previous-dialogue),
//! LLM Weaver facilitates extended interactions with any LLM, seamlessly handling conversations
//! that exceed a model's maximum context token limitation.
//!
//! [`Loom`] is the core of this library. It prompts the configured LLM and stores the message
//! history as [`TapestryFragment`] instances. This trait is highly configurable through the
//! [`Config`] trait to support a wide range of use cases.
//!
//! # Nomenclature
//!
//! - **Tapestry**: A collection of [`TapestryFragment`] instances.
//! - **TapestryFragment**: A single part of a conversation containing a list of messages along with
//!   other metadata.
//! - **ContextMessage**: Represents a single message in a [`TapestryFragment`] instance.
//! - **Loom**: The machine that drives all of the core methods that should be used across any
//!   service that needs to prompt LLM and receive a response.
//! - **LLM**: Language Model.
//!
//! # Architecture
//!
//! Please refer to the [`architecture::Diagram`] for a visual representation of the core
//! components of this library.
//!
//! # Usage
//!
//! You must implement the [`Config`] trait, which defines the necessary types and methods needed by
//! [`Loom`].
//!
//! This library uses Redis as the default storage backend for storing [`TapestryFragment`]. It is
//! expected that a Redis instance is running and that the following environment variables are set:
//!
//! - `REDIS_PROTOCOL`
//! - `REDIS_HOST`
//! - `REDIS_PORT`
//! - `REDIS_PASSWORD`
//!
//! Should there be a need to integrate a distinct storage backend, you have the flexibility to
//! create a custom handler by implementing the [`TapestryChestHandler`] trait and injecting it
//! into the [`Config::Chest`] associated type.
#![feature(async_closure)]
#![feature(associated_type_defaults)]
#![feature(more_qualified_paths)]
#![feature(const_option)]
#![feature(anonymous_lifetime_in_impl_trait)]
#![feature(once_cell_try)]

use std::{
	fmt::{Debug, Display},
	marker::PhantomData,
	str::FromStr,
};

use async_trait::async_trait;
pub use bounded_integer::BoundedU8;
use num_traits::{
	CheckedAdd, CheckedDiv, CheckedMul, CheckedSub, FromPrimitive, SaturatingAdd, SaturatingMul,
	SaturatingSub, ToPrimitive, Unsigned,
};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use tracing::trace;

pub mod architecture;
pub mod loom;
pub mod storage;
pub mod types;

#[cfg(test)]
mod mock;
#[cfg(test)]
mod tests;

pub use storage::TapestryChestHandler;
use types::{LoomError, SummaryModelTokens, WeaveError};

use crate::types::{PromptModelTokens, WrapperRole};

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

/// Represents a unique identifier for any arbitrary entity.
///
/// This trait provides a method for generating a standardized key, which can be utilized across
/// various implementations in the library, such as the [`TapestryChest`] implementation for storing
/// keys in redis using the `base_key` method.
///
/// ```ignore
/// use loom::{TapestryId, Get};
/// use std::fmt::{Debug, Display};
///
/// struct MyTapestryId {
///     id: String,
///     sub_id: String,
///     // ...
/// }
///
/// impl TapestryId for MyTapestryId {
///     fn base_key(&self) -> String {
///         format!("{}:{}", self.id, self.sub_id)
///     }
/// }
/// ```
pub trait TapestryId: Debug + Clone + Send + Sync + 'static {
	/// Returns the base key.
	///
	/// This method should produce a unique string identifier, that will serve as a key for
	/// associated objects or data within [`TapestryChestHandler`] implementations.
	fn base_key(&self) -> String;
}

#[derive(Debug)]
pub struct LlmConfig<T: Config, L: Llm<T>> {
	pub model: L,
	pub params: L::Parameters,
}

#[async_trait]
pub trait Llm<T: Config>:
	Default + Sized + PartialEq + Eq + Clone + Debug + Copy + Send + Sync
{
	/// Token to word ratio.
	///
	/// Defaults to `75%`
	const TOKEN_WORD_RATIO: BoundedU8<0, 100> = BoundedU8::new(75).unwrap();

	/// Tokens are an LLM concept which represents pieces of words. For example, each ChatGPT token
	/// represents roughly 75% of a word.
	///
	/// This type is used primarily for tracking the number of tokens in a [`TapestryFragment`] and
	/// counting the number of tokens in a string.
	///
	/// This type is configurable to allow for different types of tokens to be used. For example,
	/// [`u16`] can be used to represent the number of tokens in a string.
	type Tokens: Copy
		+ ToString
		+ FromStr
		+ Display
		+ Debug
		+ ToString
		+ Serialize
		+ DeserializeOwned
		+ Default
		+ TryFrom<usize>
		+ Unsigned
		+ FromPrimitive
		+ ToPrimitive
		+ std::iter::Sum
		+ CheckedAdd
		+ CheckedSub
		+ SaturatingAdd
		+ SaturatingSub
		+ SaturatingMul
		+ CheckedDiv
		+ CheckedMul
		+ Ord
		+ Sync
		+ Send;
	/// Type representing the prompt request.
	type Request: Clone + From<ContextMessage<T>> + Display + Send;
	/// Type representing the response to a prompt.
	type Response: Clone + Into<Option<String>> + Send;
	/// Type representing the parameters for a prompt.
	type Parameters: Debug + Clone + Send + Sync;

	/// The maximum number of tokens that can be processed at once by an LLM model.
	fn max_context_length(&self) -> Self::Tokens;
	/// Get the model name.
	///
	/// This is used for logging purposes but also can be used to fetch a specific model based on
	/// `&self`. For example, the model passed to [`Loom::weave`] can be represented as an enum with
	/// a multitude of variants, each representing a different model.
	fn name(&self) -> &'static str;
	/// Alias for the model.
	///
	/// Can be used for any unforseen use cases where the model name is not sufficient.
	fn alias(&self) -> &'static str;
	/// Calculates the number of tokens in a string.
	///
	/// This may vary depending on the type of tokens used by the LLM. In the case of ChatGPT, can be calculated using the [tiktoken-rs](https://github.com/zurawiki/tiktoken-rs#counting-token-length) crate.
	fn count_tokens(content: &str) -> Result<Self::Tokens>;
	/// Prompt LLM with the supplied messages and parameters.
	async fn prompt(
		&self,
		is_summarizing: bool,
		prompt_tokens: Self::Tokens,
		msgs: Vec<Self::Request>,
		params: &Self::Parameters,
		max_tokens: Self::Tokens,
	) -> Result<Self::Response>;
	/// Compute cost of a message based on model.
	fn compute_cost(&self, prompt_tokens: Self::Tokens, response_tokens: Self::Tokens) -> f64;
	/// Calculate the upperbound of tokens allowed for the current [`Config::PromptModel`] before a
	/// summary is generated.
	///
	/// This is calculated by multiplying the maximum context length (tokens) for the current
	/// [`Config::PromptModel`] by the [`Config::TOKEN_THRESHOLD_PERCENTILE`] and dividing by 100.
	fn get_max_prompt_token_limit(&self) -> Self::Tokens {
		let max_context_length = self.max_context_length();
		let token_threshold = Self::Tokens::from_u8(T::TOKEN_THRESHOLD_PERCENTILE.get()).unwrap();
		let tokens = match max_context_length.checked_mul(&token_threshold) {
			Some(tokens) => tokens,
			None => max_context_length,
		};

		tokens.checked_div(&Self::Tokens::from_u8(100).unwrap()).unwrap()
	}
	/// Get optional max completion token limit.
	fn get_max_completion_token_limit(&self) -> Option<Self::Tokens> {
		None
	}
	/// [`ContextMessage`]s to [`Llm::Request`] conversion.
	fn ctx_msgs_to_prompt_requests(&self, msgs: &[ContextMessage<T>]) -> Vec<Self::Request> {
		msgs.iter().map(|m| m.clone().into()).collect()
	}
	/// Convert tokens to words.
	///
	/// In the case of ChatGPT, each token represents roughly 75% of a word.
	fn convert_tokens_to_words(&self, tokens: Self::Tokens) -> Self::Tokens {
		tokens.saturating_mul(&Self::Tokens::from_u8(Self::TOKEN_WORD_RATIO.get()).unwrap()) /
			Self::Tokens::from_u8(100).unwrap()
	}
}

/// A trait consisting of the main configuration needed to implement [`Loom`].
#[async_trait]
pub trait Config: Debug + Sized + Clone + Default + Send + Sync + 'static {
	/// Number between 0 and 100. Represents the percentile of the maximum number of tokens allowed
	/// for the current [`Config::PromptModel`] before a summary is generated.
	///
	/// Defaults to `85%`
	const TOKEN_THRESHOLD_PERCENTILE: BoundedU8<0, 100> = BoundedU8::new(85).unwrap();
	/// Ensures that the maximum completion tokens is at least the minimum response length.
	///
	/// If the maximum completion tokens is less than the minimum response length, a summary
	/// will be generated and a new tapestry fragment will be created.
	const MINIMUM_RESPONSE_LENGTH: u64;

	/// The LLM to use for generating responses to prompts.
	type PromptModel: Llm<Self>;
	/// The LLM to use for generating summaries of the current [`TapestryFragment`] instance.
	///
	/// This is separate from [`Config::PromptModel`] to allow for a larger model to be used for
	/// generating summaries.
	type SummaryModel: Llm<Self>;
	/// Storage handler interface for storing and retrieving tapestry fragments.
	///
	/// You can optionally enable the `redis` or `rocksdb` features to use the default storage
	/// implementations for these storage backends.
	type Chest: TapestryChestHandler<Self>;

	/// Convert [`Config::PromptModel`] to [`Config::SummaryModel`] tokens.
	fn convert_prompt_tokens_to_summary_model_tokens(
		tokens: PromptModelTokens<Self>,
	) -> SummaryModelTokens<Self>;
}

/// Context message that represent a single message in a [`TapestryFragment`] instance.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ContextMessage<T: Config> {
	pub role: WrapperRole,
	pub content: String,
	pub account_id: Option<String>,
	pub timestamp: String,

	_phantom: PhantomData<T>,
}

impl<T: Config> ContextMessage<T> {
	/// Create a new `ContextMessage` instance.
	pub fn new(
		role: WrapperRole,
		content: String,
		account_id: Option<String>,
		timestamp: String,
	) -> Self {
		Self { role, content, account_id, timestamp, _phantom: PhantomData }
	}
}

/// Represents a single part of a conversation containing a list of messages along with other
/// metadata.
///
/// LLM can only hold a limited amount of tokens in a the entire message history/context.
/// The total number of `context_tokens` is tracked when [`Loom::weave`] is executed and if it
/// exceeds the maximum number of tokens allowed for the current GPT [`Config::PromptModel`], then a
/// summary is generated and a new [`TapestryFragment`] instance is created.
#[derive(Debug, Serialize, Deserialize, Default, PartialEq, Clone)]
pub struct TapestryFragment<T: Config> {
	/// Total number of _GPT tokens_ in the `context_messages`.
	pub context_tokens: <T::PromptModel as Llm<T>>::Tokens,
	/// List of [`ContextMessage`]s that represents the message history.
	pub context_messages: Vec<ContextMessage<T>>,
}

impl<T: Config> TapestryFragment<T> {
	fn new() -> Self {
		Self::default()
	}

	/// Add a [`ContextMessage`] to the `context_messages` list.
	///
	/// Also increments the `context_tokens` by the number of tokens in the message.
	fn push_message(&mut self, msg: ContextMessage<T>) -> Result<()> {
		let tokens = T::PromptModel::count_tokens(&msg.content)?;
		let new_token_count = self.context_tokens.checked_add(&tokens).ok_or_else(|| {
			LoomError::from(WeaveError::BadConfig(
				"Number of tokens exceeds max tokens for model".to_string(),
			))
		})?;

		trace!("Pushing message: {:?}, new token count: {}", msg, new_token_count);

		self.context_tokens = new_token_count;
		self.context_messages.push(msg);
		Ok(())
	}

	/// Add a [`ContextMessage`] to the `context_messages` list.
	///
	/// Also increments the `context_tokens` by the number of tokens in the message.
	fn extend_messages(&mut self, msgs: Vec<ContextMessage<T>>) -> Result<()> {
		let total_new_tokens = msgs
			.iter()
			.map(|m| T::PromptModel::count_tokens(&m.content).unwrap())
			.collect::<Vec<_>>();

		let sum: PromptModelTokens<T> = total_new_tokens
			.iter()
			.fold(PromptModelTokens::<T>::default(), |acc, x| acc.saturating_add(x));

		trace!("Extending messages with token sum: {}", sum);

		let new_token_count = self.context_tokens.checked_add(&sum).ok_or_else(|| {
			LoomError::from(WeaveError::BadConfig(
				"Number of tokens exceeds max tokens for model".to_string(),
			))
		})?;

		// Update the token count and messages only if all checks pass
		self.context_tokens = new_token_count;
		for m in msgs {
			self.context_messages.push(m);
		}

		Ok(())
	}
}
