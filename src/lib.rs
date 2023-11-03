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

use std::{
	collections::VecDeque,
	fmt::{Debug, Display},
	marker::PhantomData,
	str::FromStr,
};

use async_trait::async_trait;
pub use bounded_integer::BoundedU8;
use num_traits::{
	CheckedAdd, CheckedDiv, FromPrimitive, SaturatingAdd, SaturatingMul, SaturatingSub,
	ToPrimitive, Unsigned,
};
use redis::ToRedisArgs;
use serde::{Deserialize, Serialize};
use storage::TapestryChest;
use tracing::{debug, error, instrument};

pub mod architecture;
pub mod storage;
pub mod types;

pub use storage::TapestryChestHandler;
use types::{LoomError, SummaryModelTokens, WeaveError, ASSISTANT_ROLE, SYSTEM_ROLE};

use crate::types::{PromptModelRequest, PromptModelTokens, WrapperRole};

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
		+ ToRedisArgs
		+ FromStr
		+ Display
		+ Debug
		+ ToString
		+ Serialize
		+ Default
		+ TryFrom<usize>
		+ Unsigned
		+ FromPrimitive
		+ ToPrimitive
		+ CheckedAdd
		+ SaturatingAdd
		+ SaturatingSub
		+ SaturatingMul
		+ CheckedDiv
		+ Ord
		+ Sync
		+ Send;
	/// Type representing the prompt request.
	type Request: Clone + From<ContextMessage<T>> + Send;
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
	/// Calculates the number of tokens in a string.
	///
	/// This may vary depending on the type of tokens used by the LLM. In the case of ChatGPT, can be calculated using the [tiktoken-rs](https://github.com/zurawiki/tiktoken-rs#counting-token-length) crate.
	fn count_tokens(content: String) -> Result<Self::Tokens>;
	/// Prompt LLM with the supplied messages and parameters.
	async fn prompt(
		&self,
		msgs: Vec<Self::Request>,
		params: &Self::Parameters,
	) -> Result<Self::Response>;
	/// Get the maximum number of tokens allowed for the current [`Config::PromptModel`].
	fn get_max_token_limit(&self) -> Self::Tokens {
		let max_tokens = self.max_context_length();
		let token_threshold = Self::Tokens::from_u8(T::TOKEN_THRESHOLD_PERCENTILE.get()).unwrap();
		let tokens = max_tokens.saturating_mul(&token_threshold);
		tokens.checked_div(&Self::Tokens::from_u8(100).unwrap()).unwrap()
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

	/// The LLM to use for generating responses to prompts.
	type PromptModel: Llm<Self>;
	/// The LLM to use for generating summaries of the current [`TapestryFragment`] instance.
	///
	/// This is separate from [`Config::PromptModel`] to allow for a larger model to be used for
	/// generating summaries.
	type SummaryModel: Llm<Self>;
	/// Storage handler interface for storing and retrieving tapestry fragments.
	///
	/// Defaults to [`TapestryChest`]. Using this default requires you to supply the `hostname`,
	/// `port` and `credentials` to connect to your instance.
	type Chest: TapestryChestHandler<Self> = TapestryChest;

	/// Convert [`Config::PromptModel`] to [`Config::SummaryModel`] tokens.
	fn convert_prompt_tokens_to_summary_model_tokens(
		tokens: PromptModelTokens<Self>,
	) -> SummaryModelTokens<Self>;
}

/// Context message that represent a single message in a [`TapestryFragment`] instance.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ContextMessage<T: Config> {
	pub role: WrapperRole,
	pub content: String,
	pub account_id: Option<String>,
	pub timestamp: String,

	_phantom: PhantomData<T>,
}

/// Represents a single part of a conversation containing a list of messages along with other
/// metadata.
///
/// LLM can only hold a limited amount of tokens in a the entire message history/context.
/// The total number of `context_tokens` is tracked when [`Loom::weave`] is executed and if it
/// exceeds the maximum number of tokens allowed for the current GPT [`Config::PromptModel`], then a
/// summary is generated and a new [`TapestryFragment`] instance is created.
#[derive(Debug, Serialize, Default, Clone)]
pub struct TapestryFragment<T: Config> {
	/// Total number of _GPT tokens_ in the `context_messages`.
	pub context_tokens: <T::PromptModel as Llm<T>>::Tokens,
	/// List of [`ContextMessage`]s that represents the message history.
	pub context_messages: Vec<ContextMessage<T>>,
}

impl<T: Config> TapestryFragment<T> {
	/// Add a [`ContextMessage`] to the `context_messages` list.
	///
	/// Also increments the `context_tokens` by the number of tokens in the message.
	fn extend_messages(&mut self, msg: Vec<ContextMessage<T>>) -> Result<()> {
		for m in msg.iter() {
			let tokens = T::PromptModel::count_tokens(m.content.clone())?;
			let new_token_count = match self.context_tokens.checked_add(&tokens) {
				Some(n) => n,
				None =>
					return Err(LoomError::from(WeaveError::BadConfig(format!(
						"Number of tokens exceeds max tokens for model: {}",
						m.content
					)))
					.into()),
			};

			self.context_tokens = new_token_count;
			self.context_messages.push(m.clone());
		}

		Ok(())
	}
}

/// The machine that drives all of the core methods that should be used across any service
/// that needs to prompt LLM and receive a response.
///
/// This is implemented over the [`Config`] trait.
#[async_trait]
pub trait Loom<T: Config> {
	/// Prompt LLM Weaver for a response for [`TapestryId`].
	///
	/// Prompts LLM with the current [`TapestryFragment`] instance and the new `msgs`.
	///
	/// A summary will be generated of the current [`TapestryFragment`] instance if the total number
	/// of tokens in the `context_messages` exceeds the maximum number of tokens allowed for the
	/// current [`Config::PromptModel`] or custom max tokens. This threshold is affected by the
	/// [`Config::TOKEN_THRESHOLD_PERCENTILE`].
	///
	/// # Parameters
	///
	/// - `prompt_config`: The [`Config::PromptModel`] to use for prompting LLM.
	/// - `summary_model_config`: The [`Config::SummaryModel`] to use for generating summaries.
	/// - `tapestry_id`: The [`TapestryId`] to use for storing the [`TapestryFragment`] instance.
	/// - `system`: The system message to be used for the current [`TapestryFragment`] instance.
	/// - `msgs`: The messages to prompt the LLM with.
	#[instrument]
	async fn weave<TID: TapestryId>(
		prompt_config: LlmConfig<T, T::PromptModel>,
		summary_model_config: LlmConfig<T, T::SummaryModel>,
		tapestry_id: TID,
		system: String,
		msgs: Vec<ContextMessage<T>>,
	) -> Result<<<T as Config>::PromptModel as Llm<T>>::Response> {
		let system_ctx_msg =
			Self::build_context_message(WrapperRole::from(SYSTEM_ROLE.to_string()), system, None);
		let sys_req_msg: PromptModelRequest<T> = system_ctx_msg.clone().into();

		// get latest tapestry fragment instance from storage
		let tapestry_fragment = T::Chest::get_tapestry_fragment(tapestry_id.clone(), None)
			.await?
			.unwrap_or_default();

		// number of tokens available according to the configured model or custom max tokens
		let max_tokens_limit = prompt_config.model.get_max_token_limit();

		// allocate space for the system message
		let mut req_msgs = VecDeque::with_capacity(tapestry_fragment.context_messages.len() + 1);
		req_msgs.push_front(sys_req_msg);
		let mut ctx_msgs = VecDeque::from(
			prompt_config
				.model
				.ctx_msgs_to_prompt_requests(&tapestry_fragment.context_messages),
		);
		req_msgs.append(&mut ctx_msgs);

		let mut maybe_summary_text = None;
		let msgs_tokens = msgs.iter().fold(
			PromptModelTokens::<T>::from_u8(0).unwrap(),
			|acc, m: &ContextMessage<T>| {
				let tokens = T::PromptModel::count_tokens(m.content.clone()).unwrap_or_default();
				acc.saturating_add(&tokens)
			},
		);

		// generate summary and start new tapestry instance if context tokens would exceed maximum
		// amount of allowed tokens
		if max_tokens_limit <= tapestry_fragment.context_tokens.saturating_add(&msgs_tokens) {
			maybe_summary_text =
				Some(Self::generate_summary(summary_model_config, &tapestry_fragment).await?);
		}
		let is_summary_generated = maybe_summary_text.is_some();

		let mut tapestry_fragment_to_persist = if let Some(s) = maybe_summary_text {
			let summary_ctx_msg = Self::build_context_message(
				WrapperRole::from(SYSTEM_ROLE.to_string()),
				format!("\n\"\"\"\n {}", s),
				None,
			);

			req_msgs.push_front(summary_ctx_msg.clone().into());

			// keep system and summary messages
			req_msgs.truncate(2);

			TapestryFragment {
				context_tokens: T::PromptModel::count_tokens(s)?,
				context_messages: Vec::from([system_ctx_msg, summary_ctx_msg]),
			}
		} else {
			tapestry_fragment
		};

		let max_tokens = max_tokens_limit
			.saturating_sub(&tapestry_fragment_to_persist.context_tokens)
			.saturating_sub(&msgs_tokens);

		req_msgs.extend(
			prompt_config.model.ctx_msgs_to_prompt_requests(
				&[
					msgs.as_slice(),
					&[Self::build_context_message(
						WrapperRole::from(SYSTEM_ROLE.to_string()),
						format!(
							"Respond with {} words or less",
							prompt_config.model.convert_tokens_to_words(max_tokens)
						),
						None,
					)],
				]
				.concat(),
			),
		);

		let response = prompt_config
			.model
			.prompt(req_msgs.into(), &prompt_config.params)
			.await
			.map_err(|e| {
				error!("Failed to prompt LLM: {}", e);
				e
			})?;

		if let Err(e) = tapestry_fragment_to_persist.extend_messages(
			msgs.into_iter()
				.chain(vec![Self::build_context_message(
					WrapperRole::from(ASSISTANT_ROLE.to_string()),
					response.clone().into().unwrap_or_default(),
					None,
				)])
				.collect(),
		) {
			error!("Failed to extend tapestry fragment: {}", e);
			return Err(e);
		}
		debug!("Saving tapestry fragment: {:?}", tapestry_fragment_to_persist);

		// save tapestry fragment to storage
		// when summarized, the tapestry_fragment will be saved under a new instance
		T::Chest::save_tapestry_fragment(
			tapestry_id,
			tapestry_fragment_to_persist,
			is_summary_generated,
		)
		.await
		.map_err(|e| {
			error!("Failed to save tapestry fragment: {}", e);
			e
		})?;

		Ok(response)
	}

	/// Generates the summary of the current [`TapestryFragment`] instance.
	///
	/// This is will utilize the GPT-4 32K model to generate the summary to allow the maximum number
	/// of possible tokens in the GPT-4 8K model stored in the tapestry fragment.
	///
	/// Returns the summary message as a string.
	async fn generate_summary(
		summary_model_config: LlmConfig<T, T::SummaryModel>,
		tapestry_fragment: &TapestryFragment<T>,
	) -> Result<String> {
		let mut summary_generation_prompt = summary_model_config
			.model
			.ctx_msgs_to_prompt_requests(&tapestry_fragment.context_messages);

		let summary_model_tokens =
			T::convert_prompt_tokens_to_summary_model_tokens(tapestry_fragment.context_tokens);

		let gen_summary_prompt = Self::build_context_message(
			WrapperRole::from(SYSTEM_ROLE.to_string()),
			format!(
				"Generate a summary of the entire adventure so far. Respond with {} words or less",
				summary_model_config.model.convert_tokens_to_words(summary_model_tokens)
			),
			None,
		)
		.into();

		summary_generation_prompt.extend(
			summary_model_config
				.model
				.ctx_msgs_to_prompt_requests(tapestry_fragment.context_messages.as_slice()),
		);
		summary_generation_prompt.push(gen_summary_prompt);

		let res = summary_model_config
			.model
			.prompt(summary_generation_prompt, &summary_model_config.params)
			.await
			.map_err(|e| {
				error!("Failed to prompt LLM: {}", e);
				e
			})?;
		let summary_response_content = res.into();

		Ok(summary_response_content.unwrap_or_default())
	}

	/// Helper method to build a [`ContextMessage`]
	fn build_context_message(
		role: WrapperRole,
		content: String,
		account_id: Option<String>,
	) -> ContextMessage<T> {
		ContextMessage {
			role,
			content,
			account_id,
			timestamp: chrono::Utc::now().to_rfc3339(),
			_phantom: PhantomData,
		}
	}
}
