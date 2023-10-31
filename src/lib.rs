//! Flexible library developed for creating and managing coherent narratives which leverage LLMs
//! (Large Language Models) to generate dynamic responses.
//!
//! Built based on [OpenAI's recommended tactics](https://platform.openai.com/docs/guides/gpt-best-practices/tactic-for-dialogue-applications-that-require-very-long-conversations-summarize-or-filter-previous-dialogue),
//! Loom facilitates extended interactions with any LLM, seamlessly handling conversations
//! that exceed a model's maximum context token limitation.
//!
//! [`Loom`] is the core of this library. It prompts the configured LLM and stores the message
//! history as [`TapestryFragment`] instances. This trait is highly configurable through the
//! [`Config`] trait to support a wide range of use cases.
//!
//! You must implement the [`Config`] trait, which defines the necessary types and methods needed by
//! [`Loom`].
//!
//! If you are using the default implementation of [`Config::TapestryChest`], it is expected that a
//! Redis instance is running and that the following environment variables are set:
//!
//! - `REDIS_PROTOCOL`
//! - `REDIS_HOST`
//! - `REDIS_PORT`
//! - `REDIS_PASSWORD`
//!
//! Should there be a need to integrate a distinct storage backend, you have the flexibility to
//! create a custom handler by implementing the [`TapestryChestHandler`] trait and injecting it
//! into the [`Config::TapestryChest`] associated type.
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
use bounded_integer::BoundedU8;
use num_traits::{
	CheckedAdd, CheckedDiv, FromPrimitive, SaturatingAdd, SaturatingMul, SaturatingSub, Unsigned,
};
use redis::ToRedisArgs;
use serde::{Deserialize, Serialize};
use storage::TapestryChest;
use tracing::{debug, error, instrument};

pub mod storage;
pub mod types;

pub use storage::TapestryChestHandler;
use types::{LoomError, WeaveError, ASSISTANT_ROLE, SYSTEM_ROLE};

use crate::types::WrapperRole;

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

pub trait Llm<T: Config>: Sized + PartialEq + Eq + Clone + Debug + Copy + Send + Sync {
	/// The maximum number of tokens that can be processed at once by an LLM model.
	fn max_context_length(&self) -> T::Tokens;

	/// Get the model name.
	///
	/// This is used for logging purposes but also can be used to fetch a specific model based on
	/// `&self`. For example, the model passed to [`Loom::weave`] can be represented as an enum with
	/// a multitude of variants, each representing a different model.
	fn name(&self) -> &'static str;

	/// Calculates the number of tokens in a string.
	///
	/// This may vary depending on the type of tokens used by the LLM. In the case of ChatGPT,
	/// each token represents roughly 75% of a word and can be calculated using the [tiktoken-rs](https://github.com/zurawiki/tiktoken-rs#counting-token-length) crate.
	fn count_tokens(content: String) -> Result<T::Tokens>;
}

/// A trait consisting of the main configuration needed to implement [`Loom`].
#[async_trait]
pub trait Config: Debug + Sized + Clone + Default + Send + Sync + 'static {
	/// Number between 0 and 100. Represents the percentile of the maximum number of tokens allowed
	/// for the current [`Config::Model`] before a summary is generated.
	///
	/// Defaults to `85%`
	const TOKEN_THRESHOLD_PERCENTILE: BoundedU8<0, 100> = BoundedU8::new(85).unwrap();
	/// Token to word ratio.
	///
	/// Every token equates to 75% of a word.
	const TOKEN_WORD_RATIO: BoundedU8<0, 100> = BoundedU8::new(75).unwrap();

	/// Type representing the prompt request.
	type PromptRequest: Clone + From<ContextMessage<Self>> + Send;
	/// Type representing the response to a prompt.
	type PromptResponse: Clone + Into<Option<String>> + Send;
	/// Type representing the parameters for a prompt.
	type PromptParameters: Debug + Clone + Send + Sync;
	/// Tokens are an LLM concept which represents pieces of words. For example, each ChatGPT token
	/// represents roughly 75% of a word.
	///
	/// This type is used primarily for tracking the number of tokens in a [`TapestryFragment`] and
	/// counting the number of tokens in a string.
	///
	/// This type is configurable to allow for different types of tokens to be used. For example,
	/// [`u16`] can be used to represent the number of tokens in a string. This is in line with
	type Tokens: Copy
		+ ToRedisArgs
		+ FromStr
		+ Display
		+ Debug
		+ ToString
		+ Default
		+ TryFrom<usize>
		+ Unsigned
		+ FromPrimitive
		+ CheckedAdd
		+ SaturatingAdd
		+ SaturatingSub
		+ SaturatingMul
		+ CheckedDiv
		+ Ord
		+ Sync
		+ Send;
	/// The LLM to use for generating responses to prompts.
	type Model: Llm<Self>;
	/// The LLM to use for generating summaries of the current [`TapestryFragment`] instance.
	///
	/// This is separate from [`Config::Model`] to allow for a larger model to be used for
	/// generating summaries.
	type SummaryModel: Llm<Self>;
	/// Storage handler interface for storing and retrieving tapestry fragments.
	///
	/// Defaults to [`TapestryChest`]. Using this default requires you to supply the `hostname`,
	/// `port` and `credentials` to connect to your instance.
	type TapestryChest: TapestryChestHandler<Self> = TapestryChest;

	/// The action to query an LLM with the supplied messages and parameters.
	async fn prompt(
		model: impl Llm<Self>,
		msgs: Vec<Self::PromptRequest>,
		params: &Self::PromptParameters,
	) -> Result<Self::PromptResponse>;
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
/// exceeds the maximum number of tokens allowed for the current GPT [`Config::Model`], then a
/// summary is generated and a new [`TapestryFragment`] instance is created.
#[derive(Debug, Serialize, Default, Clone)]
pub struct TapestryFragment<T: Config> {
	/// Total number of _GPT tokens_ in the `context_messages`.
	pub context_tokens: T::Tokens,
	/// List of [`ContextMessage`]s that represents the message history.
	pub context_messages: Vec<ContextMessage<T>>,
}

impl<T: Config> TapestryFragment<T> {
	/// Add a [`ContextMessage`] to the `context_messages` list.
	///
	/// Also increments the `context_tokens` by the number of tokens in the message.
	fn extend_messages(&mut self, msg: Vec<ContextMessage<T>>) -> Result<()> {
		for m in msg.iter() {
			let tokens = T::Model::count_tokens(m.content.clone())?;
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
	/// Prompt Loom for a response for [`TapestryId`].
	///
	/// Prompts LLM with the current [`TapestryFragment`] instance and the new `msgs`.
	///
	/// A summary will be generated of the current [`TapestryFragment`] instance if the total number
	/// of tokens in the `context_messages` exceeds the maximum number of tokens allowed for the
	/// current [`Config::Model`] or custom max tokens. This threshold is affected by the
	/// [`Config::TOKEN_THRESHOLD_PERCENTILE`].
	///
	/// # Parameters
	///
	/// - `tapestry_id`: The [`TapestryId`] to prompt and save context messages to.
	/// - `system`: The system message to prompt LLM with.
	///  the current [`Config::Model`].
	/// - `msgs`: The list of [`ContextMessage`]s to prompt LLM with.
	/// - `prompt_params`: The [`Config::PromptParameters`] to use when prompting LLM.
	#[instrument]
	async fn weave<TID: TapestryId>(
		model: T::Model,
		summary_model: T::SummaryModel,
		tapestry_id: TID,
		system: String,
		msgs: Vec<ContextMessage<T>>,
		prompt_params: T::PromptParameters,
	) -> Result<T::PromptResponse> {
		let system_ctx_msg =
			Self::build_context_message(WrapperRole::from(SYSTEM_ROLE.to_string()), system, None);
		let sys_req_msg: T::PromptRequest = system_ctx_msg.clone().into();

		// get latest tapestry fragment instance from storage
		let tapestry_fragment = T::TapestryChest::get_tapestry_fragment(tapestry_id.clone(), None)
			.await?
			.unwrap_or_default();

		// number of tokens available according to the configured model or custom max tokens
		let max_tokens_limit = Self::get_max_token_limit(model);

		// allocate space for the system message
		let mut req_msgs = VecDeque::with_capacity(tapestry_fragment.context_messages.len() + 1);
		req_msgs.push_front(sys_req_msg);
		let mut ctx_msgs =
			VecDeque::from(Self::ctx_msgs_to_prompt_requests(&tapestry_fragment.context_messages));
		req_msgs.append(&mut ctx_msgs);

		let mut maybe_summary_text: Option<String> = None;
		let msgs_tokens = msgs.iter().fold::<T::Tokens, _>(
			T::Tokens::from_u8(0).unwrap(),
			|acc: T::Tokens, m: &ContextMessage<T>| {
				let tokens: T::Tokens =
					T::Model::count_tokens(m.content.clone()).unwrap_or_default();
				acc.saturating_add(&tokens)
			},
		);

		// generate summary and start new tapestry instance if context tokens would exceed maximum
		// amount of allowed tokens
		if max_tokens_limit <= tapestry_fragment.context_tokens.saturating_add(&msgs_tokens) {
			maybe_summary_text = Some(
				Self::generate_summary(
					summary_model,
					&max_tokens_limit,
					&tapestry_fragment,
					&prompt_params,
				)
				.await?,
			);
		}
		let is_summary_generated = maybe_summary_text.is_some();

		let mut tapestry_fragment_to_persist = if let Some(s) = maybe_summary_text {
			let summary_ctx_msg = Self::build_context_message(
				WrapperRole::from(SYSTEM_ROLE.to_string()),
				format!("\n\"\"\"\n {}", s),
				None,
			);
			let summary_req_msg: T::PromptRequest = summary_ctx_msg.clone().into();

			req_msgs.push_front(summary_req_msg.clone());

			// keep system and summary messages
			req_msgs.truncate(2);

			TapestryFragment {
				context_tokens: T::Model::count_tokens(s)?,
				context_messages: Vec::from([system_ctx_msg, summary_ctx_msg]),
			}
		} else {
			tapestry_fragment
		};

		let max_tokens = max_tokens_limit
			.saturating_sub(&tapestry_fragment_to_persist.context_tokens)
			.saturating_sub(&msgs_tokens);

		req_msgs.extend(Self::ctx_msgs_to_prompt_requests(
			&[
				msgs.as_slice(),
				&[Self::build_context_message(
					WrapperRole::from(SYSTEM_ROLE.to_string()),
					format!(
						"Respond with {} words or less",
						Self::convert_tokens_to_words(max_tokens)
					),
					None,
				)],
			]
			.concat(),
		));

		let response = T::prompt(model, req_msgs.into(), &prompt_params).await.map_err(|e| {
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
		T::TapestryChest::save_tapestry_fragment(
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
		model: impl Llm<T>,
		num_tokens: &T::Tokens,
		tapestry_fragment: &TapestryFragment<T>,
		prompt_params: &T::PromptParameters,
	) -> Result<String> {
		let model_max_tokens = Self::get_max_token_limit(model);
		if num_tokens > &model_max_tokens {
			return Err(LoomError::from(WeaveError::BadConfig(format!(
				"Number of tokens cannot exceed model's max tokens ({}): {}",
				model.name(),
				model.max_context_length()
			)))
			.into());
		}

		let mut summary_generation_prompt =
			Self::ctx_msgs_to_prompt_requests(&tapestry_fragment.context_messages);

		let gen_summary_prompt = Self::build_context_message(
			WrapperRole::from(SYSTEM_ROLE.to_string()),
			format!(
				"Generate a summary of the entire adventure so far. Respond with {} words or less",
				Self::convert_tokens_to_words(*num_tokens)
			),
			None,
		)
		.into();

		summary_generation_prompt.push(gen_summary_prompt);

		let res =
			T::prompt(model, summary_generation_prompt, prompt_params).await.map_err(|e| {
				error!("Failed to prompt LLM: {}", e);
				e
			})?;
		let summary_response_content = res.into();

		Ok(summary_response_content.unwrap_or_default())
	}

	fn ctx_msgs_to_prompt_requests(msgs: &[ContextMessage<T>]) -> Vec<T::PromptRequest> {
		msgs.iter().map(|m| m.clone().into()).collect()
	}

	/// Get the maximum number of tokens allowed for the current [`Config::Model`].
	fn get_max_token_limit(model: impl Llm<T>) -> T::Tokens {
		let max_tokens = model.max_context_length();
		let token_threshold: T::Tokens =
			T::Tokens::from_u8(T::TOKEN_THRESHOLD_PERCENTILE.get()).unwrap();
		let tokens = max_tokens.saturating_mul(&token_threshold);
		tokens.checked_div(&T::Tokens::from_u8(100).unwrap()).unwrap()
	}

	fn convert_tokens_to_words(tokens: T::Tokens) -> T::Tokens {
		tokens.saturating_mul(&T::Tokens::from_u8(T::TOKEN_WORD_RATIO.get()).unwrap()) /
			T::Tokens::from_u8(100).unwrap()
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
