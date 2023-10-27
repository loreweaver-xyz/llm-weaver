//! Flexible out-of-the-box library developed for creating and managing coherent narratives,
//! leveraging OpenAI's ChatGPT as the underlying LLM (Language Model Model).
//!
//! Built based on [OpenAI's recommended tactics](https://platform.openai.com/docs/guides/gpt-best-practices/tactic-for-dialogue-applications-that-require-very-long-conversations-summarize-or-filter-previous-dialogue),
//! Loreweaver facilitates extended interactions with ChatGPT, seamlessly handling conversations
//! that exceed the model's maximum context token limitation.
//!
//! Central to Loreweaver is the [`TapestryFragment`] struct, which archives dialogue history.
//! The [`TapestryChestHandler`] defines the interface for storing and retrieving
//! [`TapestryFragment`]s from a storage backend. By default, Loreweaver implements the
//! [`TapestryChestHandler`] trait, employing Redis for backend storage in its default
//! configuration.
//!
//! Effective utilization of Loreweaver necessitates the implementation of the [`Config`] trait,
//! ensuring the provision of mandatory types not preset by default.
//!
//! For immediate use of this library you must:
//! - open an account on [openai](https://platform.openai.com/) with an api key
//! - run a Redis instance
//!
//! You must set the following environment variables:
//!
//! - `OPENAI_API_KEY`
//! - `REDIS_PROTOCOL`
//! - `REDIS_HOST`
//! - `REDIS_PORT`
//! - `REDIS_PASSWORD`
//!
//! Should there be a need to integrate a distinct storage backend, you have the flexibility to
//! create a custom handler by implementing the [`TapestryChestHandler`] trait and injecting it
//! into the [`Config::TapestryChest`] associated type.
//!
//! This example demonstrates how to integrate Loreweaver into a chatbot service, showcasing the
//! generation of dynamic responses and maintenance of conversation history. The `Thread` struct,
//! uniquely identifying different user conversation threads, must implement the [`TapestryId`]
//! trait to enable storage and retrieval of [`TapestryFragment`]s.
//!
//! # Example
//!
//! ```ignore
//! // Importing necessary modules and traits from the loreweaver library and other dependencies.
//! use loreweaver::{Loreweaver, Config, TapestryId, ContextMessage};
//! use std::fmt::Debug;
//!
//! // Core struct for the ChatBot which will be used to implement the Config trait.
//! #[derive(Default, Debug)]
//! pub struct ChatBot;
//!
//! // Implementation of a getter to specify which GPT model to use.
//! pub struct GPTModel;
//! impl Get<Models> for GPTModel {
//!     fn get() -> Models {
//!         Models::GPT4 // Specifies using GPT-4 model.
//!     }
//! }
//!
//! // Implementing the Config trait for the ChatBot with specific parameters.
//! impl Config for ChatBot {
//!     // ChatGPT behaviours parameters.
//!     const TEMPERATURE: f32 = 0.7;
//!     const PRESENCE_PENALTY: f32 = 0.3;
//!     const FREQUENCY_PENALTY: f32 = 0.3;
//!
//!     // ChatGPT model to use.
//!     type Model = GPTModel;
//! }
//!
//! // Definition of a custom Thread structure to handle unique conversation threads.
//! #[derive(Debug, Clone)]
//! struct Thread {
//!     server_id: u64, // Identifier for the server.
//!     id: u64, // Unique identifier for the thread within the server.
//! }
//!
//! // Implementation of the TapestryId trait for the Thread structure, enabling unique identification for storage.
//! impl loreweaver::TapestryId for Thread {
//!     fn base_key(&self) -> String {
//!         format!("{}:{}", self.server_id, self.id) // Represents the unique key for the thread in storage.
//!     }
//! }
//!
//! // The main async function where the ChatBot is used.
//! #[tokio::main]
//! async fn main() {
//!   // Unique conversation thread.
//!   let tapestry_id = Thread {
//!     server_id: 1,
//!     id: 1,
//!   };
//!   // Initial system message.
//!   let system = "You are a Chat bot called Bob that always responds cleverly but is also random at times.".to_string();
//!   // No override for maximum context tokens.
//!   let override_max_context_tokens = None;
//!   // Account id of the user
//!   let account_id = Some("user1".to_string());
//!
//!   // Weave a response!
//!   let res = Loreweaver::<ChatBot>::weave(
//!     tapestry_id,
//!     system,
//!     override_max_context_tokens,
//!     vec![ContextMessage {
//! 		role: WrapperRole::from("user".to_string()),
//! 		msg: "Alice: What is your name?",
//! 		account_id: 1,
//! 		timestamp: chrono::Utc::now().to_rfc3339(),
//! 	}],
//!   )
//!   .await
//!   .unwrap();
//!
//!   println!("Response: {}", res);
//!
//!   let res = Loreweaver::<ChatBot>::weave(
//!     tapestry_id,
//!     system,
//!     override_max_context_tokens,
//!    vec![ContextMessage {
//! 		role: WrapperRole::from("user".to_string()),
//! 		msg: "Alice: What was the last question I asked you?",
//! 		account_id: 1,
//! 		timestamp: chrono::Utc::now().to_rfc3339(),
//! 	}],
//!   )
//!   .await
//!   .unwrap();
//!
//!   println!("Response: {}", res);
//! }
//! ```
#![feature(async_closure)]
#![feature(associated_type_defaults)]
#![feature(more_qualified_paths)]

use std::{
	collections::VecDeque,
	fmt::{Debug, Display},
	marker::PhantomData,
};

use async_openai::{
	config::OpenAIConfig,
	error::OpenAIError,
	types::{
		ChatCompletionRequestMessage, ChatCompletionRequestMessageArgs,
		CreateChatCompletionRequestArgs, CreateChatCompletionResponse, Role,
	},
};
use async_trait::async_trait;
use lazy_static::lazy_static;
use models::Tokens;
use serde::{Deserialize, Serialize};
use storage::{StorageError, TapestryChest};
use tokio::sync::RwLock;
use tracing::{debug, error, instrument};

use models::{Models, Token};

pub mod storage;
pub mod types;

pub use storage::TapestryChestHandler;
use types::{ASSISTANT_ROLE, F32, SYSTEM_ROLE};

use crate::types::WrapperRole;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

pub trait Get<T> {
	fn get() -> T;
}

/// Represents a unique identifier for any arbitrary entity.
///
/// The `TapestryId` trait abstractly represents identifiers, providing a method for
/// generating a standardized key, which can be utilized across various implementations
/// in the library, such as the [`TapestryChestHandler`].
///
/// ```ignore
/// use loreweaver::{TapestryId, Get};
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
	/// associated objects or data within [`TapestryChestHandler`].
	fn base_key(&self) -> String;
}

/// A trait consisting of the main configuration parameters for [`Loreweaver`].
pub trait Config {
	/// Number between 0 and 1. Percentage of the maximum number of tokens that can be in a
	/// [`TapestryFragment`] instance before a summary is generated and a new [`TapestryFragment`]
	/// instance is created.
	///
	/// Defaults to `0.15`
	const TOKEN_THRESHOLD_PERCENTILE: F32 = 0.85;
	/// Sampling temperature, between 0 and 1. Higher values like 0.8 will make the output more
	/// random, while lower values like 0.2 will make it more focused and deterministic. If set to
	/// 0, the model will use log probability to automatically increase the temperature until
	/// certain thresholds are hit.
	///
	/// Defaults to `0.0`
	const TEMPRATURE: F32 = 0.0;
	/// Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they
	/// appear in the text so far, increasing the model's likelihood to talk about new topics.
	///
	/// Defaults to `0.0`
	const PRESENCE_PENALTY: F32 = 0.0;
	/// Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing
	/// frequency in the text so far, decreasing the model's likelihood to repeat the same line
	/// verbatim.
	///
	/// Defaults to `0.0`
	const FREQUENCY_PENALTY: F32 = 0.0;

	/// Getter for GPT model to use.
	///
	/// Defaults to [`models::DefaultModel`]
	type Model: Get<Models> = models::DefaultModel;
	/// Storage handler interface for storing and retrieving tapestry fragments.
	///
	/// [`Loreweaver`] does not care how you store the data and retrieve your data.
	///
	/// Defaults to [`TapestryChest`]. Using this default requires you to supply the `hostname`,
	/// `port` and `credentials` to connect to your instance.
	type TapestryChest: TapestryChestHandler = TapestryChest;
}

/// Context message that represent a single message in a [`TapestryFragment`] instance.
#[derive(Default, Clone, Debug, Serialize, Deserialize)]
pub struct ContextMessage {
	pub role: WrapperRole,
	pub content: String,
	pub account_id: Option<String>,
	pub timestamp: String,
}

/// Represents a single part of a conversation containing a list of messages along with other
/// metadata.
///
/// ChatGPT can only hold a limited amount of tokens in a the entire message history/context.
/// Therefore, at every [`Loom::prompt`] execution, the total number of `context_tokens` is tracked
/// and if it exceeds the maximum number of tokens allowed for the current GPT [`Models`], then we
/// must generate a summary of the `context_messages` and use that as the starting point for the
/// next `TapestryFragment`.
#[derive(Debug, Serialize, Deserialize, Default, Clone)]
pub struct TapestryFragment {
	/// Total number of _GPT tokens_ in the `context_messages`.
	pub context_tokens: Tokens,
	/// List of [`ContextMessage`]s that represents the message history.
	pub context_messages: Vec<ContextMessage>,
}

impl TapestryFragment {
	/// Add a [`ContextMessage`] to the `context_messages` list.
	///
	/// Also increments the `context_tokens` by the number of tokens in the message.
	fn extend_messages(&mut self, msg: Vec<ContextMessage>) {
		msg.iter().for_each(|msg| {
			self.context_tokens += msg.content.count_tokens();
			self.context_messages.push(msg.clone());
		});
	}
}

/// A trait that defines all of the methods that [`Loreweaver`] implements.
///
/// This is the machine that drives all of the core methods that should be used across any service
/// that needs to prompt ChatGPT and receive a response.
///
/// The implementations should handle all form of validation and usage tracking all while
/// abstracting the logic from the services calling them.
#[async_trait]
pub trait Loom<T: Config> {
	/// Represents the request message type used to prompt an LLM.
	type RequestMessages: IntoIterator;
	/// Represents the response type returned by the LLM library.
	type Response;

	/// Prompt Loreweaver for a response for [`TapestryId`].
	///
	/// Prompts ChatGPT with the current [`TapestryFragment`] instance and the new `msgs`.
	///
	/// A summary will be generated of the current [`TapestryFragment`] instance if the total number
	/// of tokens in the `context_messages` exceeds the maximum number of tokens allowed for the
	/// current [`Models`] or custom max tokens. This threshold is affected by the
	/// [`Config::TOKEN_THRESHOLD_PERCENTILE`].
	///
	/// # Parameters
	///
	/// - `tapestry_id`: The [`TapestryId`] to prompt and save context messages to.
	/// - `system`: The system message to prompt ChatGPT with.
	///  the current [`Models`].
	/// - `msgs`: The list of [`ContextMessage`]s to prompt ChatGPT with.
	async fn weave<TID: TapestryId>(
		tapestry_id: TID,
		system: String,
		msgs: Vec<ContextMessage>,
	) -> Result<Self::Response>;

	/// Build the message/messages to prompt ChatGPT with.
	fn build_req_msgs(msgs: &Vec<ContextMessage>) -> Result<Self::RequestMessages>;

	/// The action to query ChatGPT with the supplied configurations and messages.
	async fn prompt(
		msgs: Self::RequestMessages,
		max_tokens: Tokens,
		model: Option<Models>,
		temprature: Option<F32>,
		presence_penalty: Option<F32>,
		frequency_penalty: Option<F32>,
	) -> Result<Self::Response>;

	/// Get the content from the response.
	async fn extract_response_content(res: &Self::Response) -> Result<String>;

	/// Generates the summary of the current [`TapestryFragment`] instance.
	///
	/// This is will utilize the GPT-4 32K model to generate the summary to allow the maximum number
	/// of possible tokens in the GPT-4 8K model stored in the tapestry fragment.
	///
	/// Returns the summary message as a string.
	async fn generate_summary(
		max_tokens_with_summary: Tokens,
		tapestry_fragment: &TapestryFragment,
	) -> Result<String>;

	/// Get the maximum number of tokens allowed for the current [`Models`].
	fn get_max_token_limit() -> Tokens {
		(T::Model::get().max_context_tokens() as f32 * (T::TOKEN_THRESHOLD_PERCENTILE)) as Tokens
	}

	/// Helper method to build a [`ContextMessage`]
	fn build_context_message(
		role: WrapperRole,
		content: String,
		account_id: Option<String>,
	) -> ContextMessage {
		ContextMessage { role, content, account_id, timestamp: chrono::Utc::now().to_rfc3339() }
	}
}

type LoomRequestMessages<T> = <Loreweaver<T> as Loom<T>>::RequestMessages;
type LoomResponse<T> = <Loreweaver<T> as Loom<T>>::Response;

#[derive(Debug, thiserror::Error)]
enum LoomError {
	Weave(#[from] WeaveError),
	Storage(#[from] StorageError),
}

impl Display for LoomError {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		match self {
			Self::Weave(e) => write!(f, "{}", e),
			Self::Storage(e) => write!(f, "{}", e),
		}
	}
}

/// The bread & butter of Loreweaver.
///
/// All core functionality is implemented by this struct.
pub struct Loreweaver<T: Config>(PhantomData<T>);

#[derive(Debug, thiserror::Error)]
enum WeaveError {
	/// Failed to prompt OpenAI.
	FailedPromptOpenAI(OpenAIError),
	/// Failed to get content from OpenAI response.
	FailedToGetContent,
	/// Bad configuration
	BadConfig(String),
}

impl Display for WeaveError {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		match self {
			Self::FailedPromptOpenAI(e) => write!(f, "Failed to prompt OpenAI: {}", e),
			Self::FailedToGetContent => write!(f, "Failed to get content from OpenAI response"),
			Self::BadConfig(msg) => write!(f, "Bad configuration: {}", msg),
		}
	}
}

/// Token to word ratio.
///
/// Every token equates to 75% of a word.
const TOKEN_WORD_RATIO: f32 = 0.75;

lazy_static! {
	/// The OpenAI client to interact with the OpenAI API.
	static ref OPENAI_CLIENT: RwLock<async_openai::Client<OpenAIConfig>> = RwLock::new(async_openai::Client::new());
}

#[async_trait]
impl<T: Config> Loom<T> for Loreweaver<T> {
	type RequestMessages = Vec<ChatCompletionRequestMessage>;
	type Response = CreateChatCompletionResponse;

	#[instrument]
	async fn weave<TID: TapestryId>(
		tapestry_id: TID,
		system: String,
		msgs: Vec<ContextMessage>,
	) -> Result<LoomResponse<T>> {
		let build_req_msgs = <Loreweaver<T> as Loom<T>>::build_req_msgs;

		let system_ctx_msg = Self::build_context_message(
			WrapperRole::from(SYSTEM_ROLE.to_string()).into(),
			system,
			None,
		);
		let sys_req_msg =
			build_req_msgs(&vec![system_ctx_msg.clone()])?.first().unwrap().to_owned();

		// get latest tapestry fragment instance from storage
		let tapestry_fragment = T::TapestryChest::get_tapestry_fragment(tapestry_id.clone(), None)
			.await?
			.unwrap_or_default();

		// number of tokens available according to the configured model or custom max tokens
		let max_tokens_limit = Self::get_max_token_limit();

		// allocate space for the system message
		let mut req_msgs = VecDeque::with_capacity(tapestry_fragment.context_messages.len() + 1);
		req_msgs.push_front(sys_req_msg);
		req_msgs.append(&mut build_req_msgs(&tapestry_fragment.context_messages)?.into());

		let mut maybe_summary_text: Option<String> = None;
		let msgs_tokens =
			msgs.iter().fold(0, |acc, m: &ContextMessage| acc + m.content.count_tokens());
		// generate summary and start new tapestry instance if context tokens would exceed maximum
		// amount of allowed tokens
		if max_tokens_limit <= tapestry_fragment.context_tokens + msgs_tokens {
			maybe_summary_text = Some(
				<Loreweaver<T> as Loom<T>>::generate_summary(max_tokens_limit, &tapestry_fragment)
					.await?,
			);
		}
		let is_summary_generated = maybe_summary_text.is_some();

		let mut tapestry_fragment_to_persist = if let Some(s) = maybe_summary_text {
			let summary = Self::build_context_message(
				WrapperRole::from(SYSTEM_ROLE.to_string()).into(),
				s.clone(),
				None,
			);

			req_msgs
				.push_front(build_req_msgs(&vec![summary.clone()])?.first().unwrap().to_owned());

			// keep system and summary messages
			req_msgs.truncate(2);

			TapestryFragment {
				context_tokens: s.count_tokens(),
				context_messages: Vec::from([system_ctx_msg, summary]),
			}
		} else {
			tapestry_fragment
		};

		let max_tokens =
			max_tokens_limit - tapestry_fragment_to_persist.context_tokens - msgs_tokens;

		let ctx_msgs = msgs
			.into_iter()
			.map(|m| {
				Self::build_context_message(
					WrapperRole::from(m.role).into(),
					m.content.clone(),
					m.account_id.clone(),
				)
			})
			.collect::<Vec<ContextMessage>>();

		// add new user message to request_messages which will be used to prompt with
		// also include the system message to indicate how many words the response should be
		req_msgs.extend(build_req_msgs(
			&[
				ctx_msgs.as_slice(),
				&[Self::build_context_message(
					WrapperRole::from(SYSTEM_ROLE.to_string()).into(),
					format!("Respond with {} words or less", max_tokens as f32 * TOKEN_WORD_RATIO),
					None,
				)],
			]
			.concat(),
		)?);

		// get response object from prompt
		let response =
			<Loreweaver<T> as Loom<T>>::prompt(req_msgs.into(), max_tokens, None, None, None, None)
				.await
				.map_err(|e| {
					error!("Failed to prompt ChatGPT: {}", e);
					e
				})?;

		// get response content from prompt
		let response_content = <Loreweaver<T> as Loom<T>>::extract_response_content(&response)
			.await
			.map_err(|e| {
				error!("Failed to get content from ChatGPT response: {}", e);
				e
			})?;

		// persist new messages and response to the tapestry fragment
		tapestry_fragment_to_persist.extend_messages(
			ctx_msgs
				.into_iter()
				.chain(vec![Self::build_context_message(
					WrapperRole::from(ASSISTANT_ROLE.to_string()).into(),
					response_content.clone(),
					None,
				)])
				.collect(),
		);

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

	async fn generate_summary(
		num_tokens: Tokens,
		tapestry_fragment: &TapestryFragment,
	) -> Result<String> {
		let model_max_tokens = <Loreweaver<T> as Loom<T>>::get_max_token_limit();
		if num_tokens > model_max_tokens {
			return Err(Box::new(WeaveError::BadConfig(format!(
				"Number of tokens cannot exceed model's max tokens ({}): {}",
				T::Model::get().name(),
				T::Model::get().max_context_tokens()
			))));
		}

		let build_ctx_msg = <Loreweaver<T> as Loom<T>>::build_context_message;
		let build_messages = <Loreweaver<T> as Loom<T>>::build_req_msgs;

		let gen_summary_prompt = build_messages(&vec![build_ctx_msg(
			WrapperRole::from(SYSTEM_ROLE.to_string()),
			format!(
				"Generate a summary of the entire adventure so far. Respond with {} words or less",
				num_tokens as f32 * TOKEN_WORD_RATIO
			),
			None,
		)])?;

		let res = <Loreweaver<T> as Loom<T>>::prompt(
			gen_summary_prompt,
			num_tokens,
			Some(Models::GPT4_32K),
			None,
			None,
			None,
		)
		.await?;

		let summary_response_content =
			<Loreweaver<T> as Loom<T>>::extract_response_content(&res).await?;

		let summary_req_msg = build_messages(&vec![build_ctx_msg(
			WrapperRole::from(SYSTEM_ROLE.to_string()),
			format!("\n\"\"\"\n {}", summary_response_content),
			None,
		)])?;

		let mut request_messages =
			VecDeque::with_capacity(tapestry_fragment.context_messages.len() + 1);
		request_messages.push_back(summary_req_msg.first().unwrap().to_owned());

		Ok(summary_response_content)
	}

	async fn prompt(
		msgs: LoomRequestMessages<T>,
		max_tokens: Tokens,
		model: Option<Models>,
		temprature: Option<f32>,
		presence_penalty: Option<f32>,
		frequency_penalty: Option<f32>,
	) -> Result<LoomResponse<T>> {
		let request = CreateChatCompletionRequestArgs::default()
			.model(model.unwrap_or(T::Model::get()).name())
			.messages(msgs.to_owned())
			.max_tokens(max_tokens)
			.temperature(temprature.unwrap_or(T::TEMPRATURE))
			.presence_penalty(presence_penalty.unwrap_or(T::PRESENCE_PENALTY))
			.frequency_penalty(frequency_penalty.unwrap_or(T::FREQUENCY_PENALTY))
			.build()?;

		OPENAI_CLIENT.read().await.chat().create(request).await.map_err(|e| {
			error!("Failed to prompt OpenAI: {}", e);
			WeaveError::FailedPromptOpenAI(e).into()
		})
	}

	fn build_req_msgs(msgs: &Vec<ContextMessage>) -> Result<LoomRequestMessages<T>> {
		msgs.into_iter()
			.map(|msg: &ContextMessage| {
				let mut builder = ChatCompletionRequestMessageArgs::default();

				let role: Role = WrapperRole::from(msg.role.clone()).into();
				builder.role(role).content(msg.content.clone());

				if let Some(ref account_id) = msg.account_id {
					builder.name(account_id);
				}

				builder.build().map_err(|e| {
					error!("Failed to build ChatCompletionRequestMessageArgs: {}", e);
					e.into()
				})
			})
			.collect()
	}

	async fn extract_response_content(res: &LoomResponse<T>) -> Result<String> {
		res.choices[0]
			.clone()
			.message
			.content
			.ok_or(WeaveError::FailedToGetContent.into())
	}

	fn get_max_token_limit() -> Tokens {
		(T::Model::get().max_context_tokens() as f32 * (T::TOKEN_THRESHOLD_PERCENTILE)) as Tokens
	}
}

pub mod models {
	use clap::{builder::PossibleValue, ValueEnum};
	use tiktoken_rs::p50k_base;

	use crate::Get;

	/// Tokens are a ChatGPT concept which represent normally a third of a word (or 75%).
	pub type Tokens = u16;

	/// Tokens are a ChatGPT concept which represent normally a third of a word (or 75%).
	///
	/// This trait auto implements some basic utility methods for counting the number of tokens from
	/// a string.
	pub trait Token: ToString {
		/// Count the number of tokens in the string.
		fn count_tokens(&self) -> Tokens {
			let bpe = p50k_base().unwrap();
			let tokens = bpe.encode_with_special_tokens(&self.to_string());

			tokens.len() as Tokens
		}
	}

	/// Implement the trait for String.
	///
	/// This is done so that we can call `count_tokens` on a String.
	impl Token for String {}

	/// The ChatGPT language models that are available to use.
	#[derive(PartialEq, Eq, Clone, Debug, Copy)]
	pub enum Models {
		GPT3,
		GPT4,
		GPT4_32K,
	}

	/// Clap value enum implementation for argument parsing.
	impl ValueEnum for Models {
		fn value_variants<'a>() -> &'a [Self] {
			&[Self::GPT3, Self::GPT4, Self::GPT4_32K]
		}

		fn to_possible_value(&self) -> Option<PossibleValue> {
			Some(match self {
				Self::GPT3 => PossibleValue::new(Self::GPT3.name()),
				Self::GPT4 => PossibleValue::new(Self::GPT4.name()),
				Self::GPT4_32K => PossibleValue::new(Self::GPT4_32K.name()),
			})
		}
	}

	impl Models {
		/// Get the model name.
		pub fn name(&self) -> &'static str {
			match self {
				Self::GPT3 => "gpt-3.5-turbo",
				Self::GPT4 => "gpt-4",
				Self::GPT4_32K => "gpt-4-32k",
			}
		}

		/// Maximum number of tokens that can be processed at once by ChatGPT.
		pub fn max_context_tokens(&self) -> Tokens {
			match self {
				Self::GPT3 => 4_096,
				Self::GPT4 => 8_192,
				Self::GPT4_32K => 32_768,
			}
		}
	}

	pub struct DefaultModel;

	impl Get<Models> for DefaultModel {
		fn get() -> Models {
			Models::GPT3
		}
	}
}
