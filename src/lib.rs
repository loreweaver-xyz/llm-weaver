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
//! use loreweaver::{Loreweaver, Config, TapestryId};
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
//!     "What was the last question I asked you?".to_string(),
//!     account_id,
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
//!     "What was the last question I asked you?".to_string(),
//!     account_id,
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

pub use storage::TapestryChestHandler;

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
	/// Maximum percentage of tokens allowed to generate a summary.
	///
	/// This is not a fixed amount of tokens since the maximum amount of context tokens can change
	/// depending on the model or custom max tokens.
	///
	/// Defaults to `0.15`
	const SUMMARY_PERCENTAGE: f32 = 0.15;
	/// The sampling temperature, between 0 and 1. Higher values like 0.8 will make the output more
	/// random, while lower values like 0.2 will make it more focused and deterministic. If set to
	/// 0, the model will use log probability to automatically increase the temperature until
	/// certain thresholds are hit.
	///
	/// Defaults to `0.0`
	const TEMPRATURE: f32 = 0.0;
	/// Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they
	/// appear in the text so far, increasing the model's likelihood to talk about new topics.
	///
	/// Defaults to `0.0`
	const PRESENCE_PENALTY: f32 = 0.0;
	/// Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing
	/// frequency in the text so far, decreasing the model's likelihood to repeat the same line
	/// verbatim.
	///
	/// Defaults to `0.0`
	const FREQUENCY_PENALTY: f32 = 0.0;

	/// Getter for GPT model to use.
	///
	/// Defaults to [`models::DefaultModel`]
	type Model: Get<Models> = models::DefaultModel;
	/// Storage handler implementation for storing and retrieving tapestry fragments.
	///
	/// This can simply be a struct that implements [`TapestryChestHandler`] utilizing the default
	/// implementation which uses Redis as the storage backend.
	///
	/// If you wish to implement your own storage backend, you can implement the methods from the
	/// trait. [`Loreweaver`] does not care how you store the data and retrieve your data.
	///
	/// Defaults to [`TapestryChest`]. Using this default requires you to supply the `hostname`,
	/// `port` and `credentials` to connect to your instance.
	type TapestryChest: TapestryChestHandler = TapestryChest;
}

/// Context message that represent a single message in a [`TapestryFragment`] instance.
#[derive(Default, Clone, Debug, Serialize, Deserialize)]
pub struct ContextMessage {
	pub role: String,
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
	fn add_message(&mut self, msg: Vec<ContextMessage>) {
		msg.iter().for_each(|msg| {
			self.context_tokens += msg.content.count_tokens();
			self.context_messages.push(msg.clone());
		});
	}
}

/// A trait that defines all of the public associated methods that [`Loreweaver`] implements.
///
/// This is the machine that drives all of the core methods that should be used across any service
/// that needs to prompt ChatGPT and receive a response.
///
/// The implementations should handle all form of validation and usage tracking all while
/// abstracting the logic from the services calling them.
#[async_trait]
pub trait Loom<T: Config> {
	/// Represents an object to use for constructing [`Loom::RequestMessages`] from.
	type Message;
	/// Represents the request message type used to prompt a certain LLM.
	///
	/// This varies between LLMs and their libraries.
	type RequestMessages: IntoIterator;
	/// Represents the response type returned by the LLM library.
	type Response;

	/// Prompt Loreweaver for a response for [`TapestryId`].
	///
	/// Prompts ChatGPT with the current [`TapestryFragment`] instance and the new `msg`.
	///
	/// If 80% of the maximum number of tokens allowed in a message history for the configured
	/// ChatGPT [`Models`] has been reached, a summary will be generated instead of the current
	/// message history and saved to the cloud. A new message history will begin.
	///
	/// # Parameters
	///
	/// - `tapestry_id`: The [`TapestryId`] to prompt and save context messages to.
	/// - `system`: The system message to prompt ChatGPT with.
	/// - `override_max_context_tokens`: Override the maximum number of context tokens allowed for
	///  the current [`Models`].
	/// - `msg`: The user message to prompt ChatGPT with.
	/// - `account_id`: An optional arbitrary representation of an account id. This will be used as
	///   the `name` parameter when prompting ChatGPT. Leaving it at `None` will leave the `name`
	///   parameter empty.
	async fn weave<TID: TapestryId>(
		tapestry_id: TID,
		system: String,
		override_max_context_tokens: Option<Tokens>,
		msg: String,
		account_id: Option<String>,
	) -> Result<String>;

	/// Build the message/messages to prompt ChatGPT with.
	fn build_messages(msg: Vec<Self::Message>) -> Result<Self::RequestMessages>;

	/// The action to query ChatGPT with the supplied configurations and messages.
	async fn prompt(msgs: Self::RequestMessages, max_tokens: Tokens) -> Result<Self::Response>;

	/// Get the content from the response.
	async fn get_content(res: &Self::Response) -> Result<String>;

	/// Maximum number tokens allowed for the current [`Models`] or custom max tokens including the
	/// tokens allocated for the summary based on percentage [`Config::SUMMARY_PERCENTAGE`].
	fn max_tokens_including_summary(model: Models, custom_max_tokens: Option<Tokens>) -> Tokens;

	/// Generates the summary of the current [`TapestryFragment`] instance.
	///
	/// Returns a tuple of:
	/// 		- The new [`TapestryFragment`] instance to save to storage.
	/// 		- The new request messages to prompt ChatGPT with.
	async fn generate_summary(
		max_tokens_with_summary: Tokens,
		tapestry_fragment: TapestryFragment,
		system_req_msg: Self::RequestMessages,
	) -> Result<(TapestryFragment, Self::RequestMessages)>;
}

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
	/// A bad OpenAI role was supplied.
	BadOpenAIRole(String),
}

impl Display for WeaveError {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		match self {
			Self::FailedPromptOpenAI(e) => write!(f, "Failed to prompt OpenAI: {}", e),
			Self::FailedToGetContent => write!(f, "Failed to get content from OpenAI response"),
			Self::BadOpenAIRole(role) => write!(f, "Bad OpenAI role: {}", role),
		}
	}
}

/// Wrapper around [`async_openai::types::types::Role`] for custom implementation.
enum WrapperRole {
	Role(Role),
}

impl From<WrapperRole> for Role {
	fn from(role: WrapperRole) -> Self {
		match role {
			WrapperRole::Role(role) => role,
		}
	}
}

impl From<String> for WrapperRole {
	fn from(role: String) -> Self {
		match role.as_str() {
			"system" => Self::Role(Role::System),
			"assistant" => Self::Role(Role::Assistant),
			"user" => Self::Role(Role::User),
			_ => panic!("Bad OpenAI role"),
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

pub struct MessageParams {
	role: Role,
	content: String,
	account_id: Option<String>,
}

const SYSTEM_ROLE: &str = "system";
const ASSISTANT_ROLE: &str = "assistant";
const USER_ROLE: &str = "user";

type LoomMessage<T> = <Loreweaver<T> as Loom<T>>::Message;
type LoomRequestMessages<T> = <Loreweaver<T> as Loom<T>>::RequestMessages;
type LoomResponse<T> = <Loreweaver<T> as Loom<T>>::Response;

#[async_trait]
impl<T: Config> Loom<T> for Loreweaver<T> {
	type Message = MessageParams;
	type RequestMessages = Vec<ChatCompletionRequestMessage>;
	type Response = CreateChatCompletionResponse;

	#[instrument]
	async fn weave<TID: TapestryId>(
		tapestry_id: TID,
		system: String,
		override_max_context_tokens: Option<Tokens>,
		msg: String,
		account_id: Option<String>,
	) -> Result<String> {
		let build_messages = <Loreweaver<T> as Loom<T>>::build_messages;
		let prompt = <Loreweaver<T> as Loom<T>>::prompt;

		// Early return on invalid token configuration
		let model = T::Model::get();
		if let Some(custom_max_tokens) = override_max_context_tokens {
			if custom_max_tokens > model.max_context_tokens() {
				return Err(Box::new(WeaveError::BadOpenAIRole(format!(
					"Custom max tokens cannot exceed model's max tokens ({}): {}",
					model.name(),
					model.max_context_tokens()
				))))
			}
		}

		// system request message pre built to extend to vecs within this function
		let system_req_msg = build_messages(vec![LoomMessage::<T> {
			role: Role::System,
			content: system.clone(),
			account_id: None,
		}])?;

		// get latest tapestry fragment instance from storage
		let tapestry_fragment = T::TapestryChest::get_tapestry_fragment(tapestry_id.clone(), None)
			.await?
			.unwrap_or_default();

		// number of tokens available according to the configured model or custom max tokens
		let max_tokens_with_summary = <Loreweaver<T> as Loom<T>>::max_tokens_including_summary(
			T::Model::get(),
			override_max_context_tokens,
		);

		// base request messages
		// in the case where we generate a summary or simply go straight to prompting for a
		// response, we need to build this iterator of request messages
		let request_messages = system_req_msg
			.clone()
			.into_iter()
			.chain(tapestry_fragment.context_messages.clone().into_iter().flat_map(
				|msg: ContextMessage| {
					// Assuming build_messages returns a
					// Result<Vec<ChatCompletionRequestMessage>>
					build_messages(vec![LoomMessage::<T> {
						role: WrapperRole::from(msg.role).into(),
						content: msg.content,
						account_id: msg.account_id,
					}])
					.expect("Failed to build messages")
					.into_iter() // Convert the Vec to an Iterator
				},
			))
			.collect::<LoomRequestMessages<T>>();

		// generate summary and start new tapestry instance if context tokens are exceed maximum +
		// the new message token count exceed the amount of allowed tokens
		let generate_summary =
			max_tokens_with_summary <= tapestry_fragment.context_tokens + msg.count_tokens();
		let (mut tapestry_fragment_to_persist, mut request_messages) = match generate_summary {
			true =>
				<Loreweaver<T> as Loom<T>>::generate_summary(
					max_tokens_with_summary,
					tapestry_fragment,
					system_req_msg,
				)
				.await?,
			false => (tapestry_fragment, request_messages),
		};

		let max_tokens = max_tokens_with_summary -
			tapestry_fragment_to_persist.context_tokens -
			msg.count_tokens();

		// add new user message to request_messages which will be used to prompt with
		// also include the system message to indicate how many words the response should be
		request_messages.extend(build_messages(vec![
			LoomMessage::<T> {
				role: Role::User,
				content: msg.clone(),
				account_id: account_id.clone(),
			},
			LoomMessage::<T> {
				role: Role::System,
				content: format!(
					"Respond with {} words or less",
					max_tokens as f32 * TOKEN_WORD_RATIO
				),
				account_id: None,
			},
		])?);

		// get response object from prompt
		let res = prompt(request_messages, max_tokens).await.map_err(|e| {
			error!("Failed to prompt ChatGPT: {}", e);
			e
		})?;

		// get response content from prompt
		let response_content =
			<Loreweaver<T> as Loom<T>>::get_content(&res).await.map_err(|e| {
				error!("Failed to get content from ChatGPT response: {}", e);
				e
			})?;

		// persist new user message and response to the
		tapestry_fragment_to_persist.add_message(vec![
			build_context_message(USER_ROLE.to_string(), msg.clone(), account_id.clone()),
			build_context_message(
				ASSISTANT_ROLE.to_string(),
				response_content.clone(),
				account_id.clone(),
			),
		]);

		debug!("Saving tapestry fragment: {:?}", tapestry_fragment_to_persist);

		// save tapestry fragment to storage
		// when summarized, the tapestry_fragment will be saved under a new instance
		T::TapestryChest::save_tapestry_fragment(
			tapestry_id,
			tapestry_fragment_to_persist,
			generate_summary,
		)
		.await
		.map_err(|e| {
			error!("Failed to save tapestry fragment: {}", e);
			e
		})?;

		Ok(response_content)
	}

	fn build_messages(msgs: Vec<LoomMessage<T>>) -> Result<LoomRequestMessages<T>> {
		msgs.into_iter()
			.map(|msg: LoomMessage<T>| {
				let mut builder = ChatCompletionRequestMessageArgs::default();

				builder.role(msg.role).content(msg.content);

				if let Some(account_id) = msg.account_id {
					builder.name(account_id);
				}

				builder.build().map_err(|e| {
					error!("Failed to build ChatCompletionRequestMessageArgs: {}", e);
					e.into()
				})
			})
			.collect()
	}

	async fn prompt(msgs: LoomRequestMessages<T>, max_tokens: Tokens) -> Result<LoomResponse<T>> {
		let request = CreateChatCompletionRequestArgs::default()
			.model(T::Model::get().name())
			.messages(msgs.to_owned())
			.max_tokens(max_tokens)
			.temperature(T::TEMPRATURE)
			.presence_penalty(T::PRESENCE_PENALTY)
			.frequency_penalty(T::FREQUENCY_PENALTY)
			.build()?;

		OPENAI_CLIENT.read().await.chat().create(request).await.map_err(|e| {
			error!("Failed to prompt OpenAI: {}", e);
			WeaveError::FailedPromptOpenAI(e).into()
		})
	}

	async fn get_content(res: &LoomResponse<T>) -> Result<String> {
		res.choices[0]
			.clone()
			.message
			.content
			.ok_or(WeaveError::FailedToGetContent.into())
	}

	fn max_tokens_including_summary(model: Models, custom_max_tokens: Option<Tokens>) -> Tokens {
		(custom_max_tokens.unwrap_or(model.max_context_tokens()) as f32 *
			(1.0 - T::SUMMARY_PERCENTAGE)) as Tokens
	}

	async fn generate_summary(
		max_tokens_with_summary: Tokens,
		tapestry_fragment: TapestryFragment,
		system_req_msg: LoomRequestMessages<T>,
	) -> Result<(TapestryFragment, LoomRequestMessages<T>)> {
		let tokens_left = max_tokens_with_summary.saturating_sub(tapestry_fragment.context_tokens);
		if tokens_left == 0 {
			return Err(Box::new(WeaveError::BadOpenAIRole(format!(
				"Tokens left cannot be 0: {}",
				tokens_left
			))))
		}

		let build_messages = <Loreweaver<T> as Loom<T>>::build_messages;

		let gen_summary_prompt = build_messages(vec![LoomMessage::<T> {
			role: Role::System,
			content: format!(
				"Generate a summary of the entire adventure so far. Respond with {} words or less",
				tokens_left as f32 * TOKEN_WORD_RATIO
			),
			account_id: None,
		}])?;

		let res = <Loreweaver<T> as Loom<T>>::prompt(gen_summary_prompt, tokens_left).await?;

		let summary_response_content = <Loreweaver<T> as Loom<T>>::get_content(&res).await?;

		let summary_req_msg = build_messages(vec![LoomMessage::<T> {
			role: Role::System,
			content: format!("\n\"\"\"\n {}", summary_response_content),
			account_id: None,
		}])?;

		Ok((
			TapestryFragment {
				context_tokens: summary_response_content.count_tokens(),
				context_messages: vec![build_context_message(
					SYSTEM_ROLE.to_string(),
					summary_response_content.clone(),
					None,
				)],
			},
			system_req_msg
				.into_iter()
				.chain(summary_req_msg)
				.collect::<LoomRequestMessages<T>>(),
		))
	}
}

fn build_context_message(
	role: String,
	content: String,
	account_id: Option<String>,
) -> ContextMessage {
	ContextMessage { role, content, account_id, timestamp: chrono::Utc::now().to_rfc3339() }
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
	}

	/// Clap value enum implementation for argument parsing.
	impl ValueEnum for Models {
		fn value_variants<'a>() -> &'a [Self] {
			&[Self::GPT3, Self::GPT4]
		}

		fn to_possible_value(&self) -> Option<PossibleValue> {
			Some(match self {
				Self::GPT3 => PossibleValue::new(Self::GPT3.name()),
				Self::GPT4 => PossibleValue::new(Self::GPT4.name()),
			})
		}
	}

	impl Models {
		/// Get the model name.
		pub fn name(&self) -> &'static str {
			match self {
				Self::GPT3 => "gpt-3.5-turbo",
				Self::GPT4 => "gpt-4",
			}
		}

		/// Maximum number of tokens that can be processed at once by ChatGPT.
		pub fn max_context_tokens(&self) -> Tokens {
			match self {
				Self::GPT3 => 4_096,
				Self::GPT4 => 8_192,
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
