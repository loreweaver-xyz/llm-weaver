#![feature(async_fn_in_trait)]
#![feature(async_closure)]
#![feature(associated_type_defaults)]

use std::{
	fmt::{Debug, Display},
	marker::PhantomData,
};

use async_openai::{
	error::OpenAIError,
	types::{ChatCompletionRequestMessage, ChatCompletionRequestMessageArgs, Role},
};
use async_trait::async_trait;
use models::Tokens;
use serde::{Deserialize, Serialize};
use storage::{StorageError, TapestryChest};
use tracing::{debug, error, instrument};

use self::models::Models;

mod storage;

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
pub trait TapestryId: Debug + Display + Send + Sync + 'static {
	/// Returns the base key.
	///
	/// This method should produce a unique string identifier, that will serve as a key for
	/// associated objects or data within [`TapestryChestHandler`].
	fn base_key(&self) -> String;
}

/// A trait consisting mainly of associated types implemented by [`Loreweaver`].
///
/// Normally structs implementing [`crate::Server`] would implement this trait to call methods
/// implemented by [`Loreweaver`]
#[async_trait]
/// A trait representing the configuration for a Loreweaver server.
pub trait Config {
	/// Getter for GPT model to use.
	type Model: Get<Models>;
	/// Storage handler implementation for storing and retrieving tapestry fragments.
	///
	/// This can simply be a struct that implements [`TapestryChestHandler`] utilizing the default
	/// implementation which uses Redis as the storage backend.
	///
	/// If you wish to implement your own storage backend, you can implement the methods from the
	/// trait. [`Loreweaver`] does not care how you store the data and retrieve your data.
	type TapestryChest: TapestryChestHandler = TapestryChest;
}
/// An platform agnostic type representing a user's account ID.
pub type AccountId = u64;

/// Context message that represent a single message in a [`StoryPart`].
#[derive(Default, Clone, Debug, Serialize, Deserialize)]
pub struct ContextMessage {
	pub role: String,
	pub account_id: Option<String>,
	pub username: Option<String>,
	pub content: String,
	pub timestamp: String,
}

/// Represents a single part of a story containing a list of messages along with other metadata.
///
/// ChatGPT can only hold a limited amount of tokens in a the entire message history/context.
/// Therefore, at every [`Loom::prompt`] execution, we must keep track of the number of
/// `context_tokens` in the current story part and if it exceeds the maximum number of tokens
/// allowed for the current GPT [`Models`], then we must generate a summary of the current story
/// part and use that as the starting point for the next story part. This is one of the biggest
/// challenges for Loreweaver to keep a consistent narrative throughout the many story parts.
#[derive(Debug, Serialize, Deserialize, Default, Clone)]
pub struct TapestryFragment {
	/// Number of players that are part of the story. (typically this changes based on players
	/// entering the commands `/leave` or `/join`).
	///
	/// When generating a new story part (N + 1, where N is the current story part number), we need
	/// to copy over the same number of players. The story must remain consistent throughout each
	/// part.
	pub players: Vec<AccountId>,
	/// Total number of _GPT tokens_ in the story part.
	pub context_tokens: u64,
	/// List of [`ContextMessage`]s in the story part.
	pub context_messages: Vec<ContextMessage>,
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
	/// Prompt Loreweaver for a response for [`WeavingID`].
	///
	/// Prompts ChatGPT with the current [`StoryPart`] and the `msg`.
	///
	/// If 80% of the maximum number of tokens allowed in a message history for the configured
	/// ChatGPT [`Models`] has been reached, a summary will be generated instead of the current
	/// message history and saved to the cloud. A new message history will begin.
	async fn weave<Id: TapestryId>(
		tapestry_id: &Id,
		system: String,
		override_max_context_tokens: Option<Tokens>,
		msg: String,
		account_id: AccountId,
		username: String,
	) -> Result<String>;
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

impl<T: Config> secret_lore::Sealed<T> for Loreweaver<T> {}

impl<T: Config> Loreweaver<T> {}

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

#[async_trait]
impl<T: Config> Loom<T> for Loreweaver<T> {
	#[instrument]
	async fn weave<Id: TapestryId>(
		tapestry_id: &Id,
		system: String,
		override_max_context_tokens: Option<Tokens>,
		msg: String,
		_account_id: AccountId,
		username: String,
	) -> Result<String> {
		let mut story_part = T::TapestryChest::get_tapestry_fragment(tapestry_id, None)
			.await
			.map_err(|e| {
				error!("Failed to get last story part: {}", e);
				e
			})?
			.unwrap_or_default();

		story_part.context_messages.push(ContextMessage {
			role: "user".to_string(),
			account_id: None,
			username: Some(username.clone()),
			content: msg.clone(),
			timestamp: chrono::Utc::now().to_rfc3339(),
		});

		// Add the system to the beginning of the request messages
		// This will not be persisted to the story part context messages to avoid saving aditional
		// data. TODO: maybe have a flag to save it to context messages if needed by the
		// application.
		let mut request_messages = vec![ChatCompletionRequestMessageArgs::default()
			.role(Role::System)
			.content(system)
			.build()
			.map_err(|e| {
				error!("Failed to build ChatCompletionRequestMessageArgs: {}", e);
				WeaveError::FailedPromptOpenAI(e)
			})?];

		request_messages.extend(
			story_part
				.context_messages
				.iter()
				.map(|msg: &ContextMessage| {
					ChatCompletionRequestMessageArgs::default()
						.content(msg.content.clone())
						.role(Into::<WrapperRole>::into(msg.role.clone()))
						.name(match msg.role.as_str() {
							"system" => "Loreweaver".to_string(),
							"assistant" | "user" => username.to_string(),
							_ => WeaveError::BadOpenAIRole(msg.role.clone()).to_string(),
						})
						.build()
						.unwrap()
				})
				.collect::<Vec<ChatCompletionRequestMessage>>(),
		);

		let res = <Loreweaver<T> as secret_lore::Sealed<T>>::do_prompt(
			&mut request_messages,
			story_part.context_tokens,
			override_max_context_tokens,
		)
		.await
		.map_err(|e| {
			error!("Failed to prompt ChatGPT: {}", e);
			WeaveError::FailedPromptOpenAI(e)
		})?;

		let response_content =
			res.choices[0].clone().message.content.ok_or(WeaveError::FailedToGetContent)?;

		story_part.context_messages.push(ContextMessage {
			role: "assistant".to_string(),
			account_id: None,
			username: None,
			content: response_content.clone(),
			timestamp: chrono::Utc::now().to_rfc3339(),
		});

		debug!("Saving story part: {:?}", story_part.context_messages);

		T::TapestryChest::save_tapestry_fragment(tapestry_id, story_part, false)
			.await
			.map_err(|e| {
				error!("Failed to save story part: {}", e);
				e
			})?;

		Ok(response_content)
	}
}

mod secret_lore {
	use crate::Get;
	use async_openai::{
		config::OpenAIConfig,
		error::OpenAIError,
		types::{
			ChatCompletionRequestMessage, ChatCompletionRequestMessageArgs,
			CreateChatCompletionRequestArgs, CreateChatCompletionResponse, Role,
		},
	};
	use lazy_static::lazy_static;
	use tokio::sync::RwLock;
	use tracing::error;

	use crate::models::Tokens;

	use super::{models::Models, Config};

	lazy_static! {
		/// The OpenAI client to interact with the OpenAI API.
		static ref OPENAI_CLIENT: RwLock<async_openai::Client<OpenAIConfig>> = RwLock::new(async_openai::Client::new());
	}

	/// Maximum percentage of tokens allowed to generate a summary.
	///
	/// This is not a fixed amount of tokens since the maximum amount of context tokens can change
	/// depending on the model or user input.
	const SUMMARY_PERCENTAGE: f64 = 0.1;

	/// Token to word ratio.
	///
	/// Every token equates to 75% of a word.
	const TOKEN_TO_WORD_RATIO: f64 = 0.75;

	pub trait Sealed<T: Config> {
		/// The action to query ChatGPT with the supplied configurations and messages.
		///
		/// Auto injects a system message at the end of vec of messages to instruct ChatGPT to
		/// respond with a certain number of words.
		///
		/// We do this here to avoid any other service from having to do this.
		async fn do_prompt(
			msgs: &mut Vec<ChatCompletionRequestMessage>,
			context_tokens: Tokens,
			override_max_tokens: Option<Tokens>,
		) -> Result<CreateChatCompletionResponse, OpenAIError> {
			let model = T::Model::get();

			let max_response_words = Self::max_words(model, override_max_tokens, context_tokens);

			msgs.push(
				ChatCompletionRequestMessageArgs::default()
					.content(format!("Respond with {} words or less", max_response_words))
					.role(Role::System)
					.build()
					.map_err(|e| {
						error!("Failed to build ChatCompletionRequestMessageArgs: {}", e);
						e
					})?,
			);

			let request = CreateChatCompletionRequestArgs::default()
				.max_tokens(300u16)
				.temperature(0.9f32)
				.presence_penalty(0.6f32)
				.frequency_penalty(0.6f32)
				.model(model.name())
				// .suffix("Loreweaver:")
				.messages(msgs.to_owned())
				.build()?;

			OPENAI_CLIENT.read().await.chat().create(request).await
		}

		/// Maximum number of words to return in a response based on maximum tokens of GPT model or
		/// a `custom` supplied value.
		///
		/// Every token equates to 75% of a word.
		fn max_words(
			model: Models,
			custom_max_tokens: Option<Tokens>,
			context_tokens: Tokens,
		) -> Tokens {
			let tokens_available = custom_max_tokens
				.unwrap_or(Models::default_max_response_tokens(&model, context_tokens))
				as f64 * SUMMARY_PERCENTAGE;

			(tokens_available * TOKEN_TO_WORD_RATIO) as Tokens
		}
	}
}

pub mod models {
	use clap::{builder::PossibleValue, ValueEnum};
	use tiktoken_rs::p50k_base;

	/// Tokens are a ChatGPT concept which represent normally a third of a word (or 75%).
	pub type Tokens = u64;

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

		/// Default maximum tokens to respond with for a ChatGPT prompt.
		///
		/// This would normally be used when prompting ChatGPT API and specifying the maximum tokens
		/// to return.
		///
		/// `tokens_in_context` parameter is the current number of tokens that are part of the
		/// context. This should not surpass the [`max_context_tokens`]
		pub fn default_max_response_tokens(model: &Models, tokens_in_context: Tokens) -> Tokens {
			(model.max_context_tokens() - tokens_in_context) / 3
		}

		/// Maximum number of tokens that can be processed at once by ChatGPT.
		pub fn max_context_tokens(&self) -> Tokens {
			match self {
				Self::GPT3 => 4_096,
				Self::GPT4 => 8_192,
			}
		}
	}
}
