use std::{
	fmt::{Debug, Display},
	marker::PhantomData,
};

use async_openai::{
	error::OpenAIError,
	types::{ChatCompletionRequestMessage, ChatCompletionRequestMessageArgs, Role},
};
use serde::{Deserialize, Serialize};
use serenity::async_trait;
use tracing::{debug, instrument};

use self::models::{MaxTokens, Models};

pub trait Get<T> {
	fn get() -> T;
}

/// A trait representing a unique identifier for a weaving.
pub trait WeavingID: Debug + Display + Send + Sync + 'static {
	/// Returns the base key for a given [`WeavingID`].
	fn base_key(&self) -> String;
}

#[async_trait]
/// A trait representing a storage handler for a specific type of weaving.
pub trait StorageHandler<Key: WeavingID> {
	/// Adds a [`StoryPart`] to storage for a given [`WeavingID`].
	async fn save_story_part(
		weaving_id: &Key,
		story_part: StoryPart,
		increment: bool,
	) -> Result<(), WeaveError>;
	/// Gets the last [`StoryPart`] from storage for a given [`WeavingID`].
	async fn get_last_story_part(weaving_id: &Key) -> Result<Option<StoryPart>, WeaveError>;
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
	/// Type alias encompassing a server id and a story id.
	///
	/// Used mostly for querying some blob storage in the form of a path.
	type WeavingID: WeavingID;
	/// Storage handler implementation which is used to store and retrieve story parts.
	type Storage: StorageHandler<Self::WeavingID>;
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
pub struct StoryPart {
	/// Number of players that are part of the story. (typically this changes based on players
	/// entering the commands `/leave` or `/join`).
	///
	/// When generating a new story part (N + 1, where N is the current story part number), we need
	/// to copy over the same number of players. The story must remain consistent throughout each
	/// part.
	pub players: Vec<AccountId>,
	/// Total number of _GPT tokens_ in the story part.
	pub context_tokens: u16,
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
	async fn prompt(
		weaving_id: T::WeavingID,
		msg: String,
		account_id: AccountId,
		username: String,
		pseudo_username: Option<String>,
	) -> Result<String, WeaveError>;
}

/// The bread & butter of Loreweaver.
///
/// All core functionality is implemented by this struct.
pub struct Loreweaver<T: Config>(PhantomData<T>);

impl<T: Config> secret_lore::Sealed<T> for Loreweaver<T> {}

impl<T: Config> Loreweaver<T> {
	/// Maximum number of words to return in a response based on maximum tokens of GPT model or a
	/// `custom` supplied value.
	///
	/// Every token equates to 75% of a word.
	fn max_words(
		model: Models,
		custom_max_tokens: Option<MaxTokens>,
		context_tokens: MaxTokens,
	) -> MaxTokens {
		let max_tokens = custom_max_tokens
			.unwrap_or(Models::default_max_response_tokens(&model, context_tokens));

		(max_tokens as f64 * 0.75) as MaxTokens
	}
}

#[derive(Debug)]
pub enum WeaveError {
	OpenAIError(OpenAIError),
	// TODO: Do not resort to Custom error unless it is absolutely necessary.
	// Create new error types for each error case.
	Custom(String),
}

#[async_trait]
impl<T: Config> Loom<T> for Loreweaver<T> {
	#[instrument]
	async fn prompt(
		weaving_id: T::WeavingID,
		msg: String,
		_account_id: AccountId,
		username: String,
		pseudo_username: Option<String>,
	) -> Result<String, WeaveError> {
		let model = T::Model::get();

		let mut story_part =
			T::Storage::get_last_story_part(&weaving_id).await?.unwrap_or_default();

		let username_with_nick = match pseudo_username {
			Some(pseudo_username) => format!("{}{}", username, pseudo_username),
			None => username,
		};

		story_part.context_messages.push(ContextMessage {
			role: "user".to_string(),
			account_id: None,
			username: Some(username_with_nick.clone()),
			content: msg.clone(),
			timestamp: chrono::Utc::now().to_rfc3339(),
		});

		let mut request_messages: Vec<ChatCompletionRequestMessage> = story_part
			.context_messages
			.iter()
			.map(|msg: &ContextMessage| {
				ChatCompletionRequestMessageArgs::default()
					.content(msg.content.clone())
					.role(match msg.role.as_str() {
						"system" => Role::System,
						"assistant" => Role::Assistant,
						"user" => Role::User,
						_ => Err(WeaveError::Custom("Invalid role".into())).unwrap(),
					})
					.name(match msg.role.as_str() {
						"system" => "Loreweaver",
						"assistant" => "Loreweaver", // TODO: This should be the NPC...
						"user" => username_with_nick.as_str(),
						_ => Err(WeaveError::Custom("Invalid role".into())).unwrap(),
					})
					.build()
					.unwrap()
			})
			.collect::<Vec<ChatCompletionRequestMessage>>();

		let max_words = Loreweaver::<T>::max_words(model, None, story_part.context_tokens as u128);

		debug!("Prompting ChatGPT...");

		match <Loreweaver<T> as secret_lore::Sealed<T>>::do_prompt(
			Models::GPT4,
			&mut request_messages,
			max_words,
		)
		.await
		{
			Ok(res) => Ok(match res.choices[0].clone().message.content {
				Some(content) => {
					story_part.context_messages.push(ContextMessage {
						role: "assistant".to_string(),
						account_id: None,
						username: None,
						content: content.clone(),
						timestamp: chrono::Utc::now().to_rfc3339(),
					});

					debug!("Saving story part: {:?}", story_part.context_messages);

					T::Storage::save_story_part(&weaving_id, story_part, false).await?;

					content
				},
				None =>
					return Err(WeaveError::Custom("Failed to get content from response".into())),
			}),
			Err(err) => Err(WeaveError::OpenAIError(err)),
		}
	}
}

pub mod models {
	use clap::{builder::PossibleValue, ValueEnum};

	pub type MaxTokens = u128;

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
		pub fn default_max_response_tokens(
			model: &Models,
			tokens_in_context: MaxTokens,
		) -> MaxTokens {
			(model.max_context_tokens() - tokens_in_context) / 3
		}

		/// Maximum number of tokens that can be processed at once by ChatGPT.
		pub fn max_context_tokens(&self) -> MaxTokens {
			match self {
				Self::GPT3 => 4_096,
				Self::GPT4 => 8_192,
			}
		}
	}
}

mod secret_lore {
	use async_openai::{
		config::OpenAIConfig,
		error::OpenAIError,
		types::{
			ChatCompletionRequestMessage, ChatCompletionRequestMessageArgs,
			CreateChatCompletionRequestArgs, CreateChatCompletionResponse, Role,
		},
	};
	use lazy_static::lazy_static;
	use tiktoken_rs::p50k_base;
	use tokio::sync::RwLock;
	use tracing::error;

	use super::{
		models::{MaxTokens, Models},
		Config,
	};

	/// Loreweaver system instructions with number of ChatGPT tokens.
	type System = (String, u128);

	lazy_static! {
		/// The OpenAI client to interact with the OpenAI API.
		static ref OPENAI_CLIENT: RwLock<async_openai::Client<OpenAIConfig>> = RwLock::new(async_openai::Client::new());

		/// Loreweaver System instructions
		pub static ref LOREWEAVER_SYSTEM: RwLock<System> = {
			let system = std::fs::read_to_string("loreweaver-system.txt").map_err(|e| {
				error!("Failed to read loreweaver-system.txt: {e:?}");
				panic!()
			}).unwrap();

			let tokens = system.count_tokens();

			RwLock::new((system, tokens))
		};
	}

	pub trait Sealed<T: Config> {
		/// The action to query ChatGPT with the supplied configurations and messages.
		async fn do_prompt(
			model: Models,
			msgs: &mut Vec<ChatCompletionRequestMessage>,
			_max_words: MaxTokens,
		) -> Result<CreateChatCompletionResponse, OpenAIError> {
			// Add the system to the beginning of the message history
			msgs.splice(
				0..0,
				[ChatCompletionRequestMessageArgs::default()
					.role(Role::System)
					.content(LOREWEAVER_SYSTEM.read().await.clone().0)
					.build()?],
			);

			let request = CreateChatCompletionRequestArgs::default()
				// TODO: Set max tokens response based on user subscription
				// TODO: use `_max_words` to limit the number of words in the response. ChatGPT does
				// not  make a coherent response while respecting the max_tokens() limit.
				.max_tokens(300u16)
				.model(model.name())
				.messages(msgs.to_owned())
				.build()?;

			OPENAI_CLIENT.read().await.chat().create(request).await
		}
	}

	/// Tokens are a ChatGPT concept which represent normally a third of a word (or 75%).
	///
	/// This trait auto implements some basic utility methods for counting the number of tokens from
	/// a string.
	pub trait Tokens: ToString {
		/// Count the number of tokens in the string.
		fn count_tokens(&self) -> MaxTokens {
			let bpe = p50k_base().unwrap();
			let tokens = bpe.encode_with_special_tokens(&self.to_string());

			tokens.len() as MaxTokens
		}
	}

	/// Implement the trait for String.
	///
	/// This is done so that we can call `count_tokens` on a String.
	impl Tokens for String {}
}
