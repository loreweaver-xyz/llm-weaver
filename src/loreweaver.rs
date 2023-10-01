use std::{marker::PhantomData, sync::Arc};

use async_openai::{
	error::OpenAIError,
	types::{ChatCompletionRequestMessage, ChatCompletionRequestMessageArgs, Role},
};
use serde::{Deserialize, Serialize};
use serenity::async_trait;
use tracing::{info, instrument};

use self::models::{MaxTokens, Models};

pub trait Get<T> {
	fn get() -> T;
}

#[async_trait]
pub trait StorageHandler {
	async fn get(id: WeavingID) -> Result<Option<StoryPart>, WeaveError>;
}

/// A trait consisting mainly of associated types implemented by [`Loreweaver`].
///
/// Normally structs implementing [`crate::Server`] would implement this trait to call methods
/// implemented by [`Loreweaver`]
#[async_trait]
pub trait Config {
	/// Getter for GPT model to use.
	type Model: Get<Models>;
	/// Storage handler implementation which is used to store and retrieve story parts.
	type Storage: StorageHandler;
}

/// Encompasses many [`StoryID`]s.
type ServerID = u128;
/// The `StoryID` identifies a single story which is part of a [`ServerID`].
type StoryID = u128;
/// Type alias encompassing a server id and a story id.
///
/// Used mostly for querying some blob storage in the form of a path.
pub type WeavingID = (ServerID, StoryID);

/// An platform agnostic type representing a user's account ID.
type AccountId = u64;

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
#[derive(Debug, Serialize, Deserialize, Default)]
pub struct StoryPart {
	/// The story part number.
	pub number: u16,
	/// Number of players that are part of the story. (typically this changes based on players
	/// entering the commands `/leave` or `/join`).
	///
	/// When generating a new story part (N + 1, where N is the current story part number), we need
	/// to copy over the same number of players. The story must remain consistent throughout each
	/// part.
	pub players: Vec<AccountId>,
	/// Total number of _GPT tokens_ in the story part.
	pub context_tokens: u128,
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
pub trait Loom {
	/// Prompt Loreweaver for a response for [`WeavingID`].
	///
	/// Prompts ChatGPT with the current [`StoryPart`] and the `msg`.
	///
	/// If 80% of the maximum number of tokens allowed in a message history for the configured
	/// ChatGPT [`Models`] has been reached, a summary will be generated instead of the current
	/// message history and saved to the cloud. A new message history will begin.
	async fn prompt(
		loom_id: WeavingID,
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
	Custom(String),
}

#[async_trait]
impl<T: Config> Loom for Loreweaver<T> {
	#[instrument]
	async fn prompt(
		weaving_id: WeavingID,
		msg: String,
		_account_id: AccountId,
		username: String,
		pseudo_username: Option<String>,
	) -> Result<String, WeaveError> {
		let model = T::Model::get();

		let story_part = Arc::new(T::Storage::get(weaving_id).await?);
		let story_part_clone = Arc::clone(&story_part);

		let context_messages = tokio::spawn(async move {
			match story_part.as_ref() {
				Some(part) => {
					// TODO: Call `Models` to verify that the 80% threshold has not been reached. If
					// it has been reached, then generate a summary of the current story part and
					// save it to the next story part.

					part.context_messages
						.iter()
						.map(|msg: &ContextMessage| {
							Ok(ChatCompletionRequestMessageArgs::default()
								.content(msg.content.clone())
								.role(match msg.role.as_str() {
									"system" => Role::System,
									"assistant" => Role::Assistant,
									"user" => Role::User,
									_ =>
										return Err(WeaveError::Custom(format!(
											"Invalid role: {}",
											msg.role
										))),
								})
								.name(msg.username.as_deref().unwrap_or_default())
								.build()
								.unwrap())
						})
						.collect::<Result<Vec<ChatCompletionRequestMessage>, WeaveError>>()
				},
				None => Ok(vec![]),
			}
			.unwrap_or_default()
		});

		let max_words = tokio::spawn(async move {
			Loreweaver::<T>::max_words(
				model,
				None,
				match story_part_clone.as_ref() {
					Some(part) => part.context_tokens,
					None => 0,
				},
			)
		});

		let mut context_messages = context_messages.await.map_err(|_e| {
			WeaveError::Custom("Failed to complete context messages thread exection".into())
		})?;

		context_messages.extend(
			vec![ChatCompletionRequestMessageArgs::default()
				.role(Role::User)
				.content(msg)
				.name(format!("{}{}", username, pseudo_username.unwrap_or_default()))
				.build()
				.map_err(|err| {
					return Err::<String, WeaveError>(WeaveError::Custom(err.to_string()))
				})
				.expect("Failed to build ChatCompletionRequestMessage")]
			.into_iter(),
		);

		info!("Prompting ChatGPT...");

		match <Loreweaver<T> as secret_lore::Sealed<T>>::do_prompt(
			Models::GPT4,
			&mut context_messages,
			max_words.await.map_err(|_e| {
				WeaveError::Custom("Failed to complete max words thread exection".into())
			})?,
		)
		.await
		{
			Ok(res) => Ok(match res.choices[0].clone().message.content {
				Some(content) => content,
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
	#[derive(PartialEq, Eq, Clone, Debug)]
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
