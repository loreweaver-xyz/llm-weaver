use std::{env, marker::PhantomData};

use async_openai::{
	error::OpenAIError,
	types::{
		ChatCompletionRequestMessage, ChatCompletionRequestMessageArgs,
		CreateChatCompletionResponse, Role,
	},
};
use google_cloud_storage::{
	client::{Client, ClientConfig},
	http::objects::{download::Range, get::GetObjectRequest, list::ListObjectsRequest},
};
use serenity::{async_trait, json::prelude::Serialize, model::prelude::Deserialize};
use tokio::sync::OnceCell;

use self::models::{MaxTokens, Models};

pub trait Get<T> {
	fn get() -> T;
}

/// A trait consisting mainly of associated types implemented by [`Loreweaver`].
///
/// Normally structs implementing [`crate::Server`] would implement this trait to call methods
/// implemented by [`Loreweaver`]
#[async_trait]
pub trait Config {
	/// Getter for GPT model to use.
	type Model: Get<Models>;
}

/// Encompasses many [`StoryID`]s.
type ServerID = u128;
/// The `StoryID` identifies a single story which is part of a [`ServerID`].
type StoryID = u128;
/// Type alias encompassing a server id and a story id.
///
/// Used mostly for querying some blob storage in the form of a path.
type WeavingID = (ServerID, StoryID);

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
	/// Prompt ChatGPT with a `msg_history` and a `new_msg`.
	///
	/// If 80% of the maximum number of tokens allowed in a message history for the configured
	/// ChatGPT [`Models`] has been reached, a summary will be generated instead of the current
	/// message history and saved to the cloud. A new message history will begin.
	async fn prompt(loom_id: WeavingID) -> Result<String, WeaveError>;
}

/// The bread & butter of Loreweaver.
///
/// All core functionality is implemented by this struct.
pub struct Loreweaver<T: Config>(PhantomData<T>);

impl<T: Config> private::Sealed<T> for Loreweaver<T> {}

/// Storage client to access GCP Storage
static STORAGE_CLIENT: OnceCell<Client> = OnceCell::const_new();
/// Storage bucket in GCP Storage to store all message histories
static STORAGE_BUCKET: OnceCell<String> = OnceCell::const_new();

/// An platform agnostic type representing a user's account ID.
type AccountId = u64;

/// Context message that represent a single message in a [`StoryPart`].
#[derive(Default, Clone, Debug, Serialize, Deserialize)]
pub struct ContextMessage {
	pub role: String,
	pub user_id: Option<String>,
	pub username: Option<String>,
	pub content: String,
	pub timestamp: String,
}

/// Represents a single part of a story containing a list of messages along with other metadata.
///
/// ChatGPT can only hold a limited amount of tokens in a the entire message history/context.
/// Therefore, at every [`Loom::prompt`] execution, we must keep track of the number of tokens
/// in the current story part and if it exceeds the maximum number of tokens allowed for the
/// current GPT model, then we must generate a summary of the current story part and use that
/// as the starting point for the next story part. This is one of the biggest challenges for
/// Loreweaver to keep a consistent narrative throughout the many story parts.
#[derive(Debug, Serialize, Deserialize, Default)]
pub struct StoryPart {
	/// The story part number.
	pub number: u16,
	/// Total number of _GPT tokens_ in the story part.
	pub total_tokens: u128,
	/// Number of players that are part of the story. (typically this changes based on players
	/// entering the commands `/leave` or `/join`).
	///
	/// When generating a new story part (N + 1, where N is the current story part number), we need
	/// to copy over the same number of players. The story must remain consistent throughout each
	/// part.
	pub players: Vec<AccountId>,
	/// List of [`ContextMessage`]s in the story part.
	pub context_messages: Vec<ContextMessage>,
}

impl<T: Config> Loreweaver<T> {
	/// Maximum number of words to return in a response based on maximum tokens of GPT model or a
	/// `custom` supplied value.
	///
	/// Every token equates to 75% of a word.
	fn max_words(
		model: Models,
		custom_max_tokens: Option<MaxTokens>,
		tokens_in_context: MaxTokens,
	) -> MaxTokens {
		let max_tokens = custom_max_tokens
			.unwrap_or(Models::default_max_response_tokens(&model, tokens_in_context));

		(max_tokens as f64 * 0.75) as MaxTokens
	}

	/// Download _last_ story part from GCP Storage.
	///
	/// None is returned if there are no story parts for the given `weaving_id`. This means that
	/// the story has not been started yet.
	async fn get_story(weaving_id: WeavingID) -> Result<Option<StoryPart>, WeaveError> {
		let client: &Client = STORAGE_CLIENT
			.get_or_init(async || {
				let config = ClientConfig::default().with_auth().await.unwrap();
				Client::new(config)
			})
			.await;

		let bucket = STORAGE_BUCKET
			.get_or_init(async || env::var("STORAGE_BUCKET").unwrap())
			.await
			.to_owned();

		let parts = client
			.list_objects(&ListObjectsRequest { bucket: bucket.clone(), ..Default::default() })
			.await
			.unwrap()
			.items
			.unwrap_or_default();

		let maybe_last_part = parts.last();

		match maybe_last_part {
			None => Ok(None),
			Some(last_part) => {
				let bytes = client
					.download_object(
						&GetObjectRequest {
							// TODO: Try not to use async
							bucket,
							object: format!(
								"{}/{}/{}.json",
								weaving_id.0, weaving_id.1, last_part.name
							),
							..Default::default()
						},
						&Range::default(),
					)
					.await
					.unwrap();

				Ok(Some(serde_json::from_slice(&bytes).unwrap()))
			},
		}
	}
}

#[derive(Debug)]
pub enum WeaveError {
	OpenAIError(OpenAIError),
	Custom(String),
}

#[async_trait]
impl<T: Config> Loom for Loreweaver<T> {
	async fn prompt(weaving_id: WeavingID) -> Result<String, WeaveError> {
		let model = T::Model::get();

		let tokens_in_context = Loreweaver::<T>::get_story(weaving_id);

		let max_words = Loreweaver::<T>::max_words(model, None, tokens_in_context);

		let mut messages: Vec<ChatCompletionRequestMessage> =
			vec![ChatCompletionRequestMessageArgs::default()
				.role(Role::User)
				.content("Hello! How are you?")
				.name("Michael")
				.build()
				.map_err(|err| {
					return Err::<String, WeaveError>(WeaveError::Custom(err.to_string()))
				})
				.unwrap()];

		match <Loreweaver<T> as private::Sealed<T>>::do_prompt(
			Models::GPT4,
			&mut messages,
			max_words,
		)
		.await
		{
			Ok(res) => Ok(res.choices[0].clone().message.content.unwrap()),
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

mod private {
	use crate::loreweaver::Get;
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
		static ref LOREWEAVER_SYSTEM: RwLock<System> = {
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
					.content(LOREWEAVER_SYSTEM.read().await.clone().0)
					.role(Role::System)
					.build()
					.unwrap()],
			);

			let tokens = msgs
				.iter()
				.skip(1)
				.map(|msg: &ChatCompletionRequestMessage| {
					msg.content.as_ref().map(|content| content.count_tokens()).unwrap_or(0)
				})
				.sum::<MaxTokens>();

			let request = CreateChatCompletionRequestArgs::default()
				// TODO: Set max tokens response based on user subscription
				// TODO: default max token response is a u128 but we would potentially lose some
				// bytes from this primitive type conversion to u16
				.max_tokens((T::Model::get().max_context_tokens() - tokens) as u16)
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
