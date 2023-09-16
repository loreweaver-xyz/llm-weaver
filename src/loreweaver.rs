use std::marker::PhantomData;

use async_openai::{
    error::OpenAIError,
    types::{
        ChatCompletionRequestMessage, ChatCompletionRequestMessageArgs,
        CreateChatCompletionResponse, Role,
    },
};
use num_traits::Unsigned;
use serenity::async_trait;

use self::models::{MaxTokens, Models};

pub trait Get<T> {
    fn get() -> T;
}

/// A trait consisting mainly of associated types implemented by [`Loreweaver`].
///
/// Normally structs implementing [`crate::Server`] would implement this trait to call methods implemented by [`Loreweaver`]
#[async_trait]
pub trait Config {
    /// Getter for GPT model to use.
    type Model: Get<Models>;
    /// Determines the maximum number of tokens to return in a response from ChatGPT
    type MaxTokenResponse: Unsigned;
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
/// This is the machine that drives all of the core methods that should be used across any service that needs to prompt ChatGPT
/// and receive a response.
///
/// The implementations should handle all form of validation and usage tracking all while abstracting the logic from the services calling them.
#[async_trait]
pub trait Loom {
    async fn prompt(loom_id: WeavingID) -> Result<String, WeaveError>;
}

/// The bread & butter of Loreweaver.
///
/// All core functionality is implemented by this struct.
pub struct Loreweaver<T: Config>(PhantomData<T>);

impl<T: Config> private::Sealed<T> for Loreweaver<T> {}

impl<T: Config> Loreweaver<T> {
    /// Maximum number of words to return in a response based on maximum tokens of GPT model or a `custom` supplied value.
    ///
    /// Every token equates to 75% of a word.
    fn max_words(
        model: Models,
        custom_max_tokens: Option<MaxTokens>,
        tokens_in_context: MaxTokens,
    ) -> MaxTokens {
        let max_tokens = custom_max_tokens.unwrap_or(Models::default_max_response_tokens(
            &model,
            tokens_in_context,
        ));

        (max_tokens as f64 * 0.75) as MaxTokens
    }

    /// Get story from blob storage and return an instance of it
    fn get_story(_weaving_id: WeavingID) -> MaxTokens {
        // todo!("This should query a generic blob storage impl and then create an instance of a struct called `Story` that holds all the data.")
        100
    }
}

#[derive(Debug)]
pub enum WeaveError {
    MaximumTokensExceeded,
    OpenAIError(OpenAIError),
    Custom(String),
}

#[async_trait]
impl<T: Config> Loom for Loreweaver<T> {
    async fn prompt(weaving_id: WeavingID) -> Result<String, WeaveError> {
        let model = T::Model::get();

        let tokens_in_context = Loreweaver::<T>::get_story(weaving_id);

        let max_words = Loreweaver::<T>::max_words(model, None, tokens_in_context);

        let messages: Vec<ChatCompletionRequestMessage> =
            vec![ChatCompletionRequestMessageArgs::default()
                .role(Role::User)
                .content("Hello! How are you?")
                .name("Michael")
                .build()
                .map_err(|err| {
                    return Err::<String, WeaveError>(WeaveError::Custom(err.to_string()));
                })
                .unwrap()];

        let result: Result<CreateChatCompletionResponse, OpenAIError> =
            <Loreweaver<T> as private::Sealed<T>>::do_prompt(Models::GPT4, messages, max_words)
                .await;

        let result = result.unwrap().choices[0].clone().message.content.unwrap();

        Ok(result)
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
        /// This would normally be used when prompting ChatGPT API and specifying the maximum tokens to return.
        ///
        /// `tokens_in_context` parameter is the current number of tokens that are part of the context. This should not surpass the [`max_context_tokens`]
        pub fn default_max_response_tokens(
            model: &Models,
            tokens_in_context: MaxTokens,
        ) -> MaxTokens {
            (model.max_context_tokens() - tokens_in_context) / 3
        }

        /// Maximum number of tokens that can be processed at once by ChatGPT.
        fn max_context_tokens(&self) -> MaxTokens {
            match self {
                Self::GPT3 => 4_096,
                Self::GPT4 => 8_192,
            }
        }
    }
}

mod private {
    use async_openai::{
        config::OpenAIConfig,
        error::OpenAIError,
        types::{
            ChatCompletionRequestMessage, CreateChatCompletionRequestArgs,
            CreateChatCompletionResponse,
        },
    };
    use tiktoken_rs::p50k_base;
    use tokio::sync::RwLock;

    use super::{
        models::{MaxTokens, Models},
        Config,
    };

    lazy_static::lazy_static! {
        /// The OpenAI client to interact with the OpenAI API.
        static ref OPENAI_CLIENT: RwLock<async_openai::Client<OpenAIConfig>> = RwLock::new(async_openai::Client::new());
    }

    pub trait Sealed<T: Config> {
        /// The action to query ChatGPT with the supplied configurations and messages.
        async fn do_prompt(
            model: Models,
            msgs: impl Into<Vec<ChatCompletionRequestMessage>> + Send + Clone,
            _max_words: MaxTokens,
        ) -> Result<CreateChatCompletionResponse, OpenAIError> {
            let tokens = msgs
                .clone()
                .into()
                .iter()
                .map(|msg: &ChatCompletionRequestMessage| {
                    msg.content
                        .as_ref()
                        .map(|content| content.count_tokens())
                        .unwrap_or(0)
                })
                .sum::<MaxTokens>();

            let request = CreateChatCompletionRequestArgs::default()
                // TODO: Set max tokens response based on user subscription
                // TODO: default max token response is a u128 but we would potentially lose some bytes from this primitive type conversion to u16
                .max_tokens(Models::default_max_response_tokens(&model, tokens) as u16)
                .model(model.name())
                .messages(msgs)
                .build()?;

            OPENAI_CLIENT.read().await.chat().create(request).await
        }
    }

    /// Tokens are a ChatGPT concept which represent normally a third of a word (or 75%).
    ///
    /// This trait auto implements some basic utility methods for counting the number of tokens from a string.
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
