use std::{collections::VecDeque, marker::PhantomData};

use num_traits::{CheckedAdd, FromPrimitive, SaturatingAdd, SaturatingSub, Zero};
use tracing::{debug, error, instrument, trace};

use crate::{
	types::{
		LoomError, PromptModelRequest, PromptModelTokens, SummaryModelTokens, VecPromptMsgsDeque,
		WeaveError, WrapperRole, ASSISTANT_ROLE, SYSTEM_ROLE,
	},
	Config, ContextMessage, Llm, LlmConfig, TapestryChestHandler, TapestryFragment, TapestryId,
};

/// The machine that drives all of the core methods that should be used across any service
/// that needs to prompt LLM and receive a response.
///
/// This is implemented over the [`Config`] trait.
#[derive(Debug)]
pub struct Loom<T: Config> {
	pub chest: T::Chest,
	_phantom: PhantomData<T>,
}

impl<T: Config> Loom<T> {
	/// Creates a new instance of `Loom`.
	pub fn new() -> Self {
		Self { chest: <T::Chest as TapestryChestHandler<T>>::new(), _phantom: PhantomData }
	}

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
	/// - `prompt_llm_config`: The [`Config::PromptModel`] to use for prompting LLM.
	/// - `summary_llm_config`: The [`Config::SummaryModel`] to use for generating summaries.
	/// - `tapestry_id`: The [`TapestryId`] to use for storing the [`TapestryFragment`] instance.
	/// - `instructions`: The instruction message to be used for the current [`TapestryFragment`]
	///   instance.
	/// - `msgs`: The messages to prompt the LLM with.
	#[instrument(skip(self, instructions, msgs))]
	pub async fn weave<TID: TapestryId>(
		&self,
		prompt_llm_config: LlmConfig<T, T::PromptModel>,
		summary_llm_config: LlmConfig<T, T::SummaryModel>,
		tapestry_id: TID,
		instructions: String,
		mut msgs: Vec<ContextMessage<T>>,
	) -> Result<(<<T as Config>::PromptModel as Llm<T>>::Response, u64, bool), LoomError> {
		let instructions_ctx_msg =
			Self::build_context_message(SYSTEM_ROLE.into(), instructions, None);
		let instructions_req_msg: PromptModelRequest<T> = instructions_ctx_msg.clone().into();

		trace!("Fetching current tapestry fragment for ID: {:?}", tapestry_id);

		let current_tapestry_fragment = self
			.chest
			.get_tapestry_fragment(tapestry_id.clone(), None)
			.await?
			.unwrap_or_default();

		// Get max token limit which cannot be exceeded in a tapestry fragment
		let max_prompt_tokens_limit = prompt_llm_config.model.get_max_prompt_token_limit();

		// Request messages which will be sent as a whole to the LLM
		let mut req_msgs = VecPromptMsgsDeque::<T, T::PromptModel>::with_capacity(
			current_tapestry_fragment.context_messages.len() + 1,
		);

		// Add instructions as the first message
		req_msgs.push_front(instructions_req_msg);

		// Convert and append all tapestry fragment messages to the request messages.
		let mut ctx_msgs = VecDeque::from(
			prompt_llm_config
				.model
				.ctx_msgs_to_prompt_requests(&current_tapestry_fragment.context_messages),
		);
		req_msgs.append(&mut ctx_msgs);

		// New messages are not added here yet since we first calculate if the new `msgs` would
		// have the tapestry fragment exceed the maximum token limit and require a summary
		// generation resulting in a new tapestry fragment.
		//
		// Either we are starting a new tapestry fragment with the instruction and summary messages
		// or we are continuing the current tapestry fragment.
		let msgs_tokens = Self::count_tokens_in_messages(msgs.iter());

		trace!(
			"Total tokens after adding new messages: {:?}, maximum allowed: {:?}",
			req_msgs.tokens.saturating_add(&msgs_tokens),
			max_prompt_tokens_limit
		);

		// Check if the total number of tokens in the tapestry fragment exceeds the maximum number
		// of tokens allowed after adding the new messages and the minimum response length.
		let does_exceeding_max_token_limit = max_prompt_tokens_limit <=
			req_msgs.tokens.saturating_add(&msgs_tokens).saturating_add(
				&PromptModelTokens::<T>::from_u64(T::MINIMUM_RESPONSE_LENGTH).unwrap(),
			);

		let (mut tapestry_fragment_to_persist, was_summary_generated) =
			if does_exceeding_max_token_limit {
				trace!("Generating summary as the token limit exceeded");

				// Summary generation should not exceed the maximum token limit of the prompt model
				// since it will be added to the tapestry fragment
				let summary_max_tokens: PromptModelTokens<T> =
					prompt_llm_config.model.max_context_length() - max_prompt_tokens_limit;

				let summary = Self::generate_summary(
					&summary_llm_config,
					&current_tapestry_fragment,
					T::convert_prompt_tokens_to_summary_model_tokens(summary_max_tokens),
				)
				.await?;

				let summary_ctx_msg = Self::build_context_message(
					SYSTEM_ROLE.into(),
					format!("\n\"\"\"\nSummary\n {}", summary),
					None,
				);

				// Truncate all tapestry fragment messages except for the instructions and add the
				// summary
				req_msgs.truncate(1);
				req_msgs.push_back(summary_ctx_msg.clone().into());

				// Create new tapestry fragment
				let mut new_tapestry_fragment = TapestryFragment::new();
				new_tapestry_fragment.push_message(summary_ctx_msg)?;

				(new_tapestry_fragment, true)
			} else {
				(current_tapestry_fragment, false)
			};

		// Add new messages to the request messages
		req_msgs.extend(msgs.iter().map(|m| m.clone().into()).collect::<Vec<_>>());

		// Tokens available for LLM response which would not exceed maximum token limit
		let max_completion_tokens = max_prompt_tokens_limit.saturating_sub(&req_msgs.tokens);

		trace!("Max completion tokens available: {:?}", max_completion_tokens);

		if max_completion_tokens.is_zero() {
			return Err(LoomError::from(WeaveError::MaxCompletionTokensIsZero).into());
		}

		trace!("Prompting LLM with request messages");

		let response = prompt_llm_config
			.model
			.prompt(
				false,
				req_msgs.tokens,
				req_msgs.into_vec(),
				&prompt_llm_config.params,
				max_completion_tokens,
			)
			.await
			.map_err(|e| {
				error!("Failed to prompt LLM: {}", e);
				e
			})?;

		// Add LLM response to the tapestry fragment messages to save
		msgs.push(Self::build_context_message(
			ASSISTANT_ROLE.into(),
			response.clone().into().unwrap_or_default(),
			None,
		));

		// Add new messages and response to the tapestry fragment which will be persisted in the
		// database
		tapestry_fragment_to_persist.extend_messages(msgs)?;

		debug!("Saving tapestry fragment: {:?}", tapestry_fragment_to_persist);

		// Save tapestry fragment to database
		// When summarized, the tapestry_fragment will be saved under a new instance
		let tapestry_fragment_id = self
			.chest
			.save_tapestry_fragment(
				&tapestry_id,
				tapestry_fragment_to_persist,
				was_summary_generated,
			)
			.await
			.map_err(|e| {
				error!("Failed to save tapestry fragment: {}", e);
				e
			})?;

		Ok((response, tapestry_fragment_id, was_summary_generated))
	}

	/// Generates the summary of the current [`TapestryFragment`] instance.
	///
	/// Returns the summary message as a string.
	async fn generate_summary(
		summary_model_config: &LlmConfig<T, T::SummaryModel>,
		tapestry_fragment: &TapestryFragment<T>,
		summary_max_tokens: SummaryModelTokens<T>,
	) -> Result<String, LoomError> {
		trace!(
			"Generating summary with max tokens: {:?}, for tapestry fragment: {:?}",
			summary_max_tokens,
			tapestry_fragment
		);

		let mut summary_generation_prompt = VecPromptMsgsDeque::<T, T::SummaryModel>::new();

		summary_generation_prompt.extend(
			summary_model_config
				.model
				.ctx_msgs_to_prompt_requests(tapestry_fragment.context_messages.as_slice()),
		);

		let res = summary_model_config
			.model
			.prompt(
				true,
				summary_generation_prompt.tokens,
				summary_generation_prompt.into_vec(),
				&summary_model_config.params,
				summary_max_tokens,
			)
			.await
			.map_err(|e| {
				error!("Failed to prompt LLM: {}", e);
				e
			})?;

		let summary_response_content = res.into();

		trace!("Generated summary: {:?}", summary_response_content);

		Ok(summary_response_content.unwrap_or_default())
	}

	/// Helper method to build a [`ContextMessage`]
	pub fn build_context_message(
		role: WrapperRole,
		content: String,
		account_id: Option<String>,
	) -> ContextMessage<T> {
		trace!("Building context message for role: {:?}, content: {}", role, content);

		ContextMessage {
			role,
			content,
			account_id,
			timestamp: chrono::Utc::now().to_rfc3339(),
			_phantom: PhantomData,
		}
	}

	fn count_tokens_in_messages(
		msgs: impl Iterator<Item = &ContextMessage<T>>,
	) -> <T::PromptModel as Llm<T>>::Tokens {
		msgs.fold(<T::PromptModel as Llm<T>>::Tokens::from_u8(0).unwrap(), |acc, m| {
			let tokens = T::PromptModel::count_tokens(&m.content).unwrap_or_default();
			match acc.checked_add(&tokens) {
				Some(v) => v,
				None => {
					error!("Token overflow");
					acc
				},
			}
		})
	}
}
