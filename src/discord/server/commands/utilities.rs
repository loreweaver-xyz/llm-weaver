use serenity::{
	model::prelude::{
		application_command::ApplicationCommandInteraction,
		interaction::application_command::{CommandDataOption, CommandDataOptionValue},
		InteractionResponseType,
	},
	prelude::Context,
};
use tracing::{error, instrument};

use crate::loreweaver::WeaveError;

pub fn get_command_option(
	options: &Vec<CommandDataOption>,
	option: &str,
	required: bool,
) -> Result<Option<String>, WeaveError> {
	options
		.iter()
		.find(|o| o.name == option)
		.map(|o| match extract_string_option_value(o) {
			Some(value) => Ok(Some(value)),
			None =>
				if required {
					Err(WeaveError::Custom(format!("Option {} is required", option)))
				} else {
					Ok(None)
				},
		})
		.unwrap_or_else(|| {
			if required {
				Err(WeaveError::Custom(format!("Option {} is required", option)))
			} else {
				Ok(None)
			}
		})
}
pub fn extract_string_option_value(option: &CommandDataOption) -> Option<String> {
	match option.resolved.as_ref().unwrap() {
		CommandDataOptionValue::String(value) => Some(value.clone()),
		_ => None,
	}
}

#[instrument(skip(ctx, command))]
pub(crate) async fn command_not_implemented(
	ctx: &Context,
	command: &ApplicationCommandInteraction,
) -> Result<(), WeaveError> {
	error!("The following command is not known: {:?}", command);
	command
		.create_interaction_response(&ctx.http, |response| {
			response
				.kind(InteractionResponseType::ChannelMessageWithSource)
				.interaction_response_data(|message| message.content("Unknown command"))
		})
		.await
		.map_err(|err| {
			error!("Interaction response failed: {}", err);
			WeaveError::Custom("Failed to execute interaction response".into())
		})
}
