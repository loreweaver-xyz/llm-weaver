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
