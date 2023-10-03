use serenity::{
	model::prelude::{
		application_command::ApplicationCommandInteraction,
		interaction::application_command::{CommandDataOption, CommandDataOptionValue},
		InteractionResponseType,
	},
	prelude::Context,
};
use tracing::{error, instrument};

pub fn get_command_option(
	options: &Vec<CommandDataOption>,
	option: &str,
) -> Result<Option<String>, serenity::Error> {
	options
		.iter()
		.find(|o| o.name == option)
		.map(|o| {
			extract_string_option_value(o).map(Some).ok_or_else(|| {
				error!("Option not found: {}", option);
				serenity::Error::Other("Option not found".into())
			})
		})
		.map_or_else(|| Ok(None), |result| result)
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
) -> Result<(), serenity::Error> {
	error!("The following command is not known: {:?}", command);
	command
		.create_interaction_response(&ctx.http, |response| {
			response
				.kind(InteractionResponseType::ChannelMessageWithSource)
				.interaction_response_data(|message| message.content("Unknown command"))
		})
		.await
		.map_err(|e| {
			error!("Failed to create interaction response: {}", e);
			e
		})
}
