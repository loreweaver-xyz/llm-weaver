use serenity::model::prelude::application_command::ApplicationCommandInteraction;
use serenity::model::prelude::InteractionResponseType;
use serenity::prelude::Context;
use tracing::error;
use crate::loreweaver::WeaveError;

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
            WeaveError::Custom("Failed to execute interaction response".to_string())
        })
}