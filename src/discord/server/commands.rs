use crate::loreweaver::WeaveError;
use serenity::builder::CreateApplicationCommand;
use serenity::model::prelude::application_command::ApplicationCommandInteraction;
use serenity::model::prelude::command::Command;
use serenity::model::prelude::InteractionResponseType;
use serenity::prelude::Context;
use tracing::{error, info};

use super::DiscordBot;

impl DiscordBot {
    pub(crate) async fn setup_commands(ctx: &Context) -> Result<(), ()> {
        // Register commands functions
        type RegisterFn =
            fn(command: &mut CreateApplicationCommand) -> &mut CreateApplicationCommand;

        // All commands that will be registered
        let registers: Vec<RegisterFn> = vec![Self::register_create];

        for f in registers {
            if let Err(err) =
                Command::create_global_application_command(&ctx.http, |command| f(command)).await
            {
                error!("Cannot create command: {err:?}");
            }
        }

        info!("Loreweaver bot commands bootstrapped successfully");
        Ok(())
    }

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
}
