use core::panic;
use serenity::http::Http;
use serenity::model::prelude::interaction::Interaction;
use serenity::model::prelude::Ready;
use serenity::prelude::{Context, EventHandler, GatewayIntents};
use serenity::{async_trait, Client};
use std::env;
use tracing::{error, Level};
use tracing::{event, instrument};

use crate::loreweaver::Loreweaver;
use crate::{
    loreweaver::{Config, Loom},
    GPTModel, Server,
};

mod commands;
mod event_handlers;

#[derive(Default, Debug)]
pub struct DiscordBot;

impl Config for DiscordBot {
    type Model = GPTModel;
    type MaxTokenResponse = u64;
}

impl<T: Loom> Server<T> for DiscordBot {
    #[instrument]
    async fn serve() {
        // Configure the client with your Discord bot token in the environment.
        let token = env::var("DISCORD_TOKEN").expect("Expected discord token in the environment");

        let http = Http::new(&token);
        let discord_id = http
            .get_current_user()
            .await
            .expect("Expected to fetch current user");

        // Set gateway intents, which decides what events the bot will be notified about
        let intents = GatewayIntents::GUILD_MESSAGES
            | GatewayIntents::DIRECT_MESSAGES
            | GatewayIntents::MESSAGE_CONTENT;

        // Create a new instance of the Client, logging in as a bot. This will
        // automatically prepend your bot token with "Bot ", which is a requirement
        // by Discord for bot users.
        let mut client = Client::builder(&token, intents)
            .event_handler(DiscordBot::default())
            .await
            .expect("Error creating client");

        event!(
            Level::INFO,
            "Loreweaver discord bot started as {}#{}",
            discord_id.name,
            discord_id.discriminator
        );

        // Finally, start a single shard, and start listening to events.
        //
        // Shards will automatically attempt to reconnect, and will perform
        // exponential backoff until it reconnects.
        if let Err(why) = client.start().await {
            error!("Client error: {:?}", why);
        }
    }
}

/// Discord bot event handler implementations.
#[async_trait]
impl EventHandler for DiscordBot {
    #[instrument(skip(self, ctx, _ready))]
    async fn ready(&self, ctx: Context, _ready: Ready) {
        if let Err(e) = Self::setup_commands(&ctx).await {
            error!("Failed to register discord bot commands: {e:?}");
            panic!();
        }
    }

    #[instrument(skip(self, ctx, interaction))]
    async fn interaction_create(&self, ctx: Context, interaction: Interaction) {
        match interaction {
            Interaction::ApplicationCommand(command) => {
                // Enables bot "typing..." in discord
                if let Err(err) = command.defer(&ctx.http).await {
                    error!("Cannot defer command: {err:?}");
                    return;
                }

                // Parse and execute the command
                let maybe_content = match command.data.name.as_str() {
                    "create" => Loreweaver::<Self>::prompt((100, 101)).await,
                    _ => {
                        return Self::command_not_implemented(&ctx, &command)
                            .await
                            .expect("Failed to submit not-implemented error")
                    }
                };

                // Send the response back to the user
                if let Err(err) = command
                    .edit_original_interaction_response(&ctx.http, |response| {
                        response.content(match maybe_content {
                            Ok(msg) => msg,
                            Err(e) => {
                                error!("Failed to edit original interaction response: {e:?}");
                                "Failed to execute command".to_string()
                            }
                        })
                    })
                    .await
                {
                    error!("Cannot edit original interaction response: {err:?}");
                }
            }
            _ => {}
        }
    }
}
