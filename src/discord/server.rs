use core::panic;
use serenity::{
	async_trait,
	http::Http,
	model::prelude::{command::Command, interaction::Interaction, Ready},
	prelude::{Context, EventHandler, GatewayIntents},
	Client,
};
use std::env;
use tokio::time::Instant;
use tracing::{error, event, info, instrument, Level};

use crate::{
	loreweaver::{Config, Loom, Loreweaver},
	storage::Storage,
	GPTModel, Server,
};

mod commands;

#[derive(Default, Debug)]
pub struct DiscordBot;

impl Config for DiscordBot {
	type Model = GPTModel;
	type Storage = Storage;
}

impl<T: Loom> Server<T> for DiscordBot {
	#[instrument]
	async fn serve() {
		// Configure the client with your Discord bot token in the environment.
		let token = env::var("DISCORD_TOKEN").expect("Expected discord token in the environment");

		let http = Http::new(&token);
		let discord_id = http.get_current_user().await.expect("Expected to fetch current user");

		// Set gateway intents, which decides what events the bot will be notified about
		let intents = GatewayIntents::GUILD_MESSAGES |
			GatewayIntents::DIRECT_MESSAGES |
			GatewayIntents::MESSAGE_CONTENT;

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
	#[instrument(skip(self, ctx, ready))]
	async fn ready(&self, ctx: Context, ready: Ready) {
		println!("{} is connected!", ready.user.name);

		if let Err(e) = Command::create_global_application_command(&ctx.http, |command| {
			commands::create::register(command)
		})
		.await
		{
			error!("Cannot create command: {e:?}");
			panic!("Cannot create command");
		}

		info!("Loreweaver bot commands bootstrapped successfully");
	}

	#[instrument(skip(self, ctx, interaction), fields(time_to_completion))]
	async fn interaction_create(&self, ctx: Context, interaction: Interaction) {
		match interaction {
			Interaction::ApplicationCommand(command) => {
				let start = Instant::now();

				// Enables bot "typing..." in discord
				if let Err(err) = command.defer(&ctx.http).await {
					error!("Cannot defer command: {err:?}");
					return
				}

				if let Err(e) = Loreweaver::<Self>::prompt(
					(100, 101),
					"Hello Loreweaver!".to_string(),
					12345,
					command.user.clone().name,
					command.user.nick_in(&ctx, command.guild_id.unwrap()).await,
				)
				.await
				{
					error!("Cannot prompt loreweaver: {e:?}");
					return
				}

				// Parse and execute the command
				let maybe_content = match command.data.name.as_str() {
					"create" => commands::create::run(&ctx, &command).await,
					_ =>
						return commands::utilities::command_not_implemented(&ctx, &command)
							.await
							.expect("Failed to submit not-implemented error"),
				};

				let content = match maybe_content {
					Ok(content) => content,
					Err(e) => {
						error!("Failed to execute command: {e:?}");
						"Oh! Something went wrong, I wasn't able to create the story channel in the discord server. Please checkout out our troubleshooting docs!".to_string()
					},
				};

				if let Err(err) = command
					.edit_original_interaction_response(&ctx.http, |response| {
						response.content(content)
					})
					.await
				{
					error!("Cannot edit original interaction response: {err:?}");
				}

				info!("Command executed in {} seconds", start.elapsed().as_secs());

				tracing::Span::current().record("time_to_completion", start.elapsed().as_secs());
			},
			_ => {},
		}
	}
}
