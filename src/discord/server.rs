use core::panic;
use serenity::{
	async_trait,
	http::Http,
	model::prelude::{command::Command, interaction::Interaction, Message, Ready},
	prelude::{Context, EventHandler, GatewayIntents},
	Client,
};
use std::{env, fmt::Display};
use tokio::time::Instant;
use tracing::{error, event, info, instrument, Level};

use crate::{
	loreweaver::{self, types::SystemCategory, Config, Loom, Loreweaver},
	storage::Storage,
	GPTModel, Server,
};

mod commands;

/// Encompasses many [`StoryID`]s.
type ServerID = u64;
/// The `StoryID` identifies a single story which is part of a [`ServerID`].
type StoryID = u64;
/// Type alias encompassing a server id and a story id.
///
/// Used mostly for querying some blob storage in the form of a path.
#[derive(Debug)]
pub struct WeavingID {
	pub server_id: ServerID,
	pub story_id: StoryID,
}

// TODO: Figure out how to get around this.
impl loreweaver::WeavingID for WeavingID {
	fn base_key(&self) -> String {
		format!("{}:{}", self.server_id, self.story_id)
	}
}

impl Display for WeavingID {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		write!(f, "{}:{}", self.server_id, self.story_id)
	}
}

#[derive(Default, Debug)]
pub struct DiscordBot;

impl Config for DiscordBot {
	type Model = GPTModel;
	type Storage = Storage;
	type WeavingID = WeavingID;
}

impl Server for DiscordBot {
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

				// Start bot "typing..." in discord
				if let Err(err) = command.defer(&ctx.http).await {
					error!("Cannot defer command: {err:?}");
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

	// Set a handler for the `message` event - so that whenever a new message
	// is received - the closure (or function) passed will be called.
	//
	// Event handlers are dispatched through a threadpool, and so multiple
	// events can be dispatched simultaneously.
	#[instrument(skip(self, ctx), fields(time_to_completion))]
	async fn message(&self, ctx: Context, msg: Message) {
		// Ignore messages from self
		if msg.author.bot {
			return
		}

		let maybe_category_channel = match msg.channel_id.to_channel(&ctx).await {
			Ok(channel) => match channel.guild().map(|guild_channel| guild_channel.parent_id) {
				Some(Some(parent_id)) =>
					Some(parent_id.to_channel(&ctx).await.unwrap().category().unwrap()),
				_ => None,
			},
			Err(_) => None,
		};

		// validate that the category is the Stories category
		let category_channel = match maybe_category_channel {
			Some(category_channel) => {
				// TODO: why do we need double ref here
				if !SystemCategory::pretty_categories().contains(&category_channel.name.as_str()) {
					error!("Channel is not in the Stories category");
					return
				}

				category_channel
			},
			None => {
				error!("Channel is not in a category");
				return
			},
		};

		let start = Instant::now();

		let typing = match msg.channel_id.start_typing(&ctx.http) {
			Ok(typing) => typing,
			Err(_) => {
				error!("Failed to start typing");
				return
			},
		};

		let content = match Loreweaver::<Self>::prompt(
			SystemCategory::pretty_to_system(&category_channel.name),
			WeavingID { server_id: msg.guild_id.unwrap().0, story_id: msg.channel_id.0 },
			msg.content.clone(),
			12345,
			msg.author.clone().name,
			msg.author.nick_in(&ctx, msg.guild_id.unwrap()).await,
		)
		.await
		{
			Ok(c) => c,
			Err(e) => {
				error!("Cannot prompt loreweaver: {e:?}");
				return
			},
		};

		if let Err(err) = msg.channel_id.say(&ctx.http, content).await {
			error!("Cannot send message: {err:?}");
		}

		let _ = typing.stop();

		info!("Command executed in {} seconds", start.elapsed().as_secs());

		tracing::Span::current().record("time_to_completion", start.elapsed().as_secs());
	}
}
