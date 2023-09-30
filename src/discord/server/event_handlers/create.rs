use serenity::{
	builder::{CreateApplicationCommand, CreateApplicationCommandOption},
	model::{
		prelude::{
			application_command::CommandDataOption, command::CommandOptionType, GuildChannel,
			GuildId,
		},
		user::User,
	},
	prelude::Context,
};

use crate::discord::server::DiscordBot;

use super::utilities::_extract_string_option_value;

impl DiscordBot {
	// pub async fn run_create(
	// 	ctx: &Context,
	// 	guild_id: &GuildId,
	// 	user: &User,
	// 	options: &[CommandDataOption],
	// ) -> Result<GuildChannel, Box<dyn std::error::Error>> {
	// 	let story_size = _extract_string_option_value(&options[0]).unwrap();
	// 	let hardcore_mode = _extract_string_option_value(&options[1]).unwrap();
	// 	let private_story = _extract_string_option_value(&options[2]).unwrap();
	// 	let story_theme_1 = _extract_string_option_value(&options[3]).unwrap();
	// 	let story_theme_2 = match options.len() > 4 {
	// 		true => Some(_extract_string_option_value(&options[4]).unwrap()),
	// 		false => None,
	// 	};
	// 	let story_name = match options.len() > 5 {
	// 		true => Some(_extract_string_option_value(&options[5]).unwrap()),
	// 		false => None,
	// 	};
	// 	let story_description = match options.len() > 6 {
	// 		true => Some(_extract_string_option_value(&options[6]).unwrap()),
	// 		false => None,
	// 	};

	// 	Ok(guild_id
	// 		.create_channel(ctx.http.clone(), |c| {
	// 			c.name(name)
	// 				.permissions(permissions)
	// 				.kind(kind)
	// 				.rate_limit_per_user(10)
	// 				.category(1234u64)
	// 		})
	// 		.await
	// 		.map_err(|_e| "Failed to create channel")?)
	// }

	pub fn register_create(
		command: &mut CreateApplicationCommand,
	) -> &mut CreateApplicationCommand {
		command
			.name("create")
			.description("Create a new story")
			.create_option(|option| {
				option
					.name("story-size")
					.description("The size and complexity of the story")
					.kind(CommandOptionType::String)
					.add_string_choice("Small", "small")
					.add_string_choice("Medium", "medium")
					.add_string_choice("Large", "large")
					.required(true)
			})
			.create_option(|option| {
				option
					.name("hardcore-mode")
					.description("Enable or disable hardcore mode")
					.kind(CommandOptionType::String)
					.add_string_choice("Yes", "yes")
					.add_string_choice("No", "no")
					.required(true)
			})
			.create_option(|option| {
				option
					.name("private-story")
					.description("Enable or disable private story")
					.kind(CommandOptionType::String)
					.add_string_choice("Yes", "yes")
					.add_string_choice("No", "no")
					.required(true)
			})
			.create_option(|option| {
				Self::create_theme_option(option, "story-theme-1", "The theme of the story", true)
			})
			.create_option(|option| {
				Self::create_theme_option(
					option,
					"story-theme-2",
					"Combine two themes for a more complex story",
					false,
				)
			})
			.create_option(|option| {
				option
					.name("story-name")
					.description("The name of the story")
					.kind(CommandOptionType::String)
					.required(false)
			})
			.create_option(|option| {
				option
					.name("story-description")
					.description("The description of the story")
					.kind(CommandOptionType::String)
					.required(false)
			})
	}

	fn create_theme_option<'a>(
		option: &'a mut CreateApplicationCommandOption,
		name: &str,
		description: &str,
		required: bool,
	) -> &'a mut CreateApplicationCommandOption {
		option
			.name(name)
			.description(description)
			.kind(CommandOptionType::String)
			.add_string_choice("Fantasy", "fantasy")
			.add_string_choice("Sci-Fi", "scifi")
			.add_string_choice("Horror", "horror")
			.add_string_choice("Western", "western")
			.add_string_choice("Mystery", "mystery")
			.add_string_choice("Superhero", "superhero")
			.add_string_choice("Historical", "historical")
			.add_string_choice("Modern", "modern")
			.add_string_choice("Post-Apocalyptic", "postapocalyptic")
			.add_string_choice("Romance", "romance")
			.add_string_choice("Comedy", "comedy")
			.add_string_choice("Drama", "drama")
			.add_string_choice("Action", "action")
			.add_string_choice("Story", "story")
			.add_string_choice("Thriller", "thriller")
			.required(required)
	}
}
