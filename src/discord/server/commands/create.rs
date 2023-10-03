use serenity::{
	builder::{CreateApplicationCommand, CreateApplicationCommandOption},
	model::prelude::{
		application_command::ApplicationCommandInteraction, command::CommandOptionType, ChannelType,
	},
	prelude::Context,
};
use tracing::error;

use super::utilities::get_command_option;

pub async fn run(
	ctx: &Context,
	command: &ApplicationCommandInteraction,
) -> Result<String, serenity::Error> {
	let guild_id = &command.guild_id.unwrap();
	let options = &command.data.options;

	let story_size = get_command_option(options, STORY_SIZE_OPTION)?.unwrap();
	let _hardcore_mode = get_command_option(options, HARDCORE_MODE_OPTION)?.unwrap();
	let _private_story = get_command_option(options, PRIVATE_STORY_OPTION)?.unwrap();
	let _story_theme_1 = get_command_option(options, STORY_THEME_1_OPTION)?.unwrap();
	let _story_theme_2 = get_command_option(options, STORY_THEME_2_OPTION)?.unwrap();
	let story_name = get_command_option(options, STORY_NAME_OPTION)?.unwrap();
	let _story_description = get_command_option(options, STORY_DESCRIPTION_OPTION).unwrap();

	// Check if category exists
	let channels = guild_id.channels(ctx).await.map_err(|e| {
		error!("Failed to get channels: {}", e);
		e
	})?;

	let category_name = format!("{}-stories", story_size);

	let category_id = match channels
		.iter()
		.filter(|c| c.1.kind == ChannelType::Category)
		.find(|c| c.1.name == category_name)
	{
		Some(c) => *c.0,
		None =>
			guild_id
				.create_channel(ctx.http.clone(), |c| {
					c.name(category_name).kind(ChannelType::Category).permissions(vec![])
				})
				.await
				.map_err(|e| {
					error!("Failed to create category: {}", e);
					e
				})?
				.id,
	};

	guild_id
		.create_channel(ctx.http.clone(), |c| {
			c.name(story_name)
				.permissions(vec![])
				.kind(ChannelType::Text)
				.rate_limit_per_user(10)
				.category(category_id)
		})
		.await
		.map_err(|e| {
			error!("Failed to create story channel: {}", e);
			e
		})?;

	Ok("Created story channel".to_string())
}

const STORY_SIZE_OPTION: &str = "story-size";
const HARDCORE_MODE_OPTION: &str = "hardcore-mode";
const PRIVATE_STORY_OPTION: &str = "private-story";
const STORY_THEME_1_OPTION: &str = "story-theme-1";
const STORY_THEME_2_OPTION: &str = "story-theme-2";
const STORY_NAME_OPTION: &str = "story-name";
const STORY_DESCRIPTION_OPTION: &str = "story-description";

pub fn register(command: &mut CreateApplicationCommand) -> &mut CreateApplicationCommand {
	command
		.name("create")
		.description("Create a new story")
		.create_option(|option| {
			option
				.name(STORY_SIZE_OPTION)
				.description("The size and complexity of the story")
				.kind(CommandOptionType::String)
				.add_string_choice("Small", "small")
				.add_string_choice("Medium", "medium")
				.add_string_choice("Large", "large")
				.required(true)
		})
		.create_option(|option| {
			option
				.name(HARDCORE_MODE_OPTION)
				.description("Enable or disable hardcore mode")
				.kind(CommandOptionType::String)
				.add_string_choice("Yes", "yes")
				.add_string_choice("No", "no")
				.required(true)
		})
		.create_option(|option| {
			option
				.name(PRIVATE_STORY_OPTION)
				.description("Enable or disable private story")
				.kind(CommandOptionType::String)
				.add_string_choice("Yes", "yes")
				.add_string_choice("No", "no")
				.required(true)
		})
		.create_option(|option| {
			create_theme_option(option, STORY_THEME_1_OPTION, "The theme of the story", true)
		})
		.create_option(|option| {
			create_theme_option(
				option,
				STORY_THEME_2_OPTION,
				"Combine two themes for a more complex story",
				true,
			)
		})
		.create_option(|option| {
			option
				.name(STORY_NAME_OPTION)
				.description("The name of the story")
				.kind(CommandOptionType::String)
				.required(true)
		})
		.create_option(|option| {
			option
				.name(STORY_DESCRIPTION_OPTION)
				.description("The description of the story")
				.kind(CommandOptionType::String)
				.required(true)
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
