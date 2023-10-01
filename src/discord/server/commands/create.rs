use serenity::{
	builder::{CreateApplicationCommand, CreateApplicationCommandOption},
	model::prelude::{
		application_command::ApplicationCommandInteraction, command::CommandOptionType, ChannelType,
	},
	prelude::Context,
};

use crate::loreweaver::WeaveError;

use super::utilities::extract_string_option_value;

pub async fn run(
	ctx: &Context,
	command: &ApplicationCommandInteraction,
) -> Result<String, WeaveError> {
	let guild_id = &command.guild_id.unwrap();
	let user = &command.user;
	let options = &command.data.options;

	let story_size = extract_string_option_value(&options[0]).unwrap();
	let hardcore_mode = extract_string_option_value(&options[1]).unwrap();
	let private_story = extract_string_option_value(&options[2]).unwrap();
	let story_theme_1 = extract_string_option_value(&options[3]).unwrap();
	let story_theme_2 = match options.len() > 4 {
		true => Some(extract_string_option_value(&options[4]).unwrap()),
		false => None,
	};
	let story_name = match options.len() > 5 {
		true => Some(extract_string_option_value(&options[5]).unwrap()),
		false => None,
	};
	let story_description = match options.len() > 6 {
		true => Some(extract_string_option_value(&options[6]).unwrap()),
		false => None,
	};

	guild_id
		.create_channel(ctx.http.clone(), |c| {
			c.name(story_name.unwrap())
				.permissions(vec![])
				.kind(ChannelType::Text)
				.rate_limit_per_user(10)
				.category(1234u64)
		})
		.await
		.map_err(|_e| "Failed to create channel")
		.map_err(|e| WeaveError::Custom(e.to_string()))?;

	Ok("Created story channel".to_string())
}

pub fn register(command: &mut CreateApplicationCommand) -> &mut CreateApplicationCommand {
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
			create_theme_option(option, "story-theme-1", "The theme of the story", true)
		})
		.create_option(|option| {
			create_theme_option(
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
