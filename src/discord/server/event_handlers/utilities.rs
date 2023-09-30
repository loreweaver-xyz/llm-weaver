use serenity::model::prelude::interaction::application_command::{
	CommandDataOption, CommandDataOptionValue,
};

pub fn _extract_string_option_value(option: &CommandDataOption) -> Option<String> {
	match option.resolved.as_ref().unwrap() {
		CommandDataOptionValue::String(value) => Some(value.clone()),
		_ => None,
	}
}
