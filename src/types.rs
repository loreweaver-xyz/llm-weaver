/// System Categories
///
/// These are the different types of systems instructions which Loreweaver will base his responses.
///
/// The appropriate system instructions will be read from the `loreweaver-systems` directory and
/// injected at the beginning of the context messages.
#[derive(Debug)]
pub enum SystemCategory {
	Story(StorySize),
	RPG(StorySize),
}

impl SystemCategory {
	pub fn pretty_categories() -> Vec<&'static str> {
		vec![
			"Small Stories",
			"Medium Stories",
			"Long Stories",
			"Small RPGs",
			"Medium RPGs",
			"Long RPGs",
		]
	}

	pub fn to_pretty(&self) -> &'static str {
		match self {
			Self::Story(StorySize::Small) => "Small Stories",
			Self::Story(StorySize::Medium) => "Medium Stories",
			Self::Story(StorySize::Long) => "Long Stories",
			Self::RPG(StorySize::Small) => "Small RPGs",
			Self::RPG(StorySize::Medium) => "Medium RPGs",
			Self::RPG(StorySize::Long) => "Long RPGs",
		}
	}

	pub fn pretty_to_system(pretty: &str) -> Self {
		match pretty {
			"Small Stories" => Self::Story(StorySize::Small),
			"Medium Stories" => Self::Story(StorySize::Medium),
			"Long Stories" => Self::Story(StorySize::Long),
			"Small RPGs" => Self::RPG(StorySize::Small),
			"Medium RPGs" => Self::RPG(StorySize::Medium),
			"Long RPGs" => Self::RPG(StorySize::Long),
			_ => panic!("Bad pretty category"),
		}
	}
}

/// Implementation to convert [`SystemCategory`] to a string which is used to read the appropriate
/// system instructions from the `loreweaver-systems` directory.
impl From<String> for SystemCategory {
	fn from(system: String) -> Self {
		match system.as_str() {
			"story-small" => Self::Story(StorySize::Small),
			"story-medium" => Self::Story(StorySize::Medium),
			"story-long" => Self::Story(StorySize::Long),
			"rpg-small" => Self::RPG(StorySize::Small),
			"rpg-medium" => Self::RPG(StorySize::Medium),
			"rpg-long" => Self::RPG(StorySize::Long),
			_ => panic!("Bad system category"),
		}
	}
}

impl Into<String> for SystemCategory {
	fn into(self) -> String {
		match self {
			Self::Story(StorySize::Small) => "story-small".to_string(),
			Self::Story(StorySize::Medium) => "story-medium".to_string(),
			Self::Story(StorySize::Long) => "story-long".to_string(),
			Self::RPG(StorySize::Small) => "rpg-small".to_string(),
			Self::RPG(StorySize::Medium) => "rpg-medium".to_string(),
			Self::RPG(StorySize::Long) => "rpg-long".to_string(),
		}
	}
}

#[derive(Debug)]
pub enum StorySize {
	Small,
	Medium,
	Long,
}
