use std::fmt::Display;

use crate::{
	loreweaver::{self, Config, Loom, Loreweaver},
	storage::Storage,
	GPTModel, Server,
};

/// Encompasses many [`StoryID`]s.
type ServerID = u128;
/// The `StoryID` identifies a single story which is part of a [`ServerID`].
type StoryID = u128;
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
		format!("{}-{}", self.server_id, self.story_id)
	}
}

impl Display for WeavingID {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		write!(f, "{}-{}", self.server_id, self.story_id)
	}
}

#[derive(Default, Debug)]
pub struct ApiService;

impl Config for ApiService {
	type Model = GPTModel;
	type Storage = Storage;
	type WeavingID = WeavingID;
}

impl Server for ApiService {
	async fn serve() {
		let prompt = Loreweaver::<Self>::prompt(
			WeavingID { server_id: 12345, story_id: 12345 },
			"Hello Loreweaver!".to_string(),
			12345,
			"Michael".to_string(),
			Some("Snowmead".to_string()),
		)
		.await;

		println!("{}", prompt.unwrap());
	}
}
