use crate::{
	loreweaver::{Config, Loom},
	storage::Storage,
	GPTModel, Server,
};

pub struct ApiService;

impl Config for ApiService {
	type Model = GPTModel;
	type Storage = Storage;
}

impl<T: Loom> Server<T> for ApiService {
	async fn serve() {
		let prompt = T::prompt(
			(100, 101),
			"Hello Loreweaver!".to_string(),
			12345,
			"Michael".to_string(),
			Some("Snowmead".to_string()),
		)
		.await;
		println!("Loreweaver: {}", prompt.unwrap());
	}
}
