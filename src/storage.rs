use std::env;

use google_cloud_storage::{
	client::{Client, ClientConfig},
	http::objects::{download::Range, get::GetObjectRequest, list::ListObjectsRequest},
};
use serenity::async_trait;
use tokio::sync::OnceCell;

use crate::loreweaver::{StorageHandler, StoryPart, WeaveError, WeavingID};

/// Storage client to access GCP Storage
static STORAGE_CLIENT: OnceCell<Client> = OnceCell::const_new();
/// Storage bucket in GCP Storage to store all message histories
static STORAGE_BUCKET: OnceCell<String> = OnceCell::const_new();

pub struct Storage;

#[async_trait]
impl StorageHandler for Storage {
	/// Download _last_ story part from GCP Storage.
	///
	/// None is returned if there are no story parts for the given `weaving_id`. This means that
	/// the story has not been started yet.
	async fn get(id: WeavingID) -> Result<Option<StoryPart>, WeaveError> {
		let client: &Client = STORAGE_CLIENT
			.get_or_init(async || {
				let config = ClientConfig::default().with_auth().await.unwrap();
				Client::new(config)
			})
			.await;

		let bucket = STORAGE_BUCKET
			.get_or_init(async || env::var("STORAGE_BUCKET").unwrap())
			.await
			.to_owned();

		let parts = client
			.list_objects(&ListObjectsRequest { bucket: bucket.clone(), ..Default::default() })
			.await
			.unwrap()
			.items
			.unwrap_or_default();

		let maybe_last_part = parts.last();

		match maybe_last_part {
			None => Ok(None),
			Some(last_part) => {
				let bytes = client
					.download_object(
						&GetObjectRequest {
							bucket,
							object: format!("{}/{}/{}.json", id.0, id.1, last_part.name),
							..Default::default()
						},
						&Range::default(),
					)
					.await;

				match bytes {
					Err(_) => Ok(None),
					Ok(bytes) => Ok(Some(serde_json::from_slice(&bytes).unwrap())),
				}
			},
		}
	}
}
