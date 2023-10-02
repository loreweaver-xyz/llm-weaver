use std::collections::BTreeMap;

use redis::{AsyncCommands, Client};
use serenity::async_trait;
use tokio::sync::OnceCell;
use tracing::{debug, info, instrument, trace, warn};

use crate::loreweaver::{
	AccountId, ContextMessage, StorageHandler, StoryPart, WeaveError, WeavingID,
};

/// Storage client to access GCP Storage
static REDIS_CLIENT: OnceCell<Client> = OnceCell::const_new();

pub struct Storage;

#[async_trait]
impl<Key: WeavingID> StorageHandler<Key> for Storage {
	#[instrument]
	async fn get_last_story_part(weaving_id: &Key) -> Result<Option<StoryPart>, WeaveError> {
		let client = get_client().await;
		let mut con = client.get_async_connection().await.expect("Failed to get redis connection");
		info!("Connected to Redis");

		let story_part: BTreeMap<String, String> = redis::cmd("HGETALL")
			.arg(Self::key(weaving_id))
			.query_async(&mut con)
			.await
			.expect("failed to execute HGETALL");

		trace!("Got story part: {:?}", story_part);
		let story_part = StoryPart {
			number: story_part
				.get("number")
				.ok_or(WeaveError::Custom(String::from("Story part number not found")))?
				.parse::<u16>()
				.map_err(|e| {
					WeaveError::Custom(format!("Failed to parse story part number: {}", e))
				})?,
			players: story_part
				.get("players")
				.ok_or(WeaveError::Custom(String::from("Story part players not found")))?
				.parse::<Vec<AccountId>>()
				.map_err(|e| {
					WeaveError::Custom(format!("Failed to parse story part players: {}", e))
				})?,
			context_tokens: story_part
				.get("context_tokens")
				.ok_or(WeaveError::Custom(String::from("Story part context_tokens not found")))?
				.parse::<u128>()
				.map_err(|e| {
					WeaveError::Custom(format!("Failed to parse story part context_tokens: {}", e))
				})?,
			context_messages: story_part
				.get("context_messages")
				.ok_or(WeaveError::Custom(String::from("Story part context_messages not found")))?
				.parse::<Vec<ContextMessage>>()
				.map_err(|e| {
					WeaveError::Custom(format!(
						"Failed to parse story part context_messages: {}",
						e
					))
				})?,
		}
		.into();

		Ok(story_part)
	}

	#[instrument]
	async fn save_story_part(
		weaving_id: &Key,
		story_part: StoryPart,
		increment: bool,
	) -> Result<(), WeaveError> {
		let client = get_client().await;
		let mut con = client.get_async_connection().await.expect("Failed to get redis connection");
		info!("Connected to Redis");

		let _: () = con.zincr(Self::key(weaving_id), "", 1).await.unwrap();

		let mut driver: BTreeMap<String, String> = BTreeMap::new();
		driver.insert(String::from("players"), String::from("redis-rs"));
		driver.insert(String::from("context_tokens"), String::from("0.19.0"));
		driver.insert(
			String::from("context_messages"),
			String::from("https://github.com/mitsuhiko/redis-rs"),
		);
		let _: () = redis::cmd("HSET")
			.arg(Self::key(weaving_id))
			.arg(driver)
			.query_async(&mut con)
			.await
			.expect("failed to execute HSET");

		debug!("Added story part to Redis");

		Ok(())
	}
}

/// Get and/or initialize the Redis Client.
#[instrument]
async fn get_client() -> Client {
	REDIS_CLIENT
		.get_or_init(async || {
			debug!("Initializing Redis client");
			// TODO: Secured uri scheme for Redis
			redis::Client::open("redis://:eYVX7EwVmmxKPCDmwMtyKVge8oLd2t81@127.0.0.1/")
				.map_err(|e| {
					WeaveError::Custom(format!("Redis configuration checks failed: {}", e))
				})
				.unwrap()
		})
		.await
		.clone()
}
