use redis::{AsyncCommands, Client};
use serenity::async_trait;
use tokio::sync::OnceCell;
use tracing::{debug, error, info, instrument};

use crate::loreweaver::{
	AccountId, ContextMessage, StorageHandler, StoryPart, WeaveError, WeavingID,
};

/// Storage client to access GCP Storage
static REDIS_CLIENT: OnceCell<Client> = OnceCell::const_new();

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

pub struct Storage;

#[async_trait]
impl<Key: WeavingID> StorageHandler<Key> for Storage {
	#[instrument]
	async fn get_last_story_part(weaving_id: &Key) -> Result<Option<StoryPart>, WeaveError> {
		let client = get_client().await;
		let mut con = client.get_async_connection().await.expect("Failed to get redis connection");
		info!("Connected to Redis");

		let base_key = &weaving_id.base_key();

		debug!("Executing ZRANGE_WITHSCORES {}...", base_key);
		let member_score: Vec<String> = con.zrange(&base_key, -1, -1).await.map_err(|e| {
			error!("Failed to save story part to Redis: {}", e);
			WeaveError::Custom(format!("Failed to save story part to Redis: {}", e))
		})?;
		debug!("Result ZRANGE_WITHSCORES: {:?}", member_score);

		if member_score.is_empty() {
			return Ok(None)
		}

		let index = member_score[0].parse::<u64>().unwrap_or(0);

		let key = format!("{base_key}:{index}");

		let story_part = StoryPart {
			players: {
				let raw_players: String = con.hget(&key, "players").await.map_err(|e| {
					error!("Failed to get \"players\" member from {} key: {}", key, e);
					return WeaveError::Custom(format!("Failed to get story part from Redis: {}", e))
				})?;
				serde_json::from_str::<Vec<AccountId>>(&raw_players).map_err(|e| {
					error!("Failed to parse story part players: {}", e);
					WeaveError::Custom(format!("Failed to parse story part players: {}", e))
				})?
			},
			context_tokens: {
				let context_tokens_str: String =
					con.hget(&key, "context_tokens").await.map_err(|e| {
						error!("Failed to get \"context_tokens\" member from {} key: {}", key, e);
						WeaveError::Custom(format!("Failed to get story part from Redis: {}", e))
					})?;
				context_tokens_str.parse::<u16>().map_err(|e| {
					error!("Failed to parse \"context_tokens\" member from {} key: {}", key, e);
					WeaveError::Custom(format!("Failed to parse context tokens: {}", e))
				})?
			},
			context_messages: {
				let context_messages_raw: Vec<u8> =
					con.hget(&key, "context_messages").await.map_err(|e| {
						error!("Failed to get \"context_messages\" member from {} key: {}", key, e);
						WeaveError::Custom(format!("Failed to get story part from Redis: {}", e))
					})?;

				serde_json::from_slice::<Vec<ContextMessage>>(&context_messages_raw).map_err(
					|e| {
						error!("Failed to parse story part context_messages: {}", e);
						WeaveError::Custom(format!(
							"Failed to parse story part context_messages: {}",
							e
						))
					},
				)?
			},
		};

		Ok(Some(story_part))
	}

	#[instrument]
	async fn save_story_part(
		weaving_id: &Key,
		story_part: StoryPart,
		_increment: bool,
	) -> Result<(), WeaveError> {
		let client = get_client().await;
		let mut con = client.get_async_connection().await.expect("Failed to get redis connection");
		info!("Connected to Redis");

		let base_key: &String = &weaving_id.base_key();

		let member_score: Vec<String> =
			con.zrange_withscores(&base_key, -1, -1).await.map_err(|e| {
				error!("Failed to save story part to Redis: {}", e);
				WeaveError::Custom(format!("Failed to save story part to Redis: {}", e))
			})?;

		info!("Got member score: {:?}", member_score);

		if member_score.is_empty() {
			// TODO
			// con.zincr(&base_key, 0, 1).await.map_err(|e| {
			// 	error!("Failed to save story part to Redis: {}", e);
			// 	WeaveError::Custom(format!("Failed to save story part to Redis: {}", e))
			// })?;

			con.zadd(&base_key, 0, 0).await.map_err(|e| {
				error!("Failed to save story part to Redis: {}", e);
				WeaveError::Custom(format!("Failed to save story part to Redis: {}", e))
			})?;
		}

		// TODO
		// let index = member_score[0].parse::<u64>().unwrap_or(0);
		let key = format!("{base_key}:0");

		con.hset(&key, "players", serde_json::to_string(&story_part.players).unwrap())
			.await
			.map_err(|e| {
				error!("Failed to save \"players\" member to {} key: {}", key, e);
				WeaveError::Custom(format!("Failed to save story part to Redis: {}", e))
			})?;

		debug!("Saved \"players\" member to {} key", key);

		con.hset(&key, "context_tokens", story_part.context_tokens).await.map_err(|e| {
			error!("Failed to save \"context_tokens\" member to {} key: {}", key, e);
			WeaveError::Custom(format!("Failed to save story part to Redis: {}", e))
		})?;

		debug!("Saved \"context_tokens\" member to {} key", key);

		con.hset(
			&key,
			"context_messages",
			serde_json::to_vec(&story_part.context_messages).map_err(|e| {
				error!("Failed to serialize story part context_messages: {}", e);
				WeaveError::Custom(format!(
					"Failed to serialize story part context_messages: {}",
					e
				))
			})?,
		)
		.await
		.map_err(|e| {
			error!("Failed to save \"context_messages\" member to {} key: {}", key, e);
			WeaveError::Custom(format!("Failed to save story part to Redis: {}", e))
		})?;

		debug!("Saved \"context_messages\" member to {} key", key);

		Ok(())
	}
}
