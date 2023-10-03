use std::fmt::Display;

use redis::{aio::Connection, AsyncCommands, Client};
use serenity::async_trait;
use tokio::sync::OnceCell;
use tracing::{debug, error, instrument};

use crate::loreweaver::{AccountId, ContextMessage, StorageHandler, StoryPart, WeavingID};

/// Storage client to access GCP Storage
static REDIS_CLIENT: OnceCell<Client> = OnceCell::const_new();

#[derive(Debug)]
pub enum StorageError {
	Redis(redis::RedisError),
	Parsing,
}

impl Display for StorageError {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		write!(f, "{}", self)
	}
}

/// Get and/or initialize the Redis Client.
#[instrument]
async fn get_client() -> Result<Client, redis::RedisError> {
	Ok(REDIS_CLIENT
		.get_or_init(async || {
			debug!("Initializing Redis client");
			// TODO: Secured uri scheme for Redis
			match redis::Client::open("redis://:eYVX7EwVmmxKPCDmwMtyKVge8oLd2t81@127.0.0.1/") {
				Ok(client) => client,
				Err(e) => {
					error!("Failed to initialize Redis client: {}", e);
					panic!("Failed to initialize Redis client: {}", e)
				},
			}
		})
		.await
		.clone())
}

#[derive(Debug)]
pub struct Storage;

#[async_trait]
impl<Key: WeavingID> StorageHandler<Key> for Storage {
	type Error = StorageError;

	#[instrument]
	async fn get_last_story_part(weaving_id: &Key) -> Result<Option<StoryPart>, Self::Error> {
		let client = get_client().await.expect("Failed to get redis client");
		let mut con = client.get_async_connection().await.expect("Failed to get redis connection");
		debug!("Connected to Redis");

		let base_key = &weaving_id.base_key();

		let index = match get_score_from_last_zset_member(&mut con, base_key).await? {
			Some(index) => index,
			None => return Ok(None),
		};

		let key = format!("{base_key}:{index}");

		let story_part = StoryPart {
			players: {
				let raw_players: String = con.hget(&key, "players").await.map_err(|e| {
					error!("Failed to get \"players\" member from {} key: {}", key, e);
					StorageError::Redis(e)
				})?;
				serde_json::from_str::<Vec<AccountId>>(&raw_players).map_err(|e| {
					error!("Failed to parse story part players: {}", e);
					StorageError::Parsing
				})?
			},
			context_tokens: {
				let context_tokens_str: String =
					con.hget(&key, "context_tokens").await.map_err(|e| {
						error!("Failed to get \"context_tokens\" member from {} key: {}", key, e);
						StorageError::Redis(e)
					})?;
				context_tokens_str.parse::<u16>().map_err(|e| {
					error!("Failed to parse \"context_tokens\" member from {} key: {}", key, e);
					StorageError::Parsing
				})?
			},
			context_messages: {
				let context_messages_raw: Vec<u8> =
					con.hget(&key, "context_messages").await.map_err(|e| {
						error!("Failed to get \"context_messages\" member from {} key: {}", key, e);
						StorageError::Redis(e)
					})?;

				serde_json::from_slice::<Vec<ContextMessage>>(&context_messages_raw).map_err(
					|e| {
						error!("Failed to parse story part context_messages: {}", e);
						StorageError::Parsing
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
	) -> Result<(), Self::Error> {
		let client = get_client().await.expect("Failed to get redis client");
		let mut con = client.get_async_connection().await.expect("Failed to get redis connection");
		debug!("Connected to Redis");

		let base_key: &String = &weaving_id.base_key();

		let index = match get_score_from_last_zset_member(&mut con, base_key).await? {
			Some(index) => index,
			None => {
				// TODO
				// con.zincr(&base_key, 0, 1).await.map_err(|e| {
				// 	error!("Failed to save story part to Redis: {}", e);
				// 	WeaveError::Storage
				// })?;

				con.zadd(&base_key, 0, 0).await.map_err(|e| {
					error!("Failed to save story part to Redis: {}", e);
					StorageError::Redis(e)
				})?;

				0
			},
		};

		let key = format!("{base_key}:{index}");

		con.hset(&key, "players", serde_json::to_string(&story_part.players).unwrap())
			.await
			.map_err(|e| {
				error!("Failed to save \"players\" member to {} key: {}", key, e);
				StorageError::Redis(e)
			})?;

		debug!("Saved \"players\" member to {} key", key);

		con.hset(&key, "context_tokens", story_part.context_tokens).await.map_err(|e| {
			error!("Failed to save \"context_tokens\" member to {} key: {}", key, e);
			StorageError::Redis(e)
		})?;

		debug!("Saved \"context_tokens\" member to {} key", key);

		con.hset(
			&key,
			"context_messages",
			serde_json::to_vec(&story_part.context_messages).map_err(|e| {
				error!("Failed to serialize story part context_messages: {}", e);
				StorageError::Parsing
			})?,
		)
		.await
		.map_err(|e| {
			error!("Failed to save \"context_messages\" member to {} key: {}", key, e);
			StorageError::Redis(e)
		})?;

		debug!("Saved \"context_messages\" member to {} key", key);

		Ok(())
	}
}

async fn get_score_from_last_zset_member(
	con: &mut Connection,
	base_key: &String,
) -> Result<Option<u64>, StorageError> {
	debug!("Executing ZRANGE_WITHSCORES {}...", base_key);
	let member_score: Vec<String> =
		con.zrange_withscores(&base_key, -1, -1).await.map_err(|e| {
			error!("Failed to save story part to Redis: {}", e);
			StorageError::Redis(e)
		})?;
	debug!("Result ZRANGE_WITHSCORES: {:?}", member_score);

	let index = match member_score.is_empty() {
		true => return Ok(None),
		false if member_score.len() == 2 => member_score[1].parse::<u64>().unwrap_or(0),
		false => {
			error!("Unexpected member score length: {}", member_score.len());
			return Err(StorageError::Parsing)
		},
	};

	Ok(Some(index))
}
