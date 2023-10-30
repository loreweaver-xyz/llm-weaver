use async_trait::async_trait;
use redis::{aio::Connection, AsyncCommands, Client, ToRedisArgs};
use serde::de::DeserializeOwned;
use std::fmt::{Debug, Display};
use tokio::sync::OnceCell;
use tracing::{debug, error, instrument};

use crate::{
	types::{LoomError, StorageError},
	Config, ContextMessage, TapestryFragment, TapestryId,
};

/// A storage handler trait designed for saving and retrieving fragments of a tapestry.
///
/// The `TapestryChestHandler` trait provides an asynchronous interface for implementing various
/// storage handlers to manage tapestry fragments.
///
/// # Usage
///
/// Implementations of `TapestryChestHandler` should provide the storage and retrieval mechanisms
/// tailored to specific use-cases or storage backends, such as databases, file systems, or
/// in-memory stores.
///
/// Checkout the out of the box implementation [`TapestryChest`] whichs uses Redis as the storage
/// backend.
#[async_trait]
pub trait TapestryChestHandler<T: Config> {
	/// Defines the error type returned by the handler methods.
	type Error: Display + Debug;

	/// Saves a tapestry fragment.
	///
	/// # Parameters
	///
	/// - `tapestry_id`: Identifies the tapestry.
	/// - `tapestry_fragment`: An instance of `TapestryFragment` to be stored.
	/// - `increment`:
	///     - A boolean flag indicating whether the tapestry instance should be incremented.
	///     - This should typically be `true` when saving a new instance of [`TapestryFragment`],
	///       and `false` when updating an existing one.
	///
	/// # Returns
	///
	/// Returns `Result<(), Self::Error>`. On successful storage, it returns `Ok(())`. If the
	/// storage operation fails, it should return an `Err` variant containing an error of type
	/// `Self::Error`.
	async fn save_tapestry_fragment<TID: TapestryId>(
		tapestry_id: TID,
		tapestry_fragment: TapestryFragment<T>,
		increment: bool,
	) -> crate::Result<()>;
	/// Save tapestry metadata.
	///
	/// Based on application use cases, you can add aditional data for a given [`TapestryId`]
	async fn save_tapestry_metadata<TID: TapestryId, M: ToRedisArgs + Send + Sync>(
		tapestry_id: TID,
		metadata: M,
	) -> crate::Result<()>;
	/// Retrieves the last tapestry fragment, or a fragment at a specified instance.
	///
	/// # Parameters
	///
	/// - `tapestry_id`: Identifies the tapestry.
	/// - `instance`: The instance of the fragment to retrieve. If `None`, the method should
	///   retrieve the last fragment.
	///
	/// # Returns
	///
	/// On successful retrieval, it returns `Ok(Some(TapestryFragment))` or `Ok(None)` if no
	/// fragment was found. If the retrieval operation fails, it should return an `Err` variant
	/// containing an error of type `Self::Error`.
	async fn get_tapestry_fragment<TID: TapestryId>(
		tapestry_id: TID,
		instance: Option<u64>,
	) -> crate::Result<Option<TapestryFragment<T>>>;
	/// Retrieves the last tapestry metadata, or a metadata at a specified instance.
	async fn get_tapestry_metadata<TID: TapestryId, M: DeserializeOwned>(
		tapestry_id: TID,
		instance: Option<u64>,
	) -> crate::Result<Option<M>>;
}

/// Default implementation of [`Config::TapestryChest`]
///
/// Storing and retrieving data using a Redis instance.
pub struct TapestryChest;

#[async_trait]
impl<T: Config> TapestryChestHandler<T> for TapestryChest {
	type Error = StorageError;

	async fn save_tapestry_fragment<TID: TapestryId>(
		tapestry_id: TID,
		tapestry_fragment: TapestryFragment<T>,
		increment: bool,
	) -> crate::Result<()> {
		let client = get_client().await.expect("Failed to get redis client");
		let mut con = client.get_async_connection().await.expect("Failed to get redis connection");
		debug!("Connected to Redis");

		let base_key: &String = &tapestry_id.base_key();

		let mut instance = match get_score_from_last_zset_member(&mut con, base_key).await? {
			Some(instance) => instance,
			None => {
				con.zadd(base_key, 0, 0).await.map_err(|e| {
					error!("Failed to save tapestry fragment to Redis: {}", e);
					LoomError::from(StorageError::Redis(e))
				})?;

				0
			},
		};

		if increment {
			con.zincr(base_key, 0, 1).await.map_err(|e| {
				error!("Failed to save tapestry fragment to Redis: {}", e);
				LoomError::from(StorageError::Redis(e))
			})?;

			instance += 1;

			debug!("Incremented instance to {} for {}", instance, base_key);
		}

		let key = format!("{base_key}:{instance}");

		con.hset(&key, "context_tokens", tapestry_fragment.context_tokens)
			.await
			.map_err(|e| {
				error!("Failed to save \"context_tokens\" member to {} key: {}", key, e);
				LoomError::from(StorageError::Redis(e))
			})?;

		debug!("Saved \"context_tokens\" member to {} key", key);

		con.hset(
			&key,
			"context_messages",
			serde_json::to_vec(&tapestry_fragment.context_messages).map_err(|e| {
				error!("Failed to serialize tapestry fragment context_messages: {}", e);
				StorageError::Parsing
			})?,
		)
		.await
		.map_err(|e| {
			error!("Failed to save \"context_messages\" member to {} key: {}", key, e);
			LoomError::from(StorageError::Redis(e))
		})?;

		debug!("Saved \"context_messages\" member to {} key", key);

		Ok(())
	}

	async fn save_tapestry_metadata<TID: TapestryId, M: ToRedisArgs + Send + Sync>(
		tapestry_id: TID,
		metadata: M,
	) -> crate::Result<()> {
		let client = get_client().await.expect("Failed to get redis client");
		let mut con = client.get_async_connection().await.expect("Failed to get redis connection");
		debug!("Connected to Redis");

		let base_key: &String = &tapestry_id.base_key();

		let instance = match get_score_from_last_zset_member(&mut con, base_key).await? {
			Some(instance) => instance,
			None => {
				con.zadd(base_key, 0, 0).await.map_err(|e| {
					error!("Failed to save tapestry fragment to Redis: {}", e);
					LoomError::from(StorageError::Redis(e))
				})?;

				0
			},
		};

		let key = format!("{base_key}:{instance}");

		con.hset(&key, "metadata", metadata).await.map_err(|e| {
			error!("Failed to save \"metadata\" member to {} key: {}", key, e);
			LoomError::from(StorageError::Redis(e))
		})?;

		debug!("Saved \"metadata\" member to {} key", key);

		Ok(())
	}

	async fn get_tapestry_fragment<TID: TapestryId>(
		tapestry_id: TID,
		instance: Option<u64>,
	) -> crate::Result<Option<TapestryFragment<T>>> {
		let client = get_client().await.expect("Failed to get redis client");
		let mut con = client.get_async_connection().await.expect("Failed to get redis connection");
		debug!("Connected to Redis");

		let base_key = &tapestry_id.base_key();

		let instance = match instance {
			Some(instance) => instance,
			None => match get_score_from_last_zset_member(&mut con, base_key).await? {
				Some(instance) => instance,
				None => return Ok(None),
			},
		};

		let key = format!("{base_key}:{instance}");

		let tapestry_fragment = TapestryFragment {
			context_tokens: {
				let context_tokens_str: String =
					con.hget(&key, "context_tokens").await.map_err(|e| {
						error!("Failed to get \"context_tokens\" member from {} key: {}", key, e);
						LoomError::from(StorageError::Redis(e))
					})?;
				context_tokens_str.parse::<T::Tokens>().map_err(|_| {
					error!("Failed to parse \"context_tokens\" member from key: {}", key);
					StorageError::Parsing
				})?
			},
			context_messages: {
				let context_messages_raw: Vec<u8> =
					con.hget(&key, "context_messages").await.map_err(|e| {
						error!("Failed to get \"context_messages\" member from {} key: {}", key, e);
						LoomError::from(StorageError::Redis(e))
					})?;

				serde_json::from_slice::<Vec<ContextMessage<T>>>(&context_messages_raw).map_err(
					|e| {
						error!("Failed to parse tapestry fragment context_messages: {}", e);
						StorageError::Parsing
					},
				)?
			},
		};

		Ok(Some(tapestry_fragment))
	}

	async fn get_tapestry_metadata<TID: TapestryId, M: DeserializeOwned>(
		tapestry_id: TID,
		instance: Option<u64>,
	) -> crate::Result<Option<M>> {
		let client = get_client().await.expect("Failed to get redis client");
		let mut con = client.get_async_connection().await.expect("Failed to get redis connection");
		debug!("Connected to Redis");

		let base_key = &tapestry_id.base_key();

		let instance = match instance {
			Some(instance) => instance,
			None => match get_score_from_last_zset_member(&mut con, base_key).await? {
				Some(instance) => instance,
				None => return Ok(None),
			},
		};

		let key = format!("{base_key}:{instance}");

		let metadata_raw: Vec<u8> = con.hget(&key, "metadata").await.map_err(|e| {
			error!("Failed to get \"metadata\" member from {} key: {}", key, e);
			LoomError::from(StorageError::Redis(e))
		})?;

		let tapestry_metadata = serde_json::from_slice::<M>(&metadata_raw).map_err(|e| {
			error!("Failed to parse tapestry fragment metadata: {}", e);
			StorageError::Parsing
		})?;

		Ok(Some(tapestry_metadata))
	}
}

/// Storage client to access GCP Storage
static REDIS_CLIENT: OnceCell<Client> = OnceCell::const_new();

/// Get and/or initialize the Redis Client.
#[instrument]
async fn get_client() -> Result<Client, redis::RedisError> {
	Ok(REDIS_CLIENT
		.get_or_init(async || {
			debug!("Initializing Redis client");

			let protocol = std::env::var("REDIS_PROTOCOL").unwrap_or_else(|_| "redis".to_string());
			let host = std::env::var("REDIS_HOST").unwrap_or_else(|_| "redis".to_string());
			let port = std::env::var("REDIS_PORT").unwrap_or_else(|_| "6379".to_string());
			let password = std::env::var("REDIS_PASSWORD").unwrap_or_default();

			match redis::Client::open(format!("{}://:{}@{}:{}", protocol, password, host, port)) {
				Ok(client) => client,
				Err(e) => {
					let m = format!("Failed to initialize Redis client: {}", e);
					error!(m);
					panic!("{}", m)
				},
			}
		})
		.await
		.clone())
}

async fn get_score_from_last_zset_member(
	con: &mut Connection,
	base_key: &String,
) -> super::Result<Option<u64>> {
	debug!("Executing ZRANGE_WITHSCORES {}...", base_key);
	let member_score: Vec<String> = con.zrange_withscores(base_key, -1, -1).await.map_err(|e| {
		error!("Failed to save tapestry fragment to Redis: {}", e);
		LoomError::from(StorageError::Redis(e))
	})?;
	debug!("Result ZRANGE_WITHSCORES: {:?}", member_score);

	let instance = match member_score.is_empty() {
		true => return Ok(None),
		false if member_score.len() == 2 => member_score[1].parse::<u64>().unwrap_or(0),
		false => {
			error!("Unexpected member score length: {}", member_score.len());
			return Err(LoomError::from(StorageError::Parsing).into());
		},
	};

	Ok(Some(instance))
}
