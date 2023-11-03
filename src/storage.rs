use async_trait::async_trait;
use redis::{aio::Connection, AsyncCommands, Client, ToRedisArgs};
use serde::de::DeserializeOwned;
use std::fmt::{Debug, Display};
use tokio::sync::OnceCell;
use tracing::{debug, error, instrument};

use crate::{
	types::{LoomError, PromptModelTokens, StorageError},
	Config, ContextMessage, TapestryFragment, TapestryId,
};

/// The key used to store the number of instances of a tapestry.
const INSTANCE_COUNT: &str = "instance_count";

/// A storage handler trait designed for saving and retrieving fragments of a tapestry.
///
/// # Usage
///
/// Implementations of `TapestryChestHandler` should provide the storage and retrieval mechanisms
/// tailored to specific use-cases or storage backends, such as databases, file systems, or
/// in-memory stores.
///
/// You can see how the default [`TapestryChest`] struct implementes this trait which uses Redis as
/// its storage backend.
#[async_trait]
pub trait TapestryChestHandler<T: Config> {
	/// Defines the error type returned by the handler methods.
	type Error: Display + Debug;

	/// Saves a tapestry fragment.
	///
	/// Tapestry fragments are stored incrementally, also refered to as "instances". Each instance
	/// is identified by an integer, starting at 1 and incrementing by 1 for each new instance.
	///
	/// This method is executed primarily by the [`crate::Loom::weave`] function.
	///
	/// # Parameters
	///
	/// - `tapestry_id`: Identifies the tapestry.
	/// - `tapestry_fragment`: An instance of `TapestryFragment` to be stored.
	/// - `increment`:
	///     - A boolean flag indicating whether the tapestry instance should be incremented.
	///     - This should typically be `true` when saving a new instance of [`TapestryFragment`],
	///       and `false` when updating an existing one.
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
	/// Retrieves the number of instances of a tapestry.
	///
	/// Returns None if the tapestry does not exist.
	async fn get_tapestry<TID: TapestryId>(tapestry_id: TID) -> crate::Result<Option<u16>>;
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
	/// fragment was found.
	async fn get_tapestry_fragment<TID: TapestryId>(
		tapestry_id: TID,
		instance: Option<u64>,
	) -> crate::Result<Option<TapestryFragment<T>>>;
	/// Retrieves the last tapestry metadata, or a metadata at a specified instance.
	async fn get_tapestry_metadata<TID: TapestryId, M: DeserializeOwned + Default>(
		tapestry_id: TID,
		instance: Option<u64>,
	) -> crate::Result<Option<M>>;
	/// Deletes a tapestry and all its instances.
	async fn delete_tapestry<TID: TapestryId>(tapestry_id: TID) -> crate::Result<()>;
}

/// Default implementation of [`Config::Chest`]
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
		let mut con = client.get_async_connection().await?;
		debug!("Connected to Redis");

		let base_key: &String = &tapestry_id.base_key();

		let mut instance = match get_last_instance(&mut con, base_key, None).await? {
			Some(instance) => instance,
			None => {
				con.hset(base_key, INSTANCE_COUNT, 1).await.map_err(|e| {
					error!("Failed to save \"instance_count\" member to {} key: {}", base_key, e);
					LoomError::from(StorageError::Redis(e))
				})?;

				debug!("Saved \"instance_count\" member to {} key", base_key);

				1
			},
		};

		if increment {
			con.hincr(base_key, INSTANCE_COUNT, 1).await.map_err(|e| {
				error!("Failed to increment \"instance_count\" member of {} key: {}", base_key, e);
				LoomError::from(StorageError::Redis(e))
			})?;

			instance += 1;

			debug!("Incremented instance to {} for {}", instance, base_key);
		}

		let instance_key = format!("{base_key}:{instance}");

		con.hset(&instance_key, "context_tokens", tapestry_fragment.context_tokens)
			.await
			.map_err(|e| {
				error!("Failed to save \"context_tokens\" member to {} key: {}", instance_key, e);
				LoomError::from(StorageError::Redis(e))
			})?;

		debug!("Saved \"context_tokens\" member to {} key", instance_key);

		con.hset(
			&instance_key,
			"context_messages",
			serde_json::to_vec(&tapestry_fragment.context_messages).map_err(|e| {
				error!("Failed to serialize tapestry fragment context_messages: {}", e);
				StorageError::Parsing
			})?,
		)
		.await
		.map_err(|e| {
			error!("Failed to save \"context_messages\" member to {} key: {}", instance_key, e);
			LoomError::from(StorageError::Redis(e))
		})?;

		debug!("Saved \"context_messages\" member to {} key", instance_key);

		Ok(())
	}

	async fn save_tapestry_metadata<TID: TapestryId, M: ToRedisArgs + Send + Sync>(
		tapestry_id: TID,
		metadata: M,
	) -> crate::Result<()> {
		let client = get_client().await.expect("Failed to get redis client");
		let mut con = client.get_async_connection().await?;
		debug!("Connected to Redis");

		let base_key: &String = &tapestry_id.base_key();

		let instance = match get_last_instance(&mut con, base_key, None).await? {
			Some(instance) => instance,
			None => {
				return Err(LoomError::from(StorageError::NotFound).into());
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

	async fn get_tapestry<TID: TapestryId>(tapestry_id: TID) -> crate::Result<Option<u16>> {
		let client = get_client().await.expect("Failed to get redis client");
		let mut con = client.get_async_connection().await?;

		let base_key = &tapestry_id.base_key();

		let exists: bool = con.exists(base_key).await.map_err(|e| {
			error!("Failed to check if {} tapestry_id exists: {}", base_key, e);
			LoomError::from(StorageError::Redis(e))
		})?;

		if !exists {
			return Ok(None);
		}

		let tapestry: u16 = con.hget(base_key, INSTANCE_COUNT).await.map_err(|e| {
			error!("Failed to get {} tapestry_id: {}", base_key, e);
			LoomError::from(StorageError::Redis(e))
		})?;

		Ok(Some(tapestry))
	}

	async fn get_tapestry_fragment<TID: TapestryId>(
		tapestry_id: TID,
		instance: Option<u64>,
	) -> crate::Result<Option<TapestryFragment<T>>> {
		let client = get_client().await.expect("Failed to get redis client");
		let mut con = client.get_async_connection().await?;
		debug!("Connected to Redis");

		let base_key = &tapestry_id.base_key();

		let instance = match get_last_instance(&mut con, base_key, instance).await? {
			Some(instance) => instance,
			None => return Ok(None),
		};

		let key = format!("{base_key}:{instance}");

		let tapestry_fragment = TapestryFragment {
			context_tokens: {
				let context_tokens_str: String =
					con.hget(&key, "context_tokens").await.map_err(|e| {
						error!("Failed to get \"context_tokens\" member from {} key: {}", key, e);
						LoomError::from(StorageError::Redis(e))
					})?;
				context_tokens_str.parse::<PromptModelTokens<T>>().map_err(|_| {
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

	async fn get_tapestry_metadata<TID: TapestryId, M: DeserializeOwned + Default>(
		tapestry_id: TID,
		instance: Option<u64>,
	) -> crate::Result<Option<M>> {
		let client = get_client().await.expect("Failed to get redis client");
		let mut con = client.get_async_connection().await?;
		debug!("Connected to Redis");

		let base_key = &tapestry_id.base_key();

		let instance = match get_last_instance(&mut con, base_key, instance).await? {
			Some(instance) => instance,
			None => return Ok(None),
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

	async fn delete_tapestry<TID: TapestryId>(tapestry_id: TID) -> crate::Result<()> {
		let client = get_client().await.expect("Failed to get redis client");
		let mut con = client.get_async_connection().await?;

		let tapestry_id = &tapestry_id.base_key();

		let exists: bool = con.exists(tapestry_id).await.map_err(|e| {
			error!("Failed to check if {} tapestry_id exists: {}", tapestry_id, e);
			LoomError::from(StorageError::Redis(e))
		})?;

		if !exists {
			debug!("{} tapestry_id does not exist", tapestry_id);
			return Ok(());
		}

		let instance_count: u16 = con.hget(tapestry_id, INSTANCE_COUNT).await.map_err(|e| {
			error!("Failed to get {} tapestry_id: {}", tapestry_id, e);
			LoomError::from(StorageError::Redis(e))
		})?;

		for i in 1..=instance_count {
			let instance_key = format!("{}:{}", tapestry_id, i);

			debug!("Deleting {} instance", instance_key);

			con.del(&instance_key).await.map_err(|e| {
				error!("Failed to delete {} tapestry_id: {}", tapestry_id, e);
				LoomError::from(StorageError::Redis(e))
			})?;
		}

		con.del(tapestry_id).await.map_err(|e| {
			error!("Failed to delete {} tapestry_id: {}", tapestry_id, e);
			LoomError::from(StorageError::Redis(e))
		})?;

		debug!("Deleted {} tapestry_id and {} instances", tapestry_id, instance_count);

		Ok(())
	}
}

/// Storage client to access GCP Storage
static REDIS_CLIENT: OnceCell<Client> = OnceCell::const_new();

/// Get the Redis Client.
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

/// Get the last instance number of a tapestry.
///
/// If the tapestry does not exist, it will be created and the instance number will be set to 1.
///
/// Passing the `instance` parameter will return the instance number if it exists, or an error if it
/// does not.
async fn get_last_instance(
	con: &mut Connection,
	base_key: &String,
	instance: Option<u64>,
) -> crate::Result<Option<u64>> {
	Ok(match con.exists(base_key).await? {
		true => match instance {
			Some(instance) =>
				if con.exists(&format!("{}:{}", base_key, instance)).await? {
					Some(instance)
				} else {
					return Err(LoomError::from(StorageError::NotFound).into());
				},
			None => con.hget(base_key, INSTANCE_COUNT).await.map_err(|e| {
				error!("Failed to get {} tapestry_id: {}", base_key, e);
				LoomError::from(StorageError::Redis(e))
			})?,
		},
		false => None,
	})
}
