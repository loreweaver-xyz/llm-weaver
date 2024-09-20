use async_trait::async_trait;
use redis::{AsyncCommands, Client, Commands, Connection, RedisError, ToRedisArgs};
use serde::de::DeserializeOwned;
use std::fmt::Debug;
use tokio::sync::OnceCell;
use tracing::{debug, error, instrument};

use crate::{
	types::{LoomError, PromptModelTokens, StorageError},
	Config, ContextMessage, TapestryFragment, TapestryId,
};

use super::TapestryChestHandler;

/// The key used to store the number of instances of a tapestry.
const INSTANCE_COUNT: &str = "instance_count";

/// Default implementation of [`Config::Chest`]
///
/// Storing and retrieving data using a Redis instance.
pub struct RedisChest;

#[async_trait]
impl<T: Config> TapestryChestHandler<T> for RedisChest {
	type Error = StorageError;

	async fn save_tapestry_fragment<TID: TapestryId>(
		tapestry_id: &TID,
		tapestry_fragment: TapestryFragment<T>,
		increment: bool,
	) -> crate::Result<u64> {
		let client = get_client().await.expect("Failed to get redis client");
		let mut con = client.get_connection()?;
		let base_key = &tapestry_id.base_key();

		let mut tapestry_instance =
			verify_and_get_instance(&mut con, base_key, None).await?.unwrap_or(0);

		redis::transaction(&mut con, &[base_key], |con, pipe| {
			// If the tapestry does not exist (i.e. instance is at 0), then set it to 1
			if tapestry_instance == 0 {
				pipe.hset(base_key, INSTANCE_COUNT, 1).ignore();
				debug!("Saved \"instance_count\" member to {} key", base_key);

				tapestry_instance = 1
			};

			if increment {
				pipe.hincr(base_key, INSTANCE_COUNT, 1).ignore();

				tapestry_instance += 1;

				debug!("Incremented instance to {} for {}", tapestry_instance, base_key);
			}

			let instance_key = format!("{base_key}:{tapestry_instance}");

			pipe.hset(
				&instance_key,
				"context_tokens",
				tapestry_fragment.context_tokens.to_string(),
			)
			.ignore();
			debug!("Saved \"context_tokens\" member to {} key", instance_key);

			pipe.hset(
				&instance_key,
				"context_messages",
				serde_json::to_vec(&tapestry_fragment.context_messages)
					.map_err(RedisError::from)?,
			)
			.ignore();
			debug!("Saved \"context_messages\" member to {} key", instance_key);

			pipe.query(con)
		})
		.map_err(|e| {
			error!("Failed to save tapestry fragment: {}", e);
			LoomError::from(StorageError::Redis(e))
		})?;

		Ok(tapestry_instance)
	}

	async fn save_tapestry_metadata<TID: TapestryId, M: ToString + Debug + Clone + Send + Sync>(
		tapestry_id: TID,
		metadata: M,
	) -> crate::Result<()> {
		let client = get_client().await.expect("Failed to get redis client");
		let mut con = client.get_multiplexed_async_connection().await?;
		debug!("Connected to Redis");

		let key: &String = &tapestry_id.base_key();

		con.hset(key, "metadata", metadata.to_string().clone()).await.map_err(|e| {
			error!("Failed to save \"metadata\" member to {} key: {}", key, e);
			LoomError::from(StorageError::Redis(e))
		})?;

		debug!("Saved \"metadata\" member to {} key with metadata {:?}", key, metadata.clone());

		Ok(())
	}

	async fn get_tapestry<TID: TapestryId>(tapestry_id: TID) -> crate::Result<Option<u16>> {
		let client = get_client().await.expect("Failed to get redis client");
		let mut con = client.get_multiplexed_async_connection().await?;

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
		let mut con = client.get_connection()?;
		debug!("Connected to Redis");

		let base_key = &tapestry_id.base_key();

		let instance = match verify_and_get_instance(&mut con, base_key, instance).await? {
			Some(instance) => instance,
			None => return Ok(None),
		};

		let key = format!("{base_key}:{instance}");

		let tapestry_fragment = TapestryFragment {
			context_tokens: {
				let context_tokens_str: String = con.hget(&key, "context_tokens").map_err(|e| {
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
					con.hget(&key, "context_messages").map_err(|e| {
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
	) -> crate::Result<Option<M>> {
		let client = get_client().await.expect("Failed to get redis client");
		let mut con = client.get_multiplexed_async_connection().await?;
		debug!("Connected to Redis");

		let key = &tapestry_id.base_key();

		let metadata_raw: Vec<u8> = con.hget(key, "metadata").await.map_err(|e| {
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
		let mut con = client.get_multiplexed_async_connection().await?;

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

	async fn delete_tapestry_fragment<TID: TapestryId>(
		tapestry_id: TID,
		instance: Option<u64>,
	) -> crate::Result<()> {
		let client = get_client().await.expect("Failed to get redis client");
		let mut con = client.get_connection()?;
		let base_key = &tapestry_id.base_key();

		let instance = match verify_and_get_instance(&mut con, base_key, instance).await? {
			Some(instance) => instance,
			None => return Ok(()),
		};

		let key = format!("{base_key}:{instance}");

		debug!("Deleting {} instance", key);

		con.del(&key).map_err(|e| {
			error!("Failed to delete {} tapestry_id: {}", key, e);
			LoomError::from(StorageError::Redis(e))
		})?;

		debug!("Deleted {} instance", key);

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
async fn verify_and_get_instance(
	con: &mut Connection,
	base_key: &String,
	instance: Option<u64>,
) -> crate::Result<Option<u64>> {
	Ok(match con.exists(base_key)? {
		true => match instance {
			Some(instance) =>
				if con.exists(&format!("{}:{}", base_key, instance))? {
					Some(instance)
				} else {
					return Err(LoomError::from(StorageError::NotFound).into());
				},
			None => con.hget(base_key, INSTANCE_COUNT).map_err(|e| {
				error!("Failed to get {} tapestry_id: {}", base_key, e);
				LoomError::from(StorageError::Redis(e))
			})?,
		},
		false => None,
	})
}
