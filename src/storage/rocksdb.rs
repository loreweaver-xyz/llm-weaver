use async_trait::async_trait;
use rocksdb::{
	DBWithThreadMode, MultiThreaded, OptimisticTransactionDB, OptimisticTransactionOptions,
	TransactionDB, WriteBatch, WriteOptions,
};

use serde::de::DeserializeOwned;
use std::sync::Arc;
use tracing::{debug, error};

use crate::{
	types::{LoomError, StorageError},
	Config, TapestryFragment, TapestryId,
};

use super::{
	common::{StorageBackend, INSTANCE_COUNT_KEY},
	TapestryChestHandler,
};

/// RocksDB implementation of the storage backend.
pub struct RocksDbBackend {
	db: Arc<OptimisticTransactionDB>,
}

impl RocksDbBackend {
	/// Creates a new `RocksDbBackend` with the provided database instance.
	pub fn new(db: Arc<OptimisticTransactionDB>) -> Self {
		Self { db }
	}

	/// Creates a new `RocksDbBackend` with default configuration.
	pub fn new_default() -> Result<Self, StorageError> {
		let path = std::env::var("ROCKSDB_PATH").unwrap_or_else(|_| "rocksdb_data".to_string());
		let db = OptimisticTransactionDB::open_default(path).map_err(StorageError::RocksDb)?;
		Ok(Self { db: Arc::new(db) })
	}
}

impl StorageBackend for RocksDbBackend {
	type Error = StorageError;

	fn get_instance_count(&self, base_key: &str) -> Result<Option<u64>, Self::Error> {
		let key = format!("{}:{}", base_key, INSTANCE_COUNT_KEY);
		match self.db.get(key.as_bytes()) {
			Ok(Some(value)) => {
				let count_str = String::from_utf8(value).map_err(|_| StorageError::Parsing)?;
				let count = count_str.parse::<u64>().map_err(|_| StorageError::Parsing)?;
				Ok(Some(count))
			},
			Ok(None) => Ok(None),
			Err(e) => Err(StorageError::RocksDb(e)),
		}
	}

	fn set_instance_count(&self, base_key: &str, count: u64) -> Result<(), Self::Error> {
		let key = format!("{}:{}", base_key, INSTANCE_COUNT_KEY);
		self.db.put(key.as_bytes(), count.to_string()).map_err(StorageError::RocksDb)
	}

	fn increment_instance_count(&self, base_key: &str) -> Result<u64, Self::Error> {
		let db = self.db.clone();

		let mut retries = 0;
		let max_retries = 5;

		loop {
			let txn = db.transaction();

			let key = format!("{}:{}", base_key, INSTANCE_COUNT_KEY);

			// Read the current count within the transaction
			let current_value = txn.get(key.as_bytes())?;
			let current_count = current_value
				.map(|v| {
					String::from_utf8(v)
						.map_err(|_| StorageError::Parsing)?
						.parse::<u64>()
						.map_err(|_| StorageError::Parsing)?
				})
				.unwrap_or(0);
			let new_count = current_count + 1;

			// Write the updated count
			txn.put(key.as_bytes(), new_count.to_string());

			// Try to commit
			match txn.commit() {
				Ok(_) => return Ok(new_count),
				Err(e) => {
					if retries < max_retries {
						retries += 1;
						continue; // Retry
					} else {
						return Err(StorageError::RocksDb(e));
					}
				},
			}
		}
	}

	fn key_exists(&self, key: &str) -> Result<bool, Self::Error> {
		match self.db.get(key.as_bytes()) {
			Ok(Some(_)) => Ok(true),
			Ok(None) => Ok(false),
			Err(e) => Err(StorageError::RocksDb(e)),
		}
	}
}

#[async_trait]
impl<T: Config + serde::Serialize + DeserializeOwned + Send + Sync> TapestryChestHandler<T>
	for RocksDbBackend
{
	type Error = StorageError;

	fn new() -> Self {
		Self::new_default().expect("Failed to create RocksDB backend")
	}

	async fn save_tapestry_fragment<TID: TapestryId + Send + Sync>(
		&self,
		tapestry_id: &TID,
		tapestry_fragment: TapestryFragment<T>,
		increment: bool,
	) -> crate::Result<u64> {
		let db = self.db.clone();
		let base_key = tapestry_id.base_key();
		let fragment_bytes = serde_json::to_vec(&tapestry_fragment).map_err(|e| {
			error!("Failed to serialize tapestry fragment: {}", e);
			StorageError::Parsing
		})?;

		let mut retries = 0;
		let max_retries = 3;

		loop {
			// Begin a transaction
			let txn = db.transaction();

			// Read the current instance count within the transaction
			let instance_count_key = format!("{}:{}", base_key, INSTANCE_COUNT_KEY);
			let instance_count_value = txn.get(instance_count_key.as_bytes())?;
			let mut instance_count = if let Some(value) = instance_count_value {
				let count_str = String::from_utf8(value).map_err(|_| StorageError::Parsing)?;
				count_str.parse::<u64>().map_err(|_| StorageError::Parsing)?
			} else {
				0
			};

			if increment {
				instance_count += 1;
			} else if instance_count == 0 {
				instance_count = 1;
			}

			// Write the updated instance count
			txn.put(instance_count_key.as_bytes(), instance_count.to_string());

			// Write the tapestry fragment
			let instance_key = format!("{}:{}", base_key, instance_count);
			txn.put(instance_key.as_bytes(), fragment_bytes.clone());

			// Try to commit the transaction
			match txn.commit() {
				Ok(_) => {
					// Success
					return Ok(instance_count);
				},
				Err(e) => {
					// Handle conflict
					if retries < max_retries {
						retries += 1;
						continue; // Retry the transaction
					} else {
						return Err(StorageError::RocksDb(e).into());
					}
				},
			}
		}
	}

	async fn save_tapestry_metadata<
		TID: TapestryId + Send + Sync,
		M: ToString + std::fmt::Debug + Clone + Send + Sync,
	>(
		&self,
		tapestry_id: TID,
		metadata: M,
	) -> crate::Result<()> {
		let db = self.db.clone();
		let key = format!("{}:metadata", tapestry_id.base_key());
		db.put(key.as_bytes(), metadata.to_string()).map_err(StorageError::RocksDb)?;
		debug!("Saved \"metadata\" key {} with metadata {:?}", key, metadata);

		Ok(())
	}

	async fn get_tapestry<TID: TapestryId + Send + Sync>(
		&self,
		tapestry_id: TID,
	) -> crate::Result<Option<u16>> {
		let db = self.db.clone();
		let base_key = tapestry_id.base_key();
		let instance_count_key = format!("{}:{}", base_key, INSTANCE_COUNT_KEY);

		match db.get(instance_count_key.as_bytes()) {
			Ok(Some(value)) => {
				let instance_count_str =
					String::from_utf8(value).map_err(|_| StorageError::Parsing)?;
				let instance_count =
					instance_count_str.parse::<u16>().map_err(|_| StorageError::Parsing)?;
				Ok(Some(instance_count))
			},
			Ok(None) => Ok(None),
			Err(e) => {
				error!("Failed to get tapestry {}: {}", base_key, e);
				Err(LoomError::from(StorageError::RocksDb(e)).into())
			},
		}
	}

	async fn get_tapestry_fragment<TID: TapestryId + Send + Sync>(
		&self,
		tapestry_id: TID,
		instance: Option<u64>,
	) -> crate::Result<Option<TapestryFragment<T>>> {
		let db = self.db.clone();
		let base_key = tapestry_id.base_key();
		let instance = match self.verify_and_get_instance(&base_key, instance)? {
			Some(instance) => instance,
			None => return Ok(None),
		};

		let instance_key = format!("{}:{}", base_key, instance);
		match db.get(instance_key.as_bytes()) {
			Ok(Some(value)) => {
				let tapestry_fragment: TapestryFragment<T> = serde_json::from_slice(&value)
					.map_err(|e| {
						error!("Failed to deserialize tapestry fragment: {}", e);
						StorageError::Parsing
					})?;
				Ok(Some(tapestry_fragment))
			},
			Ok(None) => Ok(None),
			Err(e) => {
				error!("Failed to get tapestry fragment {}: {}", instance_key, e);
				Err(LoomError::from(StorageError::RocksDb(e)).into())
			},
		}
	}

	async fn get_tapestry_metadata<TID: TapestryId + Send + Sync, M: DeserializeOwned>(
		&self,
		tapestry_id: TID,
	) -> crate::Result<Option<M>> {
		let db = self.db.clone();
		let key = format!("{}:metadata", tapestry_id.base_key());
		match db.get(key.as_bytes()) {
			Ok(Some(value)) => {
				let metadata: M = serde_json::from_slice(&value).map_err(|e| {
					error!("Failed to parse tapestry metadata: {}", e);
					StorageError::Parsing
				})?;
				Ok(Some(metadata))
			},
			Ok(None) => Ok(None),
			Err(e) => {
				error!("Failed to get tapestry metadata {}: {}", key, e);
				Err(LoomError::from(StorageError::RocksDb(e)).into())
			},
		}
	}

	async fn delete_tapestry<TID: TapestryId + Send + Sync>(
		&self,
		tapestry_id: TID,
	) -> crate::Result<()> {
		let db = self.db.clone();
		let base_key = tapestry_id.base_key();
		let instance_count_key = format!("{}:{}", base_key, INSTANCE_COUNT_KEY);

		let instance_count = match db.get(instance_count_key.as_bytes()) {
			Ok(Some(value)) => {
				let instance_count_str =
					String::from_utf8(value).map_err(|_| StorageError::Parsing)?;
				instance_count_str.parse::<u64>().map_err(|_| StorageError::Parsing)?
			},
			Ok(None) => {
				debug!("Tapestry {} does not exist", base_key);
				return Ok(());
			},
			Err(e) => {
				error!("Failed to get instance count for {}: {}", base_key, e);
				return Err(LoomError::from(StorageError::RocksDb(e)).into());
			},
		};

		for i in 1..=instance_count {
			let instance_key = format!("{}:{}", base_key, i);
			if let Err(e) = db.delete(instance_key.as_bytes()) {
				error!("Failed to delete instance {}: {}", instance_key, e);
				return Err(LoomError::from(StorageError::RocksDb(e)).into());
			}
			debug!("Deleted instance key {}", instance_key);
		}

		if let Err(e) = db.delete(instance_count_key.as_bytes()) {
			error!("Failed to delete instance count {}: {}", instance_count_key, e);
			return Err(LoomError::from(StorageError::RocksDb(e)).into());
		}

		debug!("Deleted tapestry {} and its {} instances", base_key, instance_count);

		Ok(())
	}

	async fn delete_tapestry_fragment<TID: TapestryId + Send + Sync>(
		&self,
		tapestry_id: TID,
		instance: Option<u64>,
	) -> crate::Result<()> {
		let db = self.db.clone();
		let base_key = tapestry_id.base_key();
		let instance = match self.verify_and_get_instance(&base_key, instance)? {
			Some(instance) => instance,
			None => return Ok(()),
		};

		let instance_key = format!("{}:{}", base_key, instance);
		if let Err(e) = db.delete(instance_key.as_bytes()) {
			error!("Failed to delete tapestry fragment {}: {}", instance_key, e);
			return Err(LoomError::from(StorageError::RocksDb(e)).into());
		}

		debug!("Deleted tapestry fragment {}", instance_key);

		Ok(())
	}
}

impl RocksDbBackend {
	fn verify_and_get_instance(
		&self,
		base_key: &str,
		instance: Option<u64>,
	) -> crate::Result<Option<u64>> {
		let instance_count_key = format!("{}:{}", base_key, INSTANCE_COUNT_KEY);
		let instance_count = match self.db.get(instance_count_key.as_bytes()) {
			Ok(Some(value)) => {
				let instance_count_str =
					String::from_utf8(value).map_err(|_| StorageError::Parsing)?;
				instance_count_str.parse::<u64>().map_err(|_| StorageError::Parsing)?
			},
			Ok(None) => return Ok(None),
			Err(e) => {
				error!("Failed to get instance count for {}: {}", base_key, e);
				return Err(LoomError::from(StorageError::RocksDb(e)).into());
			},
		};

		let instance = match instance {
			Some(i) => {
				if i == 0 || i > instance_count {
					return Err(LoomError::from(StorageError::NotFound).into());
				}
				i
			},
			None => instance_count,
		};

		Ok(Some(instance))
	}
}
