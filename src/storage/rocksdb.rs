use async_trait::async_trait;
use rocksdb::{
	BoundColumnFamily, ColumnFamilyDescriptor, MultiThreaded, OptimisticTransactionDB, Options,
};
use serde::{de::DeserializeOwned, Serialize};
use std::{fmt::Debug, sync::Arc};
use tracing::error;

use crate::{types::StorageError, Config, TapestryFragment, TapestryId};

use super::TapestryChestHandler;

/// RocksDB implementation of the storage backend.
///
/// This struct provides methods to interact with the RocksDB database
/// for storing tapestry fragments, metadata, and instance counts.
/// It utilizes column families to separate different types of data,
/// improving data organization and access efficiency.
#[derive(Clone)]
pub struct RocksDbBackend<'db> {
	db: Arc<OptimisticTransactionDB<MultiThreaded>>,
	phantom: std::marker::PhantomData<&'db ()>,
}

impl<'db> RocksDbBackend<'db> {
	/// Creates a new instance of the RocksDbBackend.
	///
	/// Initializes the RocksDB database with specified column families
	/// for instance counts, tapestry metadata, and tapestry fragments.
	fn new() -> Result<Self, StorageError> {
		let mut opts = Options::default();
		opts.create_if_missing(true);
		opts.create_missing_column_families(true);

		// Set up cache and memory usage options
		let cache = rocksdb::Cache::new_lru_cache(1 << 20); // 1 MB cache
		let mut block_opts = rocksdb::BlockBasedOptions::default();
		block_opts.set_block_cache(&cache);
		opts.set_block_based_table_factory(&block_opts);

		// Define the column families
		let cf_descriptors = vec![
			ColumnFamilyDescriptor::new("instance_counts", Options::default()),
			ColumnFamilyDescriptor::new("tapestry_metadata", Options::default()),
			ColumnFamilyDescriptor::new("tapestry_fragments", Options::default()),
		];

		// Open the database with the specified column families
		let db = OptimisticTransactionDB::<MultiThreaded>::open_cf_descriptors(
			&opts,
			"rocksdb-data",
			cf_descriptors,
		)
		.map_err(StorageError::RocksDb)?;

		Ok(Self { db: Arc::new(db), phantom: std::marker::PhantomData })
	}

	fn get_cf_instance_counts(&'db self) -> Result<Arc<BoundColumnFamily<'db>>, StorageError> {
		self.db
			.cf_handle("instance_counts")
			.ok_or_else(|| StorageError::InternalError("Column family not found".to_string()))
	}

	fn get_cf_tapestry_metadata(&'db self) -> Result<Arc<BoundColumnFamily<'db>>, StorageError> {
		self.db
			.cf_handle("tapestry_metadata")
			.ok_or_else(|| StorageError::InternalError("Column family not found".to_string()))
	}

	fn get_cf_tapestry_fragments(&'db self) -> Result<Arc<BoundColumnFamily<'db>>, StorageError> {
		self.db
			.cf_handle("tapestry_fragments")
			.ok_or_else(|| StorageError::InternalError("Column family not found".to_string()))
	}

	/// Executes a transactional operation on the database.
	///
	/// The provided function `func` is executed within a transaction.
	/// If the function returns an error, the transaction is aborted.
	///
	/// # Arguments
	///
	/// * `func` - A closure that takes a mutable reference to a `RocksDbTransaction` and returns a
	///   `Result`.
	pub fn transaction<F, T>(&self, func: F) -> Result<T, StorageError>
	where
		F: FnOnce(&mut RocksDbTransaction) -> Result<T, StorageError>,
	{
		let txn = self.db.transaction();
		let mut txn_backend = RocksDbTransaction { txn };
		let value = func(&mut txn_backend)?;
		txn_backend.txn.commit().map_err(StorageError::RocksDb)?;
		Ok(value)
	}

	/// Retrieves a value from the database by key and column family.
	fn get_cf(
		&self,
		cf: &Arc<BoundColumnFamily<'db>>,
		key: &str,
	) -> Result<Option<Vec<u8>>, StorageError> {
		self.db.get_cf(cf, key.as_bytes()).map_err(StorageError::RocksDb)
	}

	/// Inserts a key-value pair into the database under the specified column family.
	fn put_cf(
		&self,
		cf: &Arc<BoundColumnFamily<'db>>,
		key: &str,
		value: &[u8],
	) -> Result<(), StorageError> {
		self.db.put_cf(cf, key.as_bytes(), value).map_err(StorageError::RocksDb)
	}

	/// Deletes a key from the database under the specified column family.
	fn delete_cf(&self, cf: &Arc<BoundColumnFamily<'db>>, key: &str) -> Result<(), StorageError> {
		self.db.delete_cf(cf, key.as_bytes()).map_err(StorageError::RocksDb)
	}

	/// Checks if a key exists in the database under the specified column family.
	fn key_exists_cf(
		&self,
		cf: &Arc<BoundColumnFamily<'db>>,
		key: &str,
	) -> Result<bool, StorageError> {
		match self.db.get_cf(cf, key.as_bytes()) {
			Ok(Some(_)) => Ok(true),
			Ok(None) => Ok(false),
			Err(e) => Err(StorageError::RocksDb(e)),
		}
	}
}

/// A transactional wrapper around RocksDB operations.
///
/// This struct provides methods to perform database operations within a transaction.
struct RocksDbTransaction<'db> {
	txn: rocksdb::Transaction<'db, OptimisticTransactionDB<MultiThreaded>>,
}

impl<'db> RocksDbTransaction<'db> {
	/// Retrieves a value from the transaction by key and column family.
	fn get_cf(
		&self,
		cf: &Arc<BoundColumnFamily<'db>>,
		key: &str,
	) -> Result<Option<Vec<u8>>, StorageError> {
		self.txn.get_cf(cf, key.as_bytes()).map_err(StorageError::RocksDb)
	}

	/// Inserts a key-value pair into the transaction under the specified column family.
	fn put_cf(
		&self,
		cf: &Arc<BoundColumnFamily<'db>>,
		key: &str,
		value: &[u8],
	) -> Result<(), StorageError> {
		self.txn.put_cf(cf, key.as_bytes(), value).map_err(StorageError::RocksDb)
	}

	/// Deletes a key from the transaction under the specified column family.
	fn delete_cf(&self, cf: &Arc<BoundColumnFamily<'db>>, key: &str) -> Result<(), StorageError> {
		self.txn.delete_cf(cf, key.as_bytes()).map_err(StorageError::RocksDb)
	}

	/// Checks if a key exists in the transaction under the specified column family.
	fn key_exists_cf(
		&self,
		cf: &Arc<BoundColumnFamily<'db>>,
		key: &str,
	) -> Result<bool, StorageError> {
		match self.txn.get_cf(cf, key.as_bytes()) {
			Ok(Some(_)) => Ok(true),
			Ok(None) => Ok(false),
			Err(e) => Err(StorageError::RocksDb(e)),
		}
	}
}

#[async_trait]
impl<'db, T: Config + Serialize + DeserializeOwned + Send + Sync> TapestryChestHandler<T>
	for RocksDbBackend<'db>
{
	type Error = StorageError;

	/// Creates a new instance of the RocksDbBackend.
	fn new() -> Self {
		Self::new().expect("Failed to create RocksDB backend")
	}

	/// Saves a tapestry fragment to the database.
	///
	/// If `increment` is `true`, the instance count is incremented.
	/// Otherwise, it uses the current instance count or sets it to 1 if not present.
	///
	/// # Arguments
	///
	/// * `tapestry_id` - The identifier of the tapestry.
	/// * `tapestry_fragment` - The tapestry fragment to save.
	/// * `increment` - Whether to increment the instance count.
	async fn save_tapestry_fragment<TID: TapestryId + Send + Sync + 'static>(
		&self,
		tapestry_id: &TID,
		tapestry_fragment: TapestryFragment<T>,
		increment: bool,
	) -> crate::Result<u64> {
		let fragment_bytes = serde_json::to_vec(&tapestry_fragment).map_err(|e| {
			error!("Failed to serialize tapestry fragment: {}", e);
			StorageError::Parsing
		})?;

		let backend = self.clone();
		let tapestry_id = tapestry_id.clone();

		let new_count = backend.transaction(|txn| {
			let instance_count_key_cf = backend.get_cf_instance_counts()?;
			let tapestry_fragments_cf = backend.get_cf_tapestry_fragments()?;

			// Read the current instance count from cf_instance_counts
			let instance_count_key = derive_instance_count_key(&tapestry_id);
			let current_count_bytes = txn.get_cf(&instance_count_key_cf, &instance_count_key)?;
			let current_count = match current_count_bytes {
				Some(bytes) => deserialize_instance_count(bytes)?,
				None => 0,
			};

			let new_count = if increment {
				current_count + 1
			} else if current_count == 0 {
				1
			} else {
				current_count
			};

			// Write the updated count to cf_instance_counts
			txn.put_cf(
				&instance_count_key_cf,
				&instance_count_key,
				new_count.to_string().as_bytes(),
			)?;

			// Write the tapestry fragment to cf_tapestry_fragments
			let instance_key = derive_instance_key(&tapestry_id, new_count);
			txn.put_cf(&tapestry_fragments_cf, &instance_key, &fragment_bytes)?;

			Ok(new_count)
		})?;

		Ok(new_count)
	}

	/// Saves tapestry metadata to the database.
	///
	/// # Arguments
	///
	/// * `tapestry_id` - The identifier of the tapestry.
	/// * `metadata` - The metadata to save.
	async fn save_tapestry_metadata<TID: TapestryId, M: Serialize + Debug + Clone + Send + Sync>(
		&self,
		tapestry_id: TID,
		metadata: M,
	) -> crate::Result<()> {
		let backend = self.clone();

		let metadata_bytes = serde_json::to_vec(&metadata).map_err(|e| {
			error!("Failed to serialize tapestry metadata: {}", e);
			StorageError::Parsing
		})?;

		backend
			.transaction(|txn| {
				let instance_counts_cf = backend.get_cf_instance_counts()?;
				let tapestry_metadata_cf = backend.get_cf_tapestry_metadata()?;
				let tapestry_fragments_cf = backend.get_cf_tapestry_fragments()?;

				// Check if the tapestry exists
				let instance_count_key = derive_instance_count_key(&tapestry_id);
				if !backend.key_exists_cf(&instance_counts_cf, &instance_count_key)? {
					return Err(StorageError::NotFound.into());
				}

				let instance_count_bytes =
					backend.get_cf(&instance_counts_cf, &instance_count_key)?;
				let instance_count = match instance_count_bytes {
					Some(bytes) => deserialize_instance_count(bytes)?,
					None => return Err(StorageError::NotFound.into()),
				};

				// Check if tapestry instance exists
				let instance_key = derive_instance_key(&tapestry_id, instance_count);
				if !backend.key_exists_cf(&tapestry_fragments_cf, &instance_key)? {
					return Err(StorageError::NotFound.into());
				}

				let metadata_key = derive_metadata_key(&tapestry_id);

				txn.put_cf(&tapestry_metadata_cf, &metadata_key, &metadata_bytes)?;

				Ok(())
			})
			.map_err(|e| e.into())
	}

	/// Retrieves the instance count of a tapestry.
	///
	/// # Arguments
	///
	/// * `tapestry_id` - The identifier of the tapestry.
	async fn get_tapestry<TID: TapestryId + Send + Sync + 'static>(
		&self,
		tapestry_id: TID,
	) -> crate::Result<Option<u16>> {
		let backend = self.clone();

		let instance_counts_cf = backend.get_cf_instance_counts()?;

		let instance_count_key = derive_instance_count_key(&tapestry_id);
		match self.get_cf(&instance_counts_cf, &instance_count_key)? {
			Some(bytes) => {
				let count = deserialize_instance_count(bytes)?;
				Ok(Some(count as u16))
			},
			None => Ok(None),
		}
	}

	/// Retrieves a tapestry fragment from the database.
	///
	/// If `instance` is `None`, the latest instance is retrieved.
	///
	/// # Arguments
	///
	/// * `tapestry_id` - The identifier of the tapestry.
	/// * `instance` - The instance number to retrieve.
	async fn get_tapestry_fragment<TID: TapestryId + Send + Sync + 'static>(
		&self,
		tapestry_id: TID,
		instance: Option<u64>,
	) -> crate::Result<Option<TapestryFragment<T>>> {
		let backend = self.clone();

		let instance_counts_cf = backend.get_cf_instance_counts()?;
		let tapestry_fragments_cf = backend.get_cf_tapestry_fragments()?;

		let instance = match instance {
			Some(i) => i,
			None => {
				let instance_count_key = derive_instance_count_key(&tapestry_id);
				match backend.get_cf(&instance_counts_cf, &instance_count_key)? {
					Some(bytes) => deserialize_instance_count(bytes)?,
					None => return Ok(None),
				}
			},
		};

		let instance_key = derive_instance_key(&tapestry_id, instance);
		match backend.get_cf(&tapestry_fragments_cf, &instance_key)? {
			Some(bytes) => {
				let fragment: TapestryFragment<T> =
					serde_json::from_slice(&bytes).map_err(|e| {
						error!("Failed to deserialize tapestry fragment: {}", e);
						StorageError::Parsing
					})?;
				Ok(Some(fragment))
			},
			None => Ok(None),
		}
	}

	/// Retrieves tapestry metadata from the database.
	///
	/// # Arguments
	///
	/// * `tapestry_id` - The identifier of the tapestry.
	async fn get_tapestry_metadata<
		TID: TapestryId + Send + Sync + 'static,
		M: DeserializeOwned + Send + Sync,
	>(
		&self,
		tapestry_id: TID,
	) -> crate::Result<Option<M>> {
		let backend = self.clone();

		let cf_tapestry_metadata = backend.get_cf_tapestry_metadata()?;

		let metadata_key = derive_metadata_key(&tapestry_id);
		match backend.get_cf(&cf_tapestry_metadata, &metadata_key)? {
			Some(bytes) => {
				let metadata: M = serde_json::from_slice(&bytes).map_err(|e| {
					error!("Failed to parse tapestry metadata: {}", e);
					StorageError::Parsing
				})?;
				Ok(Some(metadata))
			},
			None => Ok(None),
		}
	}

	/// Deletes a tapestry and all its fragments and metadata from the database.
	///
	/// # Arguments
	///
	/// * `tapestry_id` - The identifier of the tapestry.
	async fn delete_tapestry<TID: TapestryId + Send + Sync + 'static>(
		&self,
		tapestry_id: TID,
	) -> crate::Result<()> {
		let backend = self.clone();

		backend
			.transaction(|txn| {
				let instance_counts_cf = backend.get_cf_instance_counts()?;
				let tapestry_metadata_cf = backend.get_cf_tapestry_metadata()?;
				let tapestry_fragments_cf = backend.get_cf_tapestry_fragments()?;

				let instance_count_key = derive_instance_count_key(&tapestry_id);
				let instance_count =
					match backend.get_cf(&instance_counts_cf, &instance_count_key)? {
						Some(bytes) => deserialize_instance_count(bytes)?,
						None => return Ok(()), // Nothing to delete
					};

				// Delete each instance
				for instance in 1..=instance_count {
					let instance_key = derive_instance_key(&tapestry_id, instance);
					txn.delete_cf(&tapestry_fragments_cf, &instance_key)?;
				}
				// Delete the instance count key
				txn.delete_cf(&instance_counts_cf, &instance_count_key)?;

				// Delete the metadata key
				let metadata_key = derive_metadata_key(&tapestry_id);
				txn.delete_cf(&tapestry_metadata_cf, &metadata_key)?;

				Ok(())
			})
			.map_err(|e| e.into())
	}

	/// Deletes a specific tapestry fragment from the database.
	///
	/// If `instance` is `None`, the latest instance is deleted.
	///
	/// # Arguments
	///
	/// * `tapestry_id` - The identifier of the tapestry.
	/// * `instance` - The instance number to delete.
	async fn delete_tapestry_fragment<TID: TapestryId + Send + Sync + 'static>(
		&self,
		tapestry_id: TID,
		instance: Option<u64>,
	) -> crate::Result<()> {
		let backend = self.clone();

		backend
			.transaction(|txn| {
				let instance_counts_cf = backend.get_cf_instance_counts()?;
				let tapestry_fragments_cf = backend.get_cf_tapestry_fragments()?;

				let instance = match instance {
					Some(i) => i,
					None => {
						// Get the instance count from the database
						let instance_count_key = derive_instance_count_key(&tapestry_id);
						match backend.get_cf(&instance_counts_cf, &instance_count_key)? {
							Some(bytes) => deserialize_instance_count(bytes)?,
							None => return Ok(()), // Nothing to delete
						}
					},
				};

				let instance_key = derive_instance_key(&tapestry_id, instance);
				let instance_count_key = derive_instance_count_key(&tapestry_id);

				txn.delete_cf(&tapestry_fragments_cf, &instance_key)?;

				// Update the instance count if needed
				let current_count_bytes = txn.get_cf(&instance_counts_cf, &instance_count_key)?;
				let current_count = match current_count_bytes {
					Some(bytes) => deserialize_instance_count(bytes)?,
					None => return Ok(()), // No instance count, nothing to update
				};

				if current_count == instance {
					// Decrease the instance count
					let new_count = current_count - 1;
					if new_count == 0 {
						txn.delete_cf(&instance_counts_cf, &instance_count_key)?;
					} else {
						txn.put_cf(
							&instance_counts_cf,
							&instance_count_key,
							new_count.to_string().as_bytes(),
						)?;
					}
				}

				Ok(())
			})
			.map_err(|e| e.into())
	}
}

/// Deserializes the instance count from a byte vector.
///
/// # Arguments
///
/// * `bytes` - The byte vector containing the instance count.
fn deserialize_instance_count(bytes: Vec<u8>) -> Result<u64, StorageError> {
	let count_str = String::from_utf8(bytes).map_err(|_| StorageError::Parsing)?;
	count_str.parse::<u64>().map_err(|_| StorageError::Parsing)
}

/// Derives the key for storing the instance count of a tapestry.
///
/// # Arguments
///
/// * `tapestry_id` - The identifier of the tapestry.
fn derive_instance_count_key<TID: TapestryId>(tapestry_id: &TID) -> String {
	tapestry_id.base_key()
}

/// Derives the key for storing a specific instance of a tapestry fragment.
///
/// # Arguments
///
/// * `tapestry_id` - The identifier of the tapestry.
/// * `instance` - The instance number.
fn derive_instance_key<TID: TapestryId>(tapestry_id: &TID, instance: u64) -> String {
	format!("{}:{}", tapestry_id.base_key(), instance)
}

/// Derives the key for storing the metadata of a tapestry.
///
/// # Arguments
///
/// * `tapestry_id` - The identifier of the tapestry.
fn derive_metadata_key<TID: TapestryId>(tapestry_id: &TID) -> String {
	tapestry_id.base_key()
}
