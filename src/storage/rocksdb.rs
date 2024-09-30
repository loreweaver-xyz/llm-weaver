use async_trait::async_trait;
use rocksdb::{
	ColumnFamilyDescriptor, DBCompressionType, OptimisticTransactionDB, Options, Transaction,
};
use serde::{de::DeserializeOwned, Serialize};
use std::{
	fmt::Debug,
	sync::{Arc, Mutex},
};

use crate::{types::StorageError, Config, Result, TapestryFragment, TapestryId};

use super::TapestryChestHandler;

const INSTANCE_INDEX_CF: &str = "instance_counts";
const TAPESTRY_METADATA_CF: &str = "tapestry_metadata";
const TAPESTRY_FRAGMENT_CF: &str = "tapestry_fragments";

lazy_static::lazy_static! {
	static ref DB_INSTANCE: Mutex<Option<Arc<OptimisticTransactionDB>>> = Mutex::new(None);
}

#[derive(Clone)]
pub struct RocksDbBackend<T: Config> {
	db: Arc<OptimisticTransactionDB>,
	phantom: std::marker::PhantomData<T>,
}

impl<T: Config> RocksDbBackend<T> {
	pub fn new() -> Result<Self, T> {
		let mut db_instance = DB_INSTANCE.lock().unwrap();

		if let Some(db) = db_instance.as_ref() {
			return Ok(Self { db: Arc::clone(db), phantom: std::marker::PhantomData });
		}

		let db = Self::initialize_db()?;
		*db_instance = Some(Arc::clone(&db));

		Ok(Self { db, phantom: std::marker::PhantomData })
	}

	fn initialize_db() -> Result<Arc<OptimisticTransactionDB>, T> {
		let mut opts = Options::default();
		opts.create_if_missing(true);
		opts.create_missing_column_families(true);
		opts.increase_parallelism(num_cpus::get() as i32);
		opts.set_max_background_jobs(4);
		opts.set_max_write_buffer_number(3);
		opts.set_write_buffer_size(64 * 1024 * 1024); // 64MB
		opts.set_target_file_size_base(64 * 1024 * 1024); // 64MB
		opts.set_level_compaction_dynamic_level_bytes(true);
		opts.set_max_bytes_for_level_base(256 * 1024 * 1024); // 256MB
		opts.set_bloom_locality(1);
		opts.set_compression_type(DBCompressionType::Lz4);
		opts.set_periodic_compaction_seconds(86400); // 24 hours

		let mut cf_opts = Options::default();
		cf_opts.set_compression_type(DBCompressionType::Lz4);
		cf_opts.set_bottommost_compression_type(DBCompressionType::Zstd);

		let cf_descriptors = vec![
			ColumnFamilyDescriptor::new(INSTANCE_INDEX_CF, cf_opts.clone()),
			ColumnFamilyDescriptor::new(TAPESTRY_METADATA_CF, cf_opts.clone()),
			ColumnFamilyDescriptor::new(TAPESTRY_FRAGMENT_CF, cf_opts),
		];

		let db_path = std::env::var("ROCKSDB_PATH").unwrap_or_else(|_| "rocksdb-data".to_string());
		OptimisticTransactionDB::open_cf_descriptors(&opts, db_path, cf_descriptors)
			.map(Arc::new)
			.map_err(|e| StorageError::DatabaseError(e.to_string()).into())
	}

	fn transaction<F, R>(&self, func: F) -> Result<R, T>
	where
		F: FnOnce(&mut RocksDbTransaction<T>) -> Result<R, T>,
	{
		let txn = self.db.transaction();
		let mut txn_backend =
			RocksDbTransaction { txn, db: Arc::clone(&self.db), phantom: std::marker::PhantomData };
		let value = func(&mut txn_backend)?;
		txn_backend
			.txn
			.commit()
			.map_err(|e| StorageError::DatabaseError(e.to_string()))?;
		Ok(value)
	}
}

struct RocksDbTransaction<'a, T: Config> {
	txn: Transaction<'a, OptimisticTransactionDB>,
	db: Arc<OptimisticTransactionDB>,
	phantom: std::marker::PhantomData<T>,
}

impl<'a, T: Config> RocksDbTransaction<'a, T> {
	fn get_cf(&self, cf: &str, key: &[u8]) -> Result<Option<Vec<u8>>, T> {
		let cf_handle = self.db.cf_handle(cf).ok_or_else(|| {
			StorageError::DatabaseError(format!("Column family not found: {}", cf))
		})?;
		self.txn
			.get_cf(&cf_handle, key)
			.map_err(|e| StorageError::DatabaseError(e.to_string()).into())
	}

	fn put_cf(&self, cf: &str, key: &[u8], value: &[u8]) -> Result<(), T> {
		let cf_handle = self.db.cf_handle(cf).ok_or_else(|| {
			StorageError::DatabaseError(format!("Column family not found: {}", cf))
		})?;
		self.txn
			.put_cf(&cf_handle, key, value)
			.map_err(|e| StorageError::DatabaseError(e.to_string()).into())
	}

	fn delete_cf(&self, cf: &str, key: &[u8]) -> Result<(), T> {
		let cf_handle = self.db.cf_handle(cf).ok_or_else(|| {
			StorageError::DatabaseError(format!("Column family not found: {}", cf))
		})?;
		self.txn
			.delete_cf(&cf_handle, key)
			.map_err(|e| StorageError::DatabaseError(e.to_string()).into())
	}
}

#[async_trait]
impl<T: Config + Serialize + DeserializeOwned + Send + Sync> TapestryChestHandler<T>
	for RocksDbBackend<T>
{
	type Error = StorageError;

	fn new() -> Self {
		Self::new().expect("Failed to create RocksDB backend")
	}

	async fn save_tapestry_fragment<TID: TapestryId>(
		&self,
		tapestry_id: &TID,
		tapestry_fragment: TapestryFragment<T>,
		increment_index: bool,
	) -> Result<u64, T> {
		let fragment_bytes = serde_json::to_vec(&tapestry_fragment)
			.map_err(|e| StorageError::SerializationError(e.to_string()))?;

		self.transaction(|txn| {
			let instance_index_key = derive_instance_index_key(tapestry_id);
			let current_index = txn
				.get_cf(INSTANCE_INDEX_CF, instance_index_key.as_bytes())?
				.map(|bytes| {
					String::from_utf8(bytes)
						.map_err(|e| StorageError::DeserializationError(e.to_string()))
				})
				.transpose()?
				.map(|s| {
					s.parse::<u64>().map_err(|e| StorageError::DeserializationError(e.to_string()))
				})
				.transpose()?
				.unwrap_or(0);

			let new_instance_index =
				if increment_index { current_index + 1 } else { current_index.max(1) };

			txn.put_cf(
				INSTANCE_INDEX_CF,
				instance_index_key.as_bytes(),
				new_instance_index.to_string().as_bytes(),
			)?;

			let fragment_key = derive_instance_key(tapestry_id, new_instance_index);
			txn.put_cf(TAPESTRY_FRAGMENT_CF, fragment_key.as_bytes(), &fragment_bytes)?;

			Ok(new_instance_index)
		})
	}

	async fn save_tapestry_metadata<TID: TapestryId, M: Serialize + Debug + Clone + Send + Sync>(
		&self,
		tapestry_id: TID,
		metadata: M,
	) -> Result<(), T> {
		let metadata_bytes = serde_json::to_vec(&metadata)
			.map_err(|e| StorageError::SerializationError(e.to_string()))?;

		self.transaction(|txn| {
			let metadata_key = derive_metadata_key(&tapestry_id);
			txn.put_cf(TAPESTRY_METADATA_CF, metadata_key.as_bytes(), &metadata_bytes)
		})
	}

	async fn get_instance_index<TID: TapestryId>(
		&self,
		tapestry_id: TID,
	) -> Result<Option<u16>, T> {
		let instance_index_key = derive_instance_index_key(&tapestry_id);
		self.transaction(|txn| {
			txn.get_cf(INSTANCE_INDEX_CF, instance_index_key.as_bytes())?
				.map(|bytes| {
					String::from_utf8(bytes)
						.map_err(|e| StorageError::DeserializationError(e.to_string()))
				})
				.transpose()?
				.map(|s| {
					s.parse::<u16>().map_err(|e| StorageError::DeserializationError(e.to_string()))
				})
				.transpose()
				.map_err(|e| e.into())
		})
	}

	async fn get_tapestry_fragment<TID: TapestryId>(
		&self,
		tapestry_id: TID,
		instance: Option<u64>,
	) -> Result<Option<TapestryFragment<T>>, T> {
		self.transaction(|txn| {
			let instance = if let Some(i) = instance {
				i
			} else {
				let instance_index_key = derive_instance_index_key(&tapestry_id);
				match txn.get_cf(INSTANCE_INDEX_CF, instance_index_key.as_bytes())? {
					Some(count_bytes) => String::from_utf8(count_bytes)
						.map_err(|e| StorageError::DeserializationError(e.to_string()))?
						.parse::<u64>()
						.map_err(|e| StorageError::DeserializationError(e.to_string()))?,
					None => return Ok(None),
				}
			};

			let fragment_key = derive_instance_key(&tapestry_id, instance);
			txn.get_cf(TAPESTRY_FRAGMENT_CF, fragment_key.as_bytes())?
				.map(|bytes| {
					serde_json::from_slice(&bytes)
						.map_err(|e| StorageError::DeserializationError(e.to_string()))
				})
				.transpose()
				.map_err(|e| e.into())
		})
	}

	async fn get_tapestry_metadata<TID: TapestryId, M: DeserializeOwned + Send + Sync>(
		&self,
		tapestry_id: TID,
	) -> Result<Option<M>, T> {
		let metadata_key = derive_metadata_key(&tapestry_id);
		self.transaction(|txn| {
			txn.get_cf(TAPESTRY_METADATA_CF, metadata_key.as_bytes())?
				.map(|bytes| {
					serde_json::from_slice(&bytes)
						.map_err(|e| StorageError::DeserializationError(e.to_string()))
				})
				.transpose()
				.map_err(|e| e.into())
		})
	}

	async fn delete_tapestry<TID: TapestryId>(&self, tapestry_id: TID) -> Result<(), T> {
		self.transaction(|txn| {
			let instance_index_key = derive_instance_index_key(&tapestry_id);
			let current_index: u64 = txn
				.get_cf(INSTANCE_INDEX_CF, instance_index_key.as_bytes())?
				.map(|bytes| {
					String::from_utf8(bytes)
						.map_err(|e| StorageError::DeserializationError(e.to_string()))
				})
				.transpose()?
				.map(|s| {
					s.parse::<u64>().map_err(|e| StorageError::DeserializationError(e.to_string()))
				})
				.transpose()?
				.unwrap_or(0);

			// Delete all fragments
			for i in 1..=current_index {
				let fragment_key = derive_instance_key(&tapestry_id, i);
				txn.delete_cf(TAPESTRY_FRAGMENT_CF, fragment_key.as_bytes())?;
			}

			// Delete instance count
			txn.delete_cf(INSTANCE_INDEX_CF, instance_index_key.as_bytes())?;

			// Delete metadata
			let metadata_key = derive_metadata_key(&tapestry_id);
			txn.delete_cf(TAPESTRY_METADATA_CF, metadata_key.as_bytes())?;

			Ok(())
		})
	}

	async fn delete_tapestry_fragment<TID: TapestryId>(
		&self,
		tapestry_id: TID,
		instance: Option<u64>,
	) -> Result<(), T> {
		self.transaction(|txn| {
			let instance_index_key = derive_instance_index_key(&tapestry_id);
			let current_index: u64 = txn
				.get_cf(INSTANCE_INDEX_CF, instance_index_key.as_bytes())?
				.map(|bytes| {
					String::from_utf8(bytes)
						.map_err(|e| StorageError::DeserializationError(e.to_string()))
				})
				.transpose()?
				.map(|s| {
					s.parse::<u64>().map_err(|e| StorageError::DeserializationError(e.to_string()))
				})
				.transpose()?
				.unwrap_or(0);

			let instance_to_delete = instance.unwrap_or(current_index);

			if instance_to_delete > 0 && instance_to_delete <= current_index {
				let fragment_key = derive_instance_key(&tapestry_id, instance_to_delete);
				txn.delete_cf(TAPESTRY_FRAGMENT_CF, fragment_key.as_bytes())?;

				// Update instance count if we deleted the last fragment
				if instance_to_delete == current_index {
					let new_count = current_index - 1;
					if new_count > 0 {
						txn.put_cf(
							INSTANCE_INDEX_CF,
							instance_index_key.as_bytes(),
							new_count.to_string().as_bytes(),
						)?;
					} else {
						txn.delete_cf(INSTANCE_INDEX_CF, instance_index_key.as_bytes())?;
					}
				}
			}

			Ok(())
		})
	}
}

/// Derives the key for storing the instance count of a tapestry.
///
/// # Arguments
///
/// * `tapestry_id` - The identifier of the tapestry.
fn derive_instance_index_key<TID: TapestryId>(tapestry_id: &TID) -> String {
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

#[cfg(test)]
mod tests {
	use super::*;
	use crate::{
		types::{PromptModelTokens, SummaryModelTokens, WrapperRole},
		Config, ContextMessage, Llm, TapestryFragment, TapestryId,
	};
	use bounded_integer::BoundedU8;
	use serde::{Deserialize, Serialize};
	use std::{
		sync::Arc,
		time::{SystemTime, UNIX_EPOCH},
	};
	use tokio::{sync::Barrier, task};
	use uuid::Uuid;

	#[derive(Default, Debug, Clone, Serialize, Deserialize, PartialEq)]
	struct MockConfig;

	impl Config for MockConfig {
		const TOKEN_THRESHOLD_PERCENTILE: BoundedU8<0, 100> = BoundedU8::new(85).unwrap();
		const MINIMUM_RESPONSE_LENGTH: u64 = 100;

		type PromptModel = MockLlm;
		type SummaryModel = MockLlm;
		type Chest = RocksDbBackend<MockConfig>;

		fn convert_prompt_tokens_to_summary_model_tokens(
			tokens: PromptModelTokens<Self>,
		) -> SummaryModelTokens<Self> {
			tokens
		}
	}

	impl From<ContextMessage<MockConfig>> for String {
		fn from(msg: ContextMessage<MockConfig>) -> Self {
			msg.content
		}
	}

	#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
	struct MockLlm;

	#[async_trait]
	impl Llm<MockConfig> for MockLlm {
		type Tokens = u16;
		type Request = String;
		type Response = String;
		type Parameters = ();
		type PromptError = MockPromptError;

		fn max_context_length(&self) -> Self::Tokens {
			1000
		}

		fn name(&self) -> &'static str {
			"TestLlm"
		}

		fn alias(&self) -> &'static str {
			"TestLlm"
		}

		fn count_tokens(content: &str) -> Result<Self::Tokens, MockConfig> {
			Ok(content.len() as u16)
		}

		async fn prompt(
			&self,
			_is_summarizing: bool,
			_prompt_tokens: Self::Tokens,
			msgs: Vec<Self::Request>,
			_params: &Self::Parameters,
			_max_tokens: Self::Tokens,
		) -> Result<Self::Response, MockConfig> {
			Ok(msgs.join(" "))
		}

		fn compute_cost(&self, prompt_tokens: Self::Tokens, response_tokens: Self::Tokens) -> f64 {
			(prompt_tokens + response_tokens) as f64 * 0.001
		}
	}

	#[derive(Debug, Clone)]
	struct MockTapestryId(Uuid);

	impl TapestryId for MockTapestryId {
		fn base_key(&self) -> String {
			self.0.to_string()
		}
	}

	#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
	struct MockMetadata {
		created_at: u64,
		updated_at: u64,
		tag: String,
	}

	#[derive(Debug, Clone, thiserror::Error)]
	pub enum MockPromptError {}

	fn create_test_backend() -> RocksDbBackend<MockConfig> {
		RocksDbBackend::new().unwrap()
	}

	fn create_test_tapestry_fragment() -> TapestryFragment<MockConfig> {
		let mut fragment = TapestryFragment::default();
		fragment.context_tokens = 10;
		fragment.context_messages.push(ContextMessage::new(
			WrapperRole::from("user"),
			"Test message".to_string(),
			Some("user1".to_string()),
			SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs().to_string(),
		));
		fragment
	}

	fn create_test_metadata() -> MockMetadata {
		MockMetadata {
			created_at: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
			updated_at: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
			tag: "test".to_string(),
		}
	}

	#[tokio::test]
	async fn test_save_and_get_tapestry_fragment_with_metadata() {
		let backend = create_test_backend();
		let tapestry_id = MockTapestryId(Uuid::new_v4());
		let fragment = create_test_tapestry_fragment();
		let metadata = create_test_metadata();

		// Save metadata first
		<RocksDbBackend<MockConfig> as TapestryChestHandler<MockConfig>>::save_tapestry_metadata::<
			MockTapestryId,
			MockMetadata,
		>(&backend, tapestry_id.clone(), metadata.clone())
		.await
		.unwrap();

		// Save fragment
		let instance = backend
			.save_tapestry_fragment(&tapestry_id, fragment.clone(), true)
			.await
			.unwrap();
		assert_eq!(instance, 1);

		// Retrieve fragment
		let retrieved_fragment: TapestryFragment<MockConfig> = backend
			.get_tapestry_fragment(tapestry_id.clone(), Some(instance))
			.await
			.unwrap()
			.unwrap();
		assert_eq!(retrieved_fragment.context_tokens, fragment.context_tokens);
		assert_eq!(retrieved_fragment.context_messages.len(), fragment.context_messages.len());

		// Retrieve metadata
		let retrieved_metadata: MockMetadata = <RocksDbBackend<MockConfig> as TapestryChestHandler<
			MockConfig,
		>>::get_tapestry_metadata(&backend, tapestry_id)
		.await
		.unwrap()
		.unwrap();
		assert_eq!(retrieved_metadata, metadata);
	}

	#[tokio::test]
	async fn test_update_tapestry_metadata() {
		let backend = create_test_backend();
		let tapestry_id = MockTapestryId(Uuid::new_v4());
		let initial_metadata = create_test_metadata();
		let fragment = create_test_tapestry_fragment();

		// Save initial metadata and fragment
		<RocksDbBackend<MockConfig> as TapestryChestHandler<MockConfig>>::save_tapestry_metadata::<
			MockTapestryId,
			MockMetadata,
		>(&backend, tapestry_id.clone(), initial_metadata.clone())
		.await
		.unwrap();
		backend.save_tapestry_fragment(&tapestry_id, fragment, true).await.unwrap();

		// Update metadata
		let updated_metadata = MockMetadata {
			created_at: initial_metadata.created_at,
			updated_at: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
			tag: "updated".to_string(),
		};
		<RocksDbBackend<MockConfig> as TapestryChestHandler<MockConfig>>::save_tapestry_metadata::<
			MockTapestryId,
			MockMetadata,
		>(&backend, tapestry_id.clone(), updated_metadata.clone())
		.await
		.unwrap();

		// Retrieve updated metadata
		let retrieved_metadata: MockMetadata = <RocksDbBackend<MockConfig> as TapestryChestHandler<
			MockConfig,
		>>::get_tapestry_metadata(&backend, tapestry_id)
		.await
		.unwrap()
		.unwrap();
		assert_eq!(retrieved_metadata, updated_metadata);
	}

	#[tokio::test]
	async fn test_delete_tapestry_with_metadata() {
		let backend = create_test_backend();
		let tapestry_id = MockTapestryId(Uuid::new_v4());
		let metadata = create_test_metadata();
		let fragment = create_test_tapestry_fragment();

		// Save metadata and fragment
		<RocksDbBackend<MockConfig> as TapestryChestHandler<MockConfig>>::save_tapestry_metadata::<
			MockTapestryId,
			MockMetadata,
		>(&backend, tapestry_id.clone(), metadata)
		.await
		.unwrap();
		backend.save_tapestry_fragment(&tapestry_id, fragment, true).await.unwrap();

		// Delete tapestry
		<RocksDbBackend<MockConfig> as TapestryChestHandler<MockConfig>>::delete_tapestry(
			&backend,
			tapestry_id.clone(),
		)
		.await
		.unwrap();

		// Check if tapestry, metadata, and index are deleted
		let instance_count =
			<RocksDbBackend<MockConfig> as TapestryChestHandler<MockConfig>>::get_instance_index(
				&backend,
				tapestry_id.clone(),
			)
			.await
			.unwrap();
		assert_eq!(instance_count, None);

		let retrieved_metadata: Option<MockMetadata> = <RocksDbBackend<MockConfig> as TapestryChestHandler<
			MockConfig,
		>>::get_tapestry_metadata(
			&backend, tapestry_id.clone()
		)
		.await
		.unwrap();
		assert_eq!(retrieved_metadata, None);

		let retrieved_fragment =
			<RocksDbBackend<MockConfig> as TapestryChestHandler<MockConfig>>::get_tapestry_fragment(
				&backend,
				tapestry_id,
				Some(1),
			)
			.await
			.unwrap();
		assert_eq!(retrieved_fragment, None);
	}

	#[tokio::test]
	async fn test_concurrent_metadata_updates() {
		let backend = Arc::new(create_test_backend());
		let tapestry_id = Arc::new(MockTapestryId(Uuid::new_v4()));
		let barrier = Arc::new(Barrier::new(10));

		// Save initial metadata
		let initial_metadata = create_test_metadata();
		<RocksDbBackend<MockConfig> as TapestryChestHandler<MockConfig>>::save_tapestry_metadata::<
			MockTapestryId,
			MockMetadata,
		>(&backend, tapestry_id.as_ref().clone(), initial_metadata)
		.await
		.unwrap();

		let mut handles = vec![];

		for i in 0..10 {
			let backend = backend.clone();
			let tapestry_id = tapestry_id.clone();
			let barrier = barrier.clone();

			handles.push(task::spawn(async move {
				let mut updated_metadata = create_test_metadata();
				updated_metadata.tag = format!("concurrent_update_{}", i);

				barrier.wait().await;
				<RocksDbBackend<MockConfig> as TapestryChestHandler<MockConfig>>::save_tapestry_metadata::<
					MockTapestryId,
					MockMetadata,
				>(&backend, tapestry_id.as_ref().clone(), updated_metadata)
				.await
				.unwrap();

				<RocksDbBackend<MockConfig> as TapestryChestHandler<MockConfig>>::get_tapestry_metadata::<
					MockTapestryId,
					MockMetadata,
				>(&backend, tapestry_id.as_ref().clone())
				.await
				.unwrap()
				.unwrap()
			}));
		}

		let results: Vec<MockMetadata> = futures::future::join_all(handles)
			.await
			.into_iter()
			.filter_map(|r| r.ok())
			.collect();

		assert_eq!(results.len(), 10);
		let final_metadata: MockMetadata = <RocksDbBackend<MockConfig> as TapestryChestHandler<
			MockConfig,
		>>::get_tapestry_metadata(
			&backend, tapestry_id.as_ref().clone()
		)
		.await
		.unwrap()
		.unwrap();
		assert!(results.contains(&final_metadata));
	}

	#[tokio::test]
	async fn test_tapestry_fragment_versioning_with_metadata() {
		let backend = create_test_backend();
		let tapestry_id = MockTapestryId(Uuid::new_v4());
		let metadata = create_test_metadata();
		let fragment1 = create_test_tapestry_fragment();
		let mut fragment2 = create_test_tapestry_fragment();
		fragment2.context_messages[0].content = "Updated message".to_string();

		// Save metadata
		<RocksDbBackend<MockConfig> as TapestryChestHandler<MockConfig>>::save_tapestry_metadata::<
			MockTapestryId,
			MockMetadata,
		>(&backend, tapestry_id.clone(), metadata.clone())
		.await
		.unwrap();

		// Save fragments
		let instance1 = backend
			.save_tapestry_fragment(&tapestry_id, fragment1.clone(), true)
			.await
			.unwrap();
		let instance2 = backend
			.save_tapestry_fragment(&tapestry_id, fragment2.clone(), true)
			.await
			.unwrap();

		assert_eq!(instance1, 1);
		assert_eq!(instance2, 2);

		// Retrieve fragments
		let retrieved_fragment1 = <RocksDbBackend<MockConfig> as TapestryChestHandler<
			MockConfig,
		>>::get_tapestry_fragment(
			&backend, tapestry_id.clone(), Some(1)
		)
		.await
		.unwrap()
		.unwrap();
		let retrieved_fragment2 = <RocksDbBackend<MockConfig> as TapestryChestHandler<
			MockConfig,
		>>::get_tapestry_fragment(
			&backend, tapestry_id.clone(), Some(2)
		)
		.await
		.unwrap()
		.unwrap();

		assert_eq!(
			retrieved_fragment1.context_messages[0].content,
			fragment1.context_messages[0].content
		);
		assert_eq!(
			retrieved_fragment2.context_messages[0].content,
			fragment2.context_messages[0].content
		);

		// Retrieve metadata
		let retrieved_metadata: MockMetadata = <RocksDbBackend<MockConfig> as TapestryChestHandler<
			MockConfig,
		>>::get_tapestry_metadata(&backend, tapestry_id)
		.await
		.unwrap()
		.unwrap();
		assert_eq!(retrieved_metadata, metadata);
	}

	#[tokio::test]
	async fn test_save_multiple_fragments_without_incrementing_index() {
		let backend = create_test_backend();
		let tapestry_id = MockTapestryId(Uuid::new_v4());
		let metadata = create_test_metadata();
		let fragment1 = create_test_tapestry_fragment();
		let fragment2 = create_test_tapestry_fragment();

		// Save metadata
		<RocksDbBackend<MockConfig> as TapestryChestHandler<MockConfig>>::save_tapestry_metadata::<
			MockTapestryId,
			MockMetadata,
		>(&backend, tapestry_id.clone(), metadata)
		.await
		.unwrap();

		// Save fragments without incrementing index
		let instance1 =
			backend.save_tapestry_fragment(&tapestry_id, fragment1, false).await.unwrap();
		let instance2 = backend
			.save_tapestry_fragment(&tapestry_id, fragment2.clone(), false)
			.await
			.unwrap();

		assert_eq!(instance1, 1);
		assert_eq!(instance2, 1);

		// Retrieve the latest fragment
		let retrieved_fragment =
			<RocksDbBackend<MockConfig> as TapestryChestHandler<MockConfig>>::get_tapestry_fragment(
				&backend,
				tapestry_id.clone(),
				None,
			)
			.await
			.unwrap()
			.unwrap();

		// Check that we have the latest fragment
		assert_eq!(
			retrieved_fragment.context_messages[0].content,
			fragment2.context_messages[0].content
		);

		// Check the tapestry index
		let instance_count =
			<RocksDbBackend<MockConfig> as TapestryChestHandler<MockConfig>>::get_instance_index(
				&backend,
				tapestry_id,
			)
			.await
			.unwrap()
			.unwrap();
		assert_eq!(instance_count, 1);
	}

	#[tokio::test]
	async fn test_delete_specific_tapestry_fragment() {
		let backend = create_test_backend();
		let tapestry_id = MockTapestryId(Uuid::new_v4());
		let metadata = create_test_metadata();
		let fragment1 = create_test_tapestry_fragment();
		let mut fragment2 = create_test_tapestry_fragment();
		fragment2.context_messages[0].content = "Second fragment".to_string();

		// Save metadata
		<RocksDbBackend<MockConfig> as TapestryChestHandler<MockConfig>>::save_tapestry_metadata::<
			MockTapestryId,
			MockMetadata,
		>(&backend, tapestry_id.clone(), metadata)
		.await
		.unwrap();

		// Save fragments
		backend
			.save_tapestry_fragment(&tapestry_id, fragment1.clone(), true)
			.await
			.unwrap();
		backend
			.save_tapestry_fragment(&tapestry_id, fragment2.clone(), true)
			.await
			.unwrap();

		// Delete the first fragment
		<RocksDbBackend<MockConfig> as TapestryChestHandler<MockConfig>>::delete_tapestry_fragment(
			&backend,
			tapestry_id.clone(),
			Some(1),
		)
		.await
		.unwrap();

		// Try to retrieve the deleted fragment
		let deleted_fragment =
			<RocksDbBackend<MockConfig> as TapestryChestHandler<MockConfig>>::get_tapestry_fragment(
				&backend,
				tapestry_id.clone(),
				Some(1),
			)
			.await
			.unwrap();
		assert_eq!(deleted_fragment, None);

		// Retrieve the remaining fragment
		let remaining_fragment =
			<RocksDbBackend<MockConfig> as TapestryChestHandler<MockConfig>>::get_tapestry_fragment(
				&backend,
				tapestry_id.clone(),
				Some(2),
			)
			.await
			.unwrap()
			.unwrap();
		assert_eq!(
			remaining_fragment.context_messages[0].content,
			fragment2.context_messages[0].content
		);

		// Check the tapestry index
		let instance_count =
			<RocksDbBackend<MockConfig> as TapestryChestHandler<MockConfig>>::get_instance_index(
				&backend,
				tapestry_id,
			)
			.await
			.unwrap()
			.unwrap();
		assert_eq!(instance_count, 2);
	}

	#[tokio::test]
	async fn test_concurrent_fragment_and_metadata_operations() {
		let backend = Arc::new(create_test_backend());
		let tapestry_id = Arc::new(MockTapestryId(Uuid::new_v4()));
		let barrier = Arc::new(Barrier::new(20));

		// Save initial metadata
		let initial_metadata = create_test_metadata();
		<RocksDbBackend<MockConfig> as TapestryChestHandler<MockConfig>>::save_tapestry_metadata::<
			MockTapestryId,
			MockMetadata,
		>(&backend, tapestry_id.as_ref().clone(), initial_metadata)
		.await
		.unwrap();

		let mut handles: Vec<task::JoinHandle<()>> = vec![];

		for i in 0..10 {
			let backend = backend.clone();
			let tapestry_id = tapestry_id.clone();
			let barrier = barrier.clone();

			handles.push(task::spawn(async move {
				let mut fragment = create_test_tapestry_fragment();
				fragment.context_messages[0].content = format!("Concurrent fragment {}", i);
				barrier.wait().await;
				backend.save_tapestry_fragment(&*tapestry_id, fragment, true).await.unwrap();
			}));
		}

		for i in 0..10 {
			let backend = backend.clone();
			let tapestry_id = tapestry_id.clone();
			let barrier = barrier.clone();

			handles.push(task::spawn(async move {
				let mut updated_metadata = create_test_metadata();
				updated_metadata.tag = format!("concurrent_update_{}", i);

				barrier.wait().await;
				<RocksDbBackend<MockConfig> as TapestryChestHandler<MockConfig>>::save_tapestry_metadata::<
					MockTapestryId,
					MockMetadata,
				>(&backend, tapestry_id.as_ref().clone(), updated_metadata)
				.await
				.unwrap()
			}));
		}

		let _ = futures::future::join_all(handles).await;

		// Check final state
		let instance_count =
			<RocksDbBackend<MockConfig> as TapestryChestHandler<MockConfig>>::get_instance_index(
				&backend,
				tapestry_id.as_ref().clone(),
			)
			.await
			.unwrap()
			.unwrap();
		assert_eq!(instance_count, 10);

		let final_metadata: MockMetadata = <RocksDbBackend<MockConfig> as TapestryChestHandler<
			MockConfig,
		>>::get_tapestry_metadata(
			&backend, tapestry_id.as_ref().clone()
		)
		.await
		.unwrap()
		.unwrap();
		assert!(final_metadata.tag.starts_with("concurrent_update_"));
	}

	#[tokio::test]
	async fn test_save_fragment_updates_metadata() {
		let backend = create_test_backend();
		let tapestry_id = MockTapestryId(Uuid::new_v4());
		let initial_metadata = create_test_metadata();
		let fragment = create_test_tapestry_fragment();

		// Save initial metadata
		<RocksDbBackend<MockConfig> as TapestryChestHandler<MockConfig>>::save_tapestry_metadata::<
			MockTapestryId,
			MockMetadata,
		>(&backend, tapestry_id.clone(), initial_metadata.clone())
		.await
		.unwrap();

		// Save fragment
		backend.save_tapestry_fragment(&tapestry_id, fragment, true).await.unwrap();

		// Retrieve updated metadata
		let updated_metadata: MockMetadata = <RocksDbBackend<MockConfig> as TapestryChestHandler<
			MockConfig,
		>>::get_tapestry_metadata(&backend, tapestry_id)
		.await
		.unwrap()
		.unwrap();

		assert_eq!(updated_metadata, initial_metadata);
	}

	#[tokio::test]
	async fn test_get_non_existent_tapestry() {
		let backend = create_test_backend();
		let tapestry_id = MockTapestryId(Uuid::new_v4());

		let result =
			<RocksDbBackend<MockConfig> as TapestryChestHandler<MockConfig>>::get_instance_index(
				&backend,
				tapestry_id,
			)
			.await
			.unwrap();
		assert_eq!(result, None);
	}

	#[tokio::test]
	async fn test_get_non_existent_tapestry_metadata() {
		let backend = create_test_backend();
		let tapestry_id = MockTapestryId(Uuid::new_v4());

		let result: Option<MockMetadata> = <RocksDbBackend<MockConfig> as TapestryChestHandler<
			MockConfig,
		>>::get_tapestry_metadata(&backend, tapestry_id)
		.await
		.unwrap();
		assert_eq!(result, None);
	}

	#[tokio::test]
	async fn test_delete_non_existent_tapestry() {
		let backend = create_test_backend();
		let tapestry_id = MockTapestryId(Uuid::new_v4());

		let result =
			<RocksDbBackend<MockConfig> as TapestryChestHandler<MockConfig>>::delete_tapestry(
				&backend,
				tapestry_id,
			)
			.await;
		assert!(result.is_ok());
	}

	#[tokio::test]
	async fn test_delete_non_existent_tapestry_fragment() {
		let backend = create_test_backend();
		let tapestry_id = MockTapestryId(Uuid::new_v4());

		let result =
			<RocksDbBackend<MockConfig> as TapestryChestHandler<MockConfig>>::delete_tapestry_fragment(
				&backend,
				tapestry_id,
				Some(1),
			)
			.await;
		assert!(result.is_ok());
	}
}
