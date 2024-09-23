use async_trait::async_trait;
use rocksdb::{ColumnFamilyDescriptor, OptimisticTransactionDB, Options, Transaction};
use serde::{de::DeserializeOwned, Serialize};
use std::{fmt::Debug, sync::Arc};

use crate::{types::StorageError, Config, Result, TapestryFragment, TapestryId};

use super::TapestryChestHandler;

const INSTANCE_INDEX_CF: &str = "instance_counts";
const TAPESTRY_METADATA_CF: &str = "tapestry_metadata";
const TAPESTRY_FRAGMENT_CF: &str = "tapestry_fragments";

pub struct RocksDbBackend {
	db: Arc<OptimisticTransactionDB>,
}

impl RocksDbBackend {
	pub fn new() -> Result<Self> {
		let mut opts = Options::default();
		opts.create_if_missing(true);
		opts.create_missing_column_families(true);

		let cf_descriptors = vec![
			ColumnFamilyDescriptor::new(INSTANCE_INDEX_CF, Options::default()),
			ColumnFamilyDescriptor::new(TAPESTRY_METADATA_CF, Options::default()),
			ColumnFamilyDescriptor::new(TAPESTRY_FRAGMENT_CF, Options::default()),
		];

		let db =
			OptimisticTransactionDB::open_cf_descriptors(&opts, "rocksdb-data", cf_descriptors)
				.map_err(|e| StorageError::DatabaseError(e.to_string()))?;

		Ok(Self { db: Arc::new(db) })
	}

	fn transaction<F, T>(&self, func: F) -> Result<T>
	where
		F: FnOnce(&mut RocksDbTransaction) -> Result<T>,
	{
		let txn = self.db.transaction();
		let mut txn_backend = RocksDbTransaction { txn, db: self.db.clone() };
		let value = func(&mut txn_backend)?;
		txn_backend
			.txn
			.commit()
			.map_err(|e| StorageError::DatabaseError(e.to_string()))?;
		Ok(value)
	}
}

struct RocksDbTransaction<'a> {
	txn: Transaction<'a, OptimisticTransactionDB>,
	db: Arc<OptimisticTransactionDB>,
}

impl<'a> RocksDbTransaction<'a> {
	fn get_cf(&self, cf: &str, key: &[u8]) -> Result<Option<Vec<u8>>> {
		let cf_handle = self.db.cf_handle(cf).ok_or_else(|| {
			StorageError::DatabaseError(format!("Column family not found: {}", cf))
		})?;
		self.txn
			.get_cf(&cf_handle, key)
			.map_err(|e| StorageError::DatabaseError(e.to_string()).into())
	}

	fn put_cf(&self, cf: &str, key: &[u8], value: &[u8]) -> Result<()> {
		let cf_handle = self.db.cf_handle(cf).ok_or_else(|| {
			StorageError::DatabaseError(format!("Column family not found: {}", cf))
		})?;
		self.txn
			.put_cf(&cf_handle, key, value)
			.map_err(|e| StorageError::DatabaseError(e.to_string()).into())
	}

	fn delete_cf(&self, cf: &str, key: &[u8]) -> Result<()> {
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
	for RocksDbBackend
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
	) -> Result<u64> {
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
	) -> Result<()> {
		let metadata_bytes = serde_json::to_vec(&metadata)
			.map_err(|e| StorageError::SerializationError(e.to_string()))?;

		self.transaction(|txn| {
			let metadata_key = derive_metadata_key(&tapestry_id);
			txn.put_cf(TAPESTRY_METADATA_CF, metadata_key.as_bytes(), &metadata_bytes)
		})
	}

	async fn get_tapestry<TID: TapestryId>(&self, tapestry_id: TID) -> Result<Option<u16>> {
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
	) -> Result<Option<TapestryFragment<T>>> {
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
	) -> Result<Option<M>> {
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

	async fn delete_tapestry<TID: TapestryId>(&self, tapestry_id: TID) -> Result<()> {
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
	) -> Result<()> {
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
	use serde::Deserialize;
	use std::{
		sync::Arc,
		time::{SystemTime, UNIX_EPOCH},
	};
	use tokio::{sync::Barrier, task};
	use uuid::Uuid;

	#[derive(Default, Debug, Clone, Serialize, Deserialize, PartialEq)]
	struct TestConfig;

	impl Config for TestConfig {
		const TOKEN_THRESHOLD_PERCENTILE: BoundedU8<0, 100> = BoundedU8::new(85).unwrap();
		const MINIMUM_RESPONSE_LENGTH: u64 = 100;

		type PromptModel = TestLlm;
		type SummaryModel = TestLlm;
		type Chest = RocksDbBackend;

		fn convert_prompt_tokens_to_summary_model_tokens(
			tokens: PromptModelTokens<Self>,
		) -> SummaryModelTokens<Self> {
			tokens
		}
	}

	impl From<ContextMessage<TestConfig>> for String {
		fn from(msg: ContextMessage<TestConfig>) -> Self {
			msg.content
		}
	}

	#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
	struct TestLlm;

	#[async_trait]
	impl Llm<TestConfig> for TestLlm {
		type Tokens = u16;
		type Request = String;
		type Response = String;
		type Parameters = ();

		fn max_context_length(&self) -> Self::Tokens {
			1000
		}

		fn name(&self) -> &'static str {
			"TestLlm"
		}

		fn alias(&self) -> &'static str {
			"TestLlm"
		}

		fn count_tokens(content: &str) -> Result<Self::Tokens> {
			Ok(content.len() as u16)
		}

		async fn prompt(
			&self,
			_is_summarizing: bool,
			_prompt_tokens: Self::Tokens,
			msgs: Vec<Self::Request>,
			_params: &Self::Parameters,
			_max_tokens: Self::Tokens,
		) -> Result<Self::Response> {
			Ok(msgs.join(" "))
		}

		fn compute_cost(&self, prompt_tokens: Self::Tokens, response_tokens: Self::Tokens) -> f64 {
			(prompt_tokens + response_tokens) as f64 * 0.001
		}
	}

	#[derive(Debug, Clone)]
	struct TestTapestryId(Uuid);

	impl TapestryId for TestTapestryId {
		fn base_key(&self) -> String {
			self.0.to_string()
		}
	}

	fn create_test_backend() -> RocksDbBackend {
		RocksDbBackend::new().unwrap()
	}

	fn create_test_tapestry_fragment() -> TapestryFragment<TestConfig> {
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

	#[tokio::test]
	async fn test_save_and_get_tapestry_fragment() {
		let backend = create_test_backend();
		let tapestry_id = TestTapestryId(Uuid::new_v4());
		let fragment = create_test_tapestry_fragment();

		let instance = backend
			.save_tapestry_fragment(&tapestry_id, fragment.clone(), true)
			.await
			.unwrap();
		assert_eq!(instance, 1);

		let retrieved_fragment: TapestryFragment<TestConfig> = backend
			.get_tapestry_fragment(tapestry_id, Some(instance))
			.await
			.unwrap()
			.unwrap();
		assert_eq!(retrieved_fragment.context_tokens, fragment.context_tokens);
		assert_eq!(retrieved_fragment.context_messages.len(), fragment.context_messages.len());
	}

	#[tokio::test]
	async fn test_save_and_get_tapestry_metadata() {
		let backend = create_test_backend();
		let tapestry_id = TestTapestryId(Uuid::new_v4());
		let metadata = "Test metadata".to_string();

		<RocksDbBackend as TapestryChestHandler<TestConfig>>::save_tapestry_metadata::<
			TestTapestryId,
			String,
		>(&backend, tapestry_id.clone(), metadata.clone())
		.await
		.unwrap();

		let retrieved_metadata: String =
			<RocksDbBackend as TapestryChestHandler<TestConfig>>::get_tapestry_metadata(
				&backend,
				tapestry_id,
			)
			.await
			.unwrap()
			.unwrap();
		assert_eq!(retrieved_metadata, metadata);
	}

	#[tokio::test]
	async fn test_get_tapestry() {
		let backend = create_test_backend();
		let tapestry_id = TestTapestryId(Uuid::new_v4());
		let fragment = create_test_tapestry_fragment();

		backend.save_tapestry_fragment(&tapestry_id, fragment, true).await.unwrap();

		let instance_count = <RocksDbBackend as TapestryChestHandler<TestConfig>>::get_tapestry(
			&backend,
			tapestry_id,
		)
		.await
		.unwrap()
		.unwrap();
		assert_eq!(instance_count, 1);
	}

	#[tokio::test]
	async fn test_delete_tapestry() {
		let backend = create_test_backend();
		let tapestry_id = TestTapestryId(Uuid::new_v4());
		let fragment = create_test_tapestry_fragment();

		backend.save_tapestry_fragment(&tapestry_id, fragment, true).await.unwrap();
		<RocksDbBackend as TapestryChestHandler<TestConfig>>::delete_tapestry(
			&backend,
			tapestry_id.clone(),
		)
		.await
		.unwrap();

		let instance_count = <RocksDbBackend as TapestryChestHandler<TestConfig>>::get_tapestry(
			&backend,
			tapestry_id,
		)
		.await
		.unwrap();
		assert_eq!(instance_count, None);
	}

	#[tokio::test]
	async fn test_delete_tapestry_fragment() {
		let backend = create_test_backend();
		let tapestry_id = TestTapestryId(Uuid::new_v4());
		let fragment = create_test_tapestry_fragment();

		backend
			.save_tapestry_fragment(&tapestry_id, fragment.clone(), true)
			.await
			.unwrap();
		backend.save_tapestry_fragment(&tapestry_id, fragment, true).await.unwrap();

		<RocksDbBackend as TapestryChestHandler<TestConfig>>::delete_tapestry_fragment(
			&backend,
			tapestry_id.clone(),
			Some(1),
		)
		.await
		.unwrap();

		let instance_count = <RocksDbBackend as TapestryChestHandler<TestConfig>>::get_tapestry(
			&backend,
			tapestry_id.clone(),
		)
		.await
		.unwrap()
		.unwrap();
		assert_eq!(instance_count, 2);

		let retrieved_fragment =
			<RocksDbBackend as TapestryChestHandler<TestConfig>>::get_tapestry_fragment(
				&backend,
				tapestry_id,
				Some(1),
			)
			.await
			.unwrap();
		assert_eq!(retrieved_fragment, None);
	}

	#[tokio::test]
	async fn test_concurrent_saves() {
		let backend = Arc::new(create_test_backend());
		let tapestry_id = Arc::new(TestTapestryId(Uuid::new_v4()));
		let barrier = Arc::new(Barrier::new(10));

		let mut handles = vec![];

		for i in 0..10 {
			let backend = backend.clone();
			let tapestry_id = tapestry_id.clone();
			let barrier = barrier.clone();

			handles.push(task::spawn(async move {
				let mut fragment = create_test_tapestry_fragment();
				fragment.context_messages[0].content = format!("Test message {}", i);

				barrier.wait().await;
				<RocksDbBackend as TapestryChestHandler<TestConfig>>::save_tapestry_fragment::<
					TestTapestryId,
				>(&backend, &tapestry_id, fragment, true)
				.await
				.unwrap()
			}));
		}

		let results: Vec<Option<u64>> = futures::future::join_all(handles)
			.await
			.into_iter()
			.filter_map(|r| r.ok())
			.map(Some)
			.collect();

		let instance_count = <RocksDbBackend as TapestryChestHandler<TestConfig>>::get_tapestry(
			&backend,
			tapestry_id.as_ref().clone(),
		)
		.await
		.unwrap()
		.unwrap();
		assert_eq!(instance_count as usize, results.len());
		assert_eq!(results.into_iter().max().unwrap(), Some(10));
	}

	#[tokio::test]
	async fn test_concurrent_reads_and_writes() {
		let backend = Arc::new(create_test_backend());
		let tapestry_id = Arc::new(TestTapestryId(Uuid::new_v4()));
		let barrier = Arc::new(Barrier::new(20));

		let mut handles = vec![];

		for i in 0..10 {
			let backend = backend.clone();
			let tapestry_id = tapestry_id.clone();
			let barrier = barrier.clone();

			handles.push(task::spawn(async move {
				let mut fragment = create_test_tapestry_fragment();
				fragment.context_messages[0].content = format!("Test message {}", i);

				barrier.wait().await;
				<RocksDbBackend as TapestryChestHandler<TestConfig>>::save_tapestry_fragment::<
					TestTapestryId,
				>(&backend, &tapestry_id, fragment, true)
				.await
				.unwrap()
			}));
		}

		for _ in 0..10 {
			let backend = backend.clone();
			let tapestry_id = tapestry_id.clone();
			let barrier = barrier.clone();

			handles.push(task::spawn(async move {
				barrier.wait().await;
				<RocksDbBackend as TapestryChestHandler<TestConfig>>::get_tapestry::<TestTapestryId>(
					&backend,
					tapestry_id.as_ref().clone(),
				)
				.await
				.unwrap().unwrap() as u64
			}));
		}

		let results: Vec<Option<u64>> = futures::future::join_all(handles)
			.await
			.into_iter()
			.filter_map(|r| r.ok())
			.map(Some)
			.collect();

		let write_results: Vec<u64> = results.iter().filter_map(|&r| r.map(|v| v as u64)).collect();
		let read_results: Vec<Option<u64>> = results
			.iter()
			.filter_map(|&r| if r.is_some() { None } else { Some(r) })
			.collect();

		assert_eq!(write_results.len(), 10);
		assert_eq!(read_results.len(), 10);
		assert!(write_results.into_iter().max().unwrap() <= 10);
	}

	#[tokio::test]
	async fn test_tapestry_fragment_versioning() {
		let backend = create_test_backend();
		let tapestry_id = TestTapestryId(Uuid::new_v4());
		let fragment1 = create_test_tapestry_fragment();
		let mut fragment2 = create_test_tapestry_fragment();
		fragment2.context_messages[0].content = "Updated message".to_string();

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

		let retrieved_fragment1 =
			<RocksDbBackend as TapestryChestHandler<TestConfig>>::get_tapestry_fragment(
				&backend,
				tapestry_id.clone(),
				Some(1),
			)
			.await
			.unwrap()
			.unwrap();
		let retrieved_fragment2 =
			<RocksDbBackend as TapestryChestHandler<TestConfig>>::get_tapestry_fragment(
				&backend,
				tapestry_id,
				Some(2),
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
	}

	#[tokio::test]
	async fn test_large_tapestry_fragment() {
		let backend = create_test_backend();
		let tapestry_id = TestTapestryId(Uuid::new_v4());
		let mut fragment = create_test_tapestry_fragment();

		// Add a large number of messages to the fragment
		for i in 0..1000 {
			fragment.context_messages.push(ContextMessage::new(
				WrapperRole::from("user"),
				format!("Large message {}", i),
				Some("user1".to_string()),
				SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs().to_string(),
			));
		}

		let instance = backend
			.save_tapestry_fragment(&tapestry_id, fragment.clone(), true)
			.await
			.unwrap();
		assert_eq!(instance, 1);

		let retrieved_fragment =
			<RocksDbBackend as TapestryChestHandler<TestConfig>>::get_tapestry_fragment(
				&backend,
				tapestry_id,
				Some(instance),
			)
			.await
			.unwrap()
			.unwrap();
		assert_eq!(retrieved_fragment.context_messages.len(), fragment.context_messages.len());
	}
}
