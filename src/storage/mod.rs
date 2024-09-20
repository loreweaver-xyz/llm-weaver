use async_trait::async_trait;
use serde::de::DeserializeOwned;
use std::fmt::{Debug, Display};

use crate::{Config, TapestryFragment, TapestryId};

pub(crate) mod common;
#[cfg(feature = "redis")]
pub mod redis;
#[cfg(feature = "rocksdb")]
pub mod rocksdb;

/// A storage handler trait designed for saving and retrieving fragments of a tapestry.
///
/// # Usage
///
/// Implementations of `TapestryChestHandler` should provide the storage and retrieval mechanisms
/// tailored to specific use-cases or storage backends, such as databases, file systems, or
/// in-memory stores.
#[async_trait]
pub trait TapestryChestHandler<T: Config> {
	/// Defines the error type returned by the handler methods.
	type Error: Display + Debug;

	fn new() -> Self;

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
		&self,
		tapestry_id: &TID,
		tapestry_fragment: TapestryFragment<T>,
		increment: bool,
	) -> crate::Result<u64>;
	/// Save tapestry metadata.
	///
	/// Based on application use cases, you can add aditional data for a given [`TapestryId`]
	async fn save_tapestry_metadata<TID: TapestryId, M: ToString + Debug + Clone + Send + Sync>(
		&self,
		tapestry_id: TID,
		metadata: M,
	) -> crate::Result<()>;
	/// Retrieves the number of instances of a tapestry.
	///
	/// Returns None if the tapestry does not exist.
	async fn get_tapestry<TID: TapestryId>(&self, tapestry_id: TID) -> crate::Result<Option<u16>>;
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
		&self,
		tapestry_id: TID,
		instance: Option<u64>,
	) -> crate::Result<Option<TapestryFragment<T>>>;
	/// Retrieves the last tapestry metadata, or a metadata at a specified instance.
	async fn get_tapestry_metadata<TID: TapestryId, M: DeserializeOwned>(
		&self,
		tapestry_id: TID,
	) -> crate::Result<Option<M>>;
	/// Deletes a tapestry and all its instances.
	async fn delete_tapestry<TID: TapestryId>(&self, tapestry_id: TID) -> crate::Result<()>;
	/// Deletes a tapestry fragment.
	async fn delete_tapestry_fragment<TID: TapestryId>(
		&self,
		tapestry_id: TID,
		instance: Option<u64>,
	) -> crate::Result<()>;
}
