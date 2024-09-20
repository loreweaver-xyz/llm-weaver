// common.rs

/// The key used to store the number of instances of a tapestry.
pub const INSTANCE_COUNT_KEY: &str = "INSTANCE_COUNT";

/// Represents the base functionality required for storage backends.
pub trait StorageBackend {
	type Error: std::error::Error + Send + Sync + 'static;

	/// Retrieves the instance count for a given base key.
	fn get_instance_count(&self, base_key: &str) -> Result<Option<u64>, Self::Error>;

	/// Sets the instance count for a given base key.
	fn set_instance_count(&self, base_key: &str, count: u64) -> Result<(), Self::Error>;

	/// Increments the instance count for a given base key.
	fn increment_instance_count(&self, base_key: &str) -> Result<u64, Self::Error>;

	/// Checks if a key exists.
	fn key_exists(&self, key: &str) -> Result<bool, Self::Error>;
}
