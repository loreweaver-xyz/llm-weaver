use async_openai::types::Role;
use serde::{Deserialize, Serialize};

/// Base type for all configuration parameters.
pub type F32 = f32;

pub const SYSTEM_ROLE: &str = "system";
pub const ASSISTANT_ROLE: &str = "assistant";
const USER_ROLE: &str = "user";
const FUNCTION_ROLE: &str = "function";

/// Wrapped [`Role`] for custom implementations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WrapperRole {
	Role(Role),
}

impl Default for WrapperRole {
	fn default() -> Self {
		Self::Role(Role::User)
	}
}

impl From<Role> for WrapperRole {
	fn from(role: Role) -> Self {
		WrapperRole::Role(role)
	}
}

impl From<WrapperRole> for Role {
	fn from(role: WrapperRole) -> Self {
		match role {
			WrapperRole::Role(role) => role,
		}
	}
}

impl From<String> for WrapperRole {
	fn from(role: String) -> Self {
		match role.as_str() {
			SYSTEM_ROLE => Self::Role(Role::System),
			ASSISTANT_ROLE => Self::Role(Role::Assistant),
			USER_ROLE => Self::Role(Role::User),
			FUNCTION_ROLE => Self::Role(Role::Function),
			_ => panic!(
				"Invalid role: {} \n Valid roles: {} | {} | {} | {}",
				role, SYSTEM_ROLE, ASSISTANT_ROLE, USER_ROLE, FUNCTION_ROLE
			),
		}
	}
}

impl From<WrapperRole> for String {
	fn from(role: WrapperRole) -> Self {
		match role {
			WrapperRole::Role(Role::System) => SYSTEM_ROLE.to_string(),
			WrapperRole::Role(Role::Assistant) => ASSISTANT_ROLE.to_string(),
			WrapperRole::Role(Role::User) => USER_ROLE.to_string(),
			WrapperRole::Role(Role::Function) => FUNCTION_ROLE.to_string(),
		}
	}
}
