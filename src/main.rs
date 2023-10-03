#![feature(async_fn_in_trait)]
#![feature(type_alias_impl_trait)]
#![feature(async_closure)]

use std::{error::Error, sync::OnceLock};

use clap::{arg, Parser};
use discord::server::DiscordBot;
use loreweaver::{models::Models, Get};
use service::server::ApiService;
use tracing::{info, Level};
use tracing_subscriber::fmt;

mod discord;
mod loreweaver;
mod service;
mod storage;

#[derive(Parser, Debug)]
struct Args {
	/// GPT model to use.
	#[arg(value_enum)]
	model: Models,
	/// Determines whether to run the application as the [`DiscordBot`] or the [`ApiService`].
	#[arg(long)]
	run_as: String,
	/// Log level
	#[arg(long, default_value = "info")]
	log_level: Level,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
	let args = Args::parse();

	let subscriber = fmt::Subscriber::builder().with_max_level(args.log_level).finish();
	tracing::subscriber::set_global_default(subscriber)?;

	info!(task = "tracing_setup", result = "success", "tracing successfully set up",);

	dotenv::dotenv().ok();

	info!(task = "dotenv_setup", result = "success", "dotenv loaded successfully");

	match args.run_as.as_str() {
		"discord" => <DiscordBot as Server>::serve().await,
		"api" => <ApiService as Server>::serve().await,
		_ => panic!("Invalid argument"),
	};

	Ok(())
}

/// A trait that defines the methods that a server should implement.
///
/// The generic type `T` ensures that the server is using [`Loom`] to drive the core functionality
/// of the server.
///
/// It also gives flexibility for a server to override the default implementation of [`Loom`]
trait Server {
	/// Starts the server.
	async fn serve();
}

const GPT_MODEL: OnceLock<Models> = OnceLock::new();

/// Getter implementation for determining the GPT model to use.
pub struct GPTModel;
impl Get<Models> for GPTModel {
	fn get() -> Models {
		*GPT_MODEL.get_or_init(|| Args::parse().model)
	}
}
