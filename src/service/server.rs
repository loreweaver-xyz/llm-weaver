use crate::{
    loreweaver::{Config, Loom},
    GPTModel, Server,
};

pub struct ApiService;

impl Config for ApiService {
    type Model = GPTModel;
    type MaxTokenResponse = u64;
}

impl<T: Loom> Server<T> for ApiService {
    async fn serve() {
        let prompt = T::prompt((100, 101)).await;
        println!("Loreweaver: {}", prompt);
    }
}
