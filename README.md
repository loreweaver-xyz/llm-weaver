# Loreweaver Library

Loreweaver is a robust library designed to interact with models like ChatGPT, with an emphasis on managing, saving, and tracking message histories over the course of multiple interactions, ensuring a continuous and coherent user experience.

> Check out the official [Loreweaver Discord Bot](https://github.com/snowmead/loreweaver-discord-bot) purpously built based on this library!

## Use Cases

- **Text-Based RPGs:** Crafting coherent and persistently evolving narratives and interactions in text-based role-playing games.

- **Customer Support Chatbots:** Developing chatbots that remember past user interactions and provide personalized support.

- **Educational Virtual Tutors:** Implementing AI tutors that remember student interactions and tailor assistance accordingly.

- **Healthcare Virtual Assistants:** Creating healthcare assistants that provide follow-up advice and reminders based on past user health queries.

- **AI-Driven MMO NPC Interactions:** Enhancing MMO experiences by enabling NPCs to have contextually relevant interactions with players based on past encounters.

## Main Components

### `trait Config`

The `Config` trait is vital for declaring application-specific configuration parameters, which serve as the entry point for all applications leveraging the Loreweaver library. Implementing this trait enables the user to define specific configurations such as the GPT `Model` to use and specific `WeavingID` - which is used for identifying and managing weaving (story/dialogue) instances.

Example:

```rust
pub trait Config {
    /// Getter for GPT model to use.
    type Model: Get<Models>;
    /// Type alias encompassing a server id and a story id.
    type WeavingID: WeavingID;
}
```

### `trait Loom`

The `Loom` trait defines the core functionality of interacting with the loreweaver library via the `prompt`` method, which is core of the implementation.

Upon calling `Loom::prompt`, if successful, it will interact perform the following operations:

1. Get current story
2. Inject system instructions and other utility system messages
3. Prompt ChatGPT
4. Append the response to the story and save to Redis instance (ran by the application developer)
5. Return the response

Applications can implement additional features, such as charging users based on API usage, or proceeding with further interactions.

## Example Usage

```rust
use loreweaver::{Config, WeavingID, Models};

struct MyConfig;

impl Config for MyConfig {
    type Model = Models::GPT4;
    type WeavingID = MyWeavingID;
}

struct MyWeavingID(String);

impl WeavingID for MyWeavingID {
    fn base_key(&self) -> String {
        self.0.clone()
    }
}

#[tokio::main]
async fn main() {
    let weaving_id = MyWeavingID("my_weaving_1".into());
    let response = Loreweaver::<MyConfig>.prompt(
        "system instructions".into(),
        weaving_id,
        "hello loreweaver!".into(),
        1234,
        "username".into(),
        None,
    )
    .await
    .unwrap();
    println!("Received response: {}", response);
}
```

## Contribution

If you are passioniate about this project, please feel free to fork the repository and submit pull requests for enhancements, bug fixes, or additional features.

## License

Loreweaver is distributed under the MIT License, ensuring maximum freedom for using and sharing it in your projects.
