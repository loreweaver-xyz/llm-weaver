# Loreweaver Library

Loreweaver is a robust library designed to interact with models like ChatGPT, with an emphasis on managing, saving, and tracking message histories over the course of multiple interactions, ensuring a continuous and coherent user experience in long conversations.

> Check out the official [Loreweaver Discord Bot](https://github.com/snowmead/loreweaver-discord-bot) purpously built based on this library!

## Implementation

This library is a rust implementation of [OpenAI's Tactic](https://platform.openai.com/docs/guides/gpt-best-practices/tactic-for-dialogue-applications-that-require-very-long-conversations-summarize-or-filter-previous-dialogue) for handling long conversations with ChatGPT that span beyond a GPT model's maximum context token limit.

Once a certain threshold of context tokens is reached, the library will summarize the entire conversation and begin a new conversation with the summarized context appended to the system instructions.

The library is designed to be as flexible as possible, allowing you to easily integrate it into your own projects.

## Use Cases

- **Text-Based RPGs:** Crafting coherent and persistently evolving narratives and interactions in text-based role-playing games.

- **Customer Support Chatbots:** Developing chatbots that remember past user interactions and provide personalized support.

- **Educational Virtual Tutors:** Implementing AI tutors that remember student interactions and tailor assistance accordingly.

- **Healthcare Virtual Assistants:** Creating healthcare assistants that provide follow-up advice and reminders based on past user health queries.

- **AI-Driven MMO NPC Interactions:** Enhancing MMO experiences by enabling NPCs to have contextually relevant interactions with players based on past encounters.

## Contribution

If you are passioniate about this project, please feel free to fork the repository and submit pull requests for enhancements, bug fixes, or additional features.

## License

Loreweaver is distributed under the MIT License, ensuring maximum freedom for using and sharing it in your projects.
