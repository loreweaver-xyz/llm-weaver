<div align="center">

[<img alt="github" src="https://img.shields.io/badge/maintenance%20status-actively%20developed-brightgreen?style=for-the-badge&labelColor=555555&logo=github" height="20">](https://github.com/snowmead/llm-weaver)
[<img alt="github" src="https://img.shields.io/badge/github-snowmead/llm_weaver-8da0cb?style=for-the-badge&labelColor=555555&logo=github" height="20">](https://github.com/snowmead/llm-weaver)
[<img alt="crates.io" src="https://img.shields.io/crates/v/llm-weaver.svg?style=for-the-badge&color=fc8d62&logo=rust" height="20">](https://crates.io/crates/llm-weaver)
[<img alt="docs.rs" src="https://img.shields.io/badge/docs.rs-llm_weaver-66c2a5?style=for-the-badge&labelColor=555555&logo=docs.rs" height="20">](https://docs.rs/llm-weaver)
[<img alt="build status" src="https://img.shields.io/github/actions/workflow/status/snowmead/llm-weaver/rust.yml?branch=main&style=for-the-badge" height="20">](https://github.com/snowmead/llm-weaver/actions?query=branch%3Amain)

</div>

# LLM Weaver

LLM Weaver is a flexible library designed to interact with any LLM, with an emphasis on managing long conversations exceeding the maximum token limit of a model, ensuring a continuous and coherent user experience.

## Implementation

This library is a rust implementation of [OpenAI's Tactic](https://platform.openai.com/docs/guides/gpt-best-practices/tactic-for-dialogue-applications-that-require-very-long-conversations-summarize-or-filter-previous-dialogue) for handling long conversations with a token context bound LLM.

Once a certain threshold of context tokens is reached, the library will summarize the entire conversation and begin a new conversation with the summarized context appended to the system instructions.

## Usage

Follow the [crate level documentation](https://docs.rs/llm-weaver/latest/llm_weaver/) for a detailed explanation of how to use the library.

## Contribution

If you are passioniate about this project, please feel free to fork the repository and submit pull requests for enhancements, bug fixes, or additional features.

## License

LLM Weaver is distributed under the MIT License, ensuring maximum freedom for using and sharing it in your projects.
