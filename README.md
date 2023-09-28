# Loreweaver - An Interactive Storytelling Bot

Loreweaver is a text-based RPG (Role Playing Game) bot written in Rust, designed to weave intricate stories for a group of players over Discord or Web. By interacting with ChatGPT, Loreweaver crafts a rich narrative where players can join together on an adventure, playing as their own unique characters.

## Features

- **Engaging Story Crafting**: Develops unique story scenarios to maintain player interest and excitement.
- **Role-playing NPCs**: Provides dialogue and interaction for the non-player characters, tracking their roles within the story.
- **Rich Descriptions**: Describes environments, characters, and events in detail, immersing players in a detailed game world.
- **Twists and Challenges**: Stimulates players' problem-solving skills and imagination with unexpected twists, complex dilemmas, and unique challenges.
- **Player Choice Empowerment**: Allows players to engage in dialogue and respond to NPCs or enemies without speaking on their behalf.
- **Collaboration Encouragement**: Fosters deeper relationships with well-developed NPCs, enriching interactions and emotional connections.
- **Compelling Story Maintenance**: Enhances the depth and variety of the story experience with a wide range of emotional tones.

## Guidelines

- **Story Initiation**: Starts the story once at least one player has joined.
- **Markdown Formatting**: Uses markdown to format messages for emphasizing important information and readability.
- **Player Autonomy**: Never speaks on behalf of the players; allows them to speak for themselves.
- **Character Naming**: Always refers to the players by their character names within the context of the story.
- **Safe Environment**: Prohibits NSFW content, hate speech, and discussions of illegal activities.
- **Story Breadcrumbs**: Creates story breadcrumbs to guide players through the story and help them make decisions.

## Hardcore Mode

- **Player Mortality**: Players can be killed or die, and cannot return to the game (they can join later with a different character).
- **PVP Combat**: Enjoyable and fair PVP combat; no consent needed.
- **Unfairness**: Unfairness is allowed in the game dynamics.
- **PVP Consent**: Does not ask the players if they wish to avoid PVP between each other.

## Story Size Criteria

- **Small Story**:
  - Single, self-contained clear quest/objective.
  - Limited NPCs, locations, and challenges for a concise story.
  - Straightforward story completed within a short time frame.
  - Focus on player interactions and quick decision-making.

- **Medium Story**:
  - Multi-layered story with interconnected quests/objectives.
  - Moderate number of NPCs, locations, and challenges for depth and variety.
  - Complex story allowing for character growth and evolving storylines.
  - Encourages deeper collaboration and decision-making among players.

- **Large Story**:
  - Expansive game world with multiple story arcs and diverse quests/objectives.
  - Vast array of NPCs, locations, and challenges for a rich and varied experience.
  - Intricate story with interconnected storylines and character development opportunities.
  - Extensive collaboration, planning, and decision-making among players to navigate story complexity.

## Getting Started

1. Clone the repository:
```bash
git clone https://github.com/your-username/loreweaver.git
```
2. Navigate to the project directory:
```bash
cd loreweaver
```
3. Build the project:
```bash
cargo build --release
```
4. Run Loreweaver:
```bash
./target/release/loreweaver gpt4 --run-as discord
```
