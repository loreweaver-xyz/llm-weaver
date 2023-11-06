/// The following diagram shows a very high level slimmed down overview of the
/// architecture of the library and how an application might use it.
///
/// Only the traits [`Loom`](crate::Loom) and [`Config`](crate::Config) are expanded to show some of
/// their main associated types.
#[cfg_attr(doc, aquamarine::aquamarine)]
/// ```mermaid
/// graph TB
///     subgraph Chat Application
///         App
///         chat_gpt[Chat GPT]
///         bard[Bard]
///     end
///     App-. impl .- Loom
///     App-. impl .- Config
///     chat_gpt[Chat GPT]-. impl .- llm
///     bard[Bard]-. impl .- llm
///     subgraph LLM Weaver
///         llm>Llm]
///         subgraph Config
///             prompt_model[PromptModel]-- prompt --> chat_gpt
///             summary_model[SummaryModel]-- prompt --> bard
///             tapestry_chest_type[Chest]
///         end        
///         subgraph Loom
///             weave-- save prompt and response --> tapestry_chest_type
///             weave-- generate summary --> summary_model
///             weave-- generate response --> prompt_model
///         end
///         tapestry_chest_handler>TapestryChestHandler]
///         tapestry_chest[TapestryChest]-. default impl .- tapestry_chest_handler
///         tapestry_chest_type --> tapestry_chest
///         tapestry_chest --> redis
///         redis[Redis]
///     end
/// ```
///
/// The application must implement the [`Loom`](crate::Loom) and [`Config`](crate::Config) traits in
/// order to utilize the library. This includes but is not limited to providing the types that
/// implement the [`Llm`](crate::Llm) trait which defines the LLMs which will be used to
/// prompt and generate summaries.
///
/// The [`Config`](crate::Config) trait also requires the application to supply an implementation
/// for [`Chest`](crate::Config::Chest) which is responsible for storing and retrieving the
/// [`TapestryFragment`](crate::TapestryFragment)s, but is not required since llm_weaver provides a
/// default implementation that uses Redis as the storage backend.
pub struct Diagram;
