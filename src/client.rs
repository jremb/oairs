//! This module contains the Client struct and a few private helpers for building the client and
//! sending a request. The Client struct is the main point of entry for interacting with the
//! endpoints of the OpenAI API.

use reqwest::{header, multipart::Part};

use crate::{
    completions::{ChatBuilder, CompletionBuilder, Messages},
    edits::EditBuilder,
    embeddings::EmbeddingBuilder,
    fine_tunes::{FineTunesBuilder, ListEventsBuilder},
    images::{ImageBuilder, ImageEdit, ImageGen, ImageVariation},
};

use super::*;

// ASIDE:
//
// I've wrestled back and forth between various designs. Initially I attempted to make the design as
// simple to use as possible (by my estimation) and stuck close to the interface of a Python library
// for accessing Twitter's API (Tweepy) that I'm somewhat familiar with. The OpenAI Python library
// diverges significantly from this and as I continued to flesh out the library I decided to rewrite
// it, sticking close to the basic *form* of the OpenAI Python library, translating  it into Rust as
// best I could. At several points, this resulted in cumbersome syntax or (IMO) odd naming (e.g.,
// `create` for uploading a file, instead of `upload`). In regards to the former, this
// meant we would do `client.chat_completion.create(...).send().await` where the Python library is
// `openai.ChatCompletion.create(...)`. Accessing the `https://api.openai.com/v1/models` endpoint
// (to list models) becomes `client.model().list().get().await` where the Python library is
// `openai.Model.list()`.
//
// In this design, the `Client` was really nothing more than a facade for various structs matching
// their resource. In the end, I couldn't convince myself that this wasn't the sort of "hobgoblin of
// little minds" (a foolish consistency) and went back to my initial intuition, rewriting a second time.
// (Furthermore, it's not obvious to me that opting for consistency, which I assume must be the
// explanation for doing something like `openai.File.create(...)`, outweighs the benefits of a somewhat
// *in*consistent but more natural syntax (`client.upload_file(...)`).)
//
// In the current design, the `Client` does its own work for endpoints that take no parameters or only
// take a path parameter. Where a request requires a body and there are optional parameters, the client
// returns a builder struct that handles storing the parameters and sending the request. The one
// exception to this is the `https://api.openai.com/v1/files` endpoint for creating (uploading) a file.
// This is the only `POST` method that takes a form and since there are only two fields, it seemed
// another foolish consistency to give it its own struct. Therefore, doing `client.upload_file(...)`
// will return a `Client<Sendable>`.

/// The main point of entry for interacting with the endpoints of the OpenAI API. Parameters
/// **required** by the API are always passed in as arguments to the relevant method. Optional
/// parameters, if any, are chained. Executing the request is always done by awaiting the
/// `send()` method.
///
/// # Example
/// ```rust,no_run
/// let key = std::env::var("OPENAI_API_KEY").unwrap();
/// let client = Client::new(&key);
///
/// // The `https://api.openai.com/v1/models` endpoint takes no parameters, so we simply
/// // await the send() method. The response is a `Result<reqwest::Response, OairsError>`.
/// // If `Ok`, the response should be deserializable into a `ModelsList` struct.
/// let models = match client.list_models().send().await {
///     Ok(response) => response.json::<ModelsList>().await.unwrap();,
///     Err(e) => panic!("Error: {:#?}", e),
/// };
///
/// // As with any struct deserializable from a response, it can be saved with `save_json()`.
/// // If the ".json" extension is not provided, it will be added.
/// match models.save_json("list_models") {
///         Ok(_) => (),
///         Err(e) => panic!("Error saving: {e}"),
///     }
/// }
/// ```
///
/// # Example
/// The API's required parameters are required parameters in the method call. If there are optional
/// parameters, we can chain them.
/// ```rust,no_run
/// let key = std::env::var("OPENAI_API_KEY").unwrap();
/// let client = Client::new(&key);
///
/// use Msg::System;
/// use Msg::User;
/// let system = System("You are a helpful assistant.");
/// let user = User("This is a test.");
///
/// // default is ChatModel::GptTurbo, or the base chat model for Gpt3.5
/// // Not every model enum has a default available.
/// let model = ChatModel::default();
/// let mut messages = Messages::default();
/// messages.push(system);
/// messages.push(user);
///
/// let completion = client
///     .chat_completion(model, &messages)
///     .max_tokens(200)
///     .send()
///     .await;
/// {
///     Ok(response) => response.json::<ChatCompletion>().await.unwrap();,
///     Err(e) => panic!("{e}"),
/// };
///
/// match completion.save("chat_completion") {
///     Ok(_) => (),
///     Err(e) => panic!("Error saving: {e}"),
/// }
/// ```
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Client<State = Unkeyed> {
    #[serde(skip)]
    key: String,

    #[serde(skip)]
    url: Option<String>,

    // These fields only used for the form of uploading a file.
    #[serde(skip_serializing_if = "Option::is_none", alias = "file")]
    upload_filename: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", alias = "purpose")]
    file_purpose: Option<Purpose>,

    #[serde(skip)]
    state: PhantomData<State>,
}

impl Client<Unkeyed> {
    pub fn new<K: Into<String> + std::fmt::Debug>(key: K) -> Client<Keyed> {
        Client {
            key: format!("Bearer {}", key.into()),
            state: PhantomData::<Keyed>,
            ..Default::default()
        }
    }
}

impl Client<Keyed> {
    /// "Immediately cancel a fine-tune job."
    /// - [OpenAI API docs](https://platform.openai.com/docs/api-reference/fine-tunes/cancel)
    ///
    /// # Arguments
    /// * `fine_tune_id` - The ID of the fine-tune job to cancel. Will have a format similar to `ft-xxxxx...`
    ///
    /// # Returns
    /// `Client<Cancel>` that can execute the request by awaiting `send()`.
    ///
    /// `Result<reqwest::Response, OairsError>`. A Successful response can be deserialized into a
    /// [`FineTuneInfo`](crate::fine_tunes::response::FineTuneInfo).
    ///
    /// # Example
    /// ```rust,no_run
    /// let key = std::env::var("OPENAI_API_KEY").unwrap();
    /// let client = Client::new(key);
    ///
    /// let fine_tune_id = "ft-2goTxssxraEdN4YI7oJUb8It";
    /// let response = match client.cancel_fine_tune(&fine_tune_id).send().await {
    ///     Ok(res) => res.json::<FineTuneInfo>().await.unwrap(),
    ///     Err(e) => panic!("{e}"),
    /// };
    /// // ...
    /// ```
    pub fn cancel_fine_tune(self, fine_tune_id: &str) -> Client<Cancel> {
        Client {
            key: self.key,
            url: Some(cancel_ft_url(fine_tune_id)),
            ..Default::default()
        }
    }

    /// For creating a chat completion with various GPT chat models (including GPT 4).
    ///
    /// # Arguments
    /// * `model` - The [`ChatModel`] variant to use for the chat completion.
    /// * `messages` - A [`Messages`] struct containing the messages to use for the chat completion.
    ///
    /// # Optional Arguments
    /// The following optional arguments are available via chaining:
    /// * `temperature` - The temperature of the model. Must be between `0.0` and `2.0`. Default is `1.0`.
    /// * `top_p` - Alternative to `temperature`. Must be between `0.0` and `1.0`.
    /// * `n` - The number of samples to return. Default is `1`.
    /// * `stream` - Whether to stream the response. If `true`, the response will be a stream of data server-sent
    /// events. Each event will be a complete JSON object prefixed with `data: ` and suffixed with `[DONE]` when done.
    /// The JSON object can be represented as a [`ChatCompletionChunk`](crate::completions::response::ChatCompletionChunk).
    /// * `stop` - A list of no more than `4` strings that will cause the chat to stop.
    /// * `max_tokens` - The maximum number of tokens to return. Documents say defaults to "Defaults to `inf`", but
    /// also that "By Default, the number of tokens the model can return will be (`4096` - prompt tokens)." 🤷
    /// * `presence_penalty` - The presence penalty. Must be between `-2.0` and `2.0`.
    /// * `frequency_penalty` - The frequency penalty. Must be between `-2.0` and `2.0`.
    /// * `logit_bias` - A hashmap of a token (as a string) and a bias (`f32`). The bias should be a number between
    /// `-100` and `100`. (E.g., `{"9703", -100.0}`).
    /// * `user` - The user ID to use for the chat completion. For moderation purposes.
    ///
    /// # Example
    /// In its most basic form, a chat completion can be created with the following:
    /// ```rust,no_run
    /// // ...
    /// let model = ChatModel::Gpt40314;
    /// let mut msgs = Messages::new();
    ///
    /// use Msg::System;
    /// use Msg::User;
    /// let system = System("You are a helpful assistant.");
    /// let user = User("Hello, world.");
    ///
    /// msgs.push(system);
    /// msgs.push(user);
    ///
    /// client.chat_completion(model, msgs)
    ///     .send()
    ///     .await
    /// // ...
    ///
    /// ```
    /// However, to carry on a "conversation", you'll want to use some sort of loop to send the previous
    /// chat messages as context. The model won't actually know about your previous request to the endpoint.
    ///
    /// Therefore, you may want do something along the lines of the following example to keep track of the
    /// messages and simplify your loop.
    ///
    /// # Example
    /// ```rust,no_run
    /// // Assume methods on some struct with Client<Keyed> and a Messages field
    /// pub async fn chat(&self, input: &Messages) -> Result<reqwest::Response, OairsError> {
    ///    let model = ChatModel::Gpt40314;
    ///    let response = self.client
    ///        .chat_completion(model, input)
    ///        .max_tokens(200)
    ///        .send()
    ///        .await;
    ///
    ///    response
    /// }
    ///
    /// pub async fn validate(&self, res: Result<reqwest::Response, OairsError>) -> Result<ChatCompletion, OairsError> {
    ///     match res {
    ///        Ok(response) => response.json::<ChatCompletion>().await.unwrap(),
    ///        Err(e) => e,
    ///     }
    /// }
    ///
    /// pub async fn update(&mut self, completion: ChatCompletion) {
    ///     self.messages.push(completion.response_message());
    /// }
    /// ```
    pub fn chat_completion(&self, model: ChatModel, msgs: &Messages) -> ChatBuilder<Sendable> {
        ChatBuilder::create(&self.key, model, msgs)
    }

    /// "Given a prompt, the model will return one or more predicted completions, and can also return
    /// the probabilities of alternative tokens at each position." -
    /// [OpenAI API Docs](https://platform.openai.com/docs/api-reference/completions)
    ///
    /// # Arguments
    /// * `model` - The [`CompletionModel`] variant to use for the completion.
    ///
    /// # Optional Arguments
    /// The following optional arguments are available via chaining:
    /// * `prompt` - The prompt to use for the completion. If not set, API will default to <|endoftext|>
    /// (special token).
    /// * `suffix` - The ending text you want the model to try to arrive at.
    /// See [`suffix`](CompletionBuilder<Sendable>::suffix) for more information.
    /// * `max_tokens` - The maximum number of tokens to return.
    /// * `temperature` - The temperature of the model. Must be between `0.0` and `2.0`. If not set, API
    /// will default to `1.0`.
    /// * `top_p` - Alternative to `temperature`. Must be between `0.0` and `1.0`.
    /// * `n` - The number of samples to return. Default is `1`.
    /// * `stream` - Whether to stream the response. If `true`, the response will be a stream of data
    /// server-sent events. Each event will be a complete JSON object prefixed with `data: ` and suffixed
    /// with `[DONE]` when done.
    /// * `stop` - A list of no more than `4` strings that will cause the chat to stop.
    /// * `presence_penalty` - The presence penalty. Must be between `-2.0` and `2.0`.
    /// * `frequency_penalty` - The frequency penalty. Must be between `-2.0` and `2.0`.
    /// * `logit_bias` - A hashmap of a token (as a string) and a bias (`f32`). The bias should be a number
    /// between `-100` and `100`. (E.g., `{"9703", -100.0}`).
    /// * `echo` - Whether to include the prompt in the response. If `true`, the response will include the
    /// prompt.
    /// * `best_of` - The number of best completions to return. Default is `1`.
    /// * `logprobs` - The number of log probabilities to return. Default is `0`.
    /// * `user` - The user ID to use for the completion. For moderation purposes.
    ///
    /// # Example
    /// ```rust,no_run
    /// let model = CompletionModel::Davinci;
    /// let completion = client.completion(model)
    ///     .prompt("This is a test.")
    ///     .max_tokens(5)
    ///     .send()
    ///     .await
    /// {
    ///      Ok(res) => res.json::<Completion>().await.unwrap(),
    ///      Err(e) => panic!("Error: {}", e),
    /// }
    /// ```
    pub fn completion(&self, model: CompletionModel) -> CompletionBuilder<Sendable> {
        CompletionBuilder::create(&self.key, model)
    }

    /// "Given a prompt and an instruction, the model will return an edited version of the prompt."
    /// - [OpenAI API Docs](https://platform.openai.com/docs/api-reference/edits)
    ///
    /// # Arguments
    /// * `model` - The [`EditModel`] variant to use for the edit.
    /// * `instruction` - The instruction to use for the edit. (E.g., "Fix the spelling mistakes.")
    ///
    /// # Optional Arguments
    /// The following optional arguments are available via chaining:
    /// * `input` - The input to use for the edit. If not set, API will default to empty string.
    /// * `temperature` - The temperature of the model. Must be between `0.0` and `2.0`. If not set, API will
    /// default to `1.0`.
    /// * `top_p` - Alternative to `temperature`. Must be between `0.0` and `1.0`.
    /// * `n` - The number of samples to return. Default is `1`.
    ///
    /// # Returns
    /// `EditBuilder<Sendable>` that can be used to set optional parameters and execute the request by awaiting
    /// `send()`.
    ///
    /// Awaiting `send()` will return `Result<reqwest::Response, OairsError>`. The `Response` can be deserialized
    /// into an [`Edit`].
    ///
    /// # Example
    /// ```rust,no_run
    /// // ...
    /// let model = EditModel::TextDavinciEdit001;
    /// let instruction = "Fix the spelling mistakes.";
    /// let s = "Tis is a test.";
    ///
    /// let edit = client.create_edit(model, instruction)
    ///     .input(s);
    ///     .send()
    ///     .await
    /// {
    ///     Ok(res) => res.json::<Edit>().await.unwrap(),
    ///     Err(e) => panic!("{e}"),
    /// }
    ///
    /// match edit.save_json("some_file") {
    ///    Ok(_) => println!("Saved edit to file."),
    ///    Err(e) => eprintln!("Error saving edit to file: {}", e),
    /// }
    /// ```
    pub fn create_edit<I: Into<String>>(
        &self,
        model: EditModel,
        instruction: I,
    ) -> EditBuilder<Sendable> {
        EditBuilder::create(&self.key, model, instruction.into())
    }

    /// "Creates an embedding vector representing the input text." -
    /// [OpenAI API Docs](https://platform.openai.com/docs/api-reference/embeddings/create)
    ///
    /// # Arguments
    /// * `model` - The [`EmbeddingModel`] variant to use for the embedding.
    /// * `inputs` - A a slice (or `&Vec<String>`) to use for the embedding. Each `String` will receive its
    /// own embedding and, therefore, you can think of each string as what is sometimes called a "document"
    /// in NLP. The slice can be thought of as a corpus (though it need not be your actual corpus). You should
    /// consider some preprocessing of the inputs prior to creating the embeddings.
    ///
    /// # Optional Arguments
    /// * `user` - "A unique identifier representing your end-user, which can help OpenAI to monitor and
    /// detect abuse." -
    /// [OpenAI API Docs](https://platform.openai.com/docs/api-reference/embeddings/create#embeddings/create-user)
    ///
    /// # Returns
    /// `EmbeddingBuilder<Sendable>` that can be used to set the optional `user` parameter and execute the
    /// request by awaiting `send()`.
    ///
    /// Awaiting `send()` will return `Result<reqwest::Response, OairsError>`. The `Response` can be
    /// deserialized into an [`Embedding`].
    ///
    /// As with all of the structs used for deserializing a
    /// [`Response`](https://docs.rs/reqwest/0.11.14/reqwest/struct.Response.html), an [`Embedding`] can be
    /// saved to a JSON file via `save_json()`. Additionally, an [`Embedding`] can be saved along with the
    /// inputs that were used to create it via `save_with_inputs()`. It is important that the inputs be in
    /// the same order that the were passed into `create_embeddings()`, otherwise the embeddings will not be
    /// correctly associated with the inputs.
    ///
    /// The [`Embedding`] struct contains some additional convenience methods for working with the
    /// embeddings. See the [`Embedding`] documentation for more information.
    ///
    /// # Example
    /// ```rust,no_run
    /// // Default is `EmbeddingModel::TextEmbeddingAda002`.
    /// // This should be the preferred model unless you have a specific reason to use another.
    /// let model = EmbeddingModel::default();
    /// let inputs = vec!["This is a test.", "This is another test."];
    ///
    /// // To make use of `save_with_inputs()`, the Embedding struct must be mutable.
    /// let mut embedding_response = match client.create_embeddings(model, &inputs).send().await {
    ///   Ok(r) => r.json::<Embedding>().await.unwrap(),
    ///   Err(e) => {
    ///       // Any OairsError can also be saved to a JSON file.
    ///       let filename = "embedding_error";
    ///       e.save_json(filename).unwrap();
    ///       panic!("{e}")
    ///   }
    /// };
    ///
    /// match embedding_response.save_with_inputs("path/to/json/file", inputs) {
    ///     Ok(_) => (),
    ///     Err(e) => panic!("{e}"),
    /// }
    pub fn create_embeddings<T>(
        &self,
        model: EmbeddingModel,
        inputs: &[T],
    ) -> EmbeddingBuilder<Sendable>
    where
        T: Into<String> + std::fmt::Display,
    {
        EmbeddingBuilder::new(&self.key, model, inputs)
    }

    /// Fine-tune a model based on a training file.
    ///
    /// # Arguments:
    /// * `training_file_id`: the id of a file that has *already been uploaded* to OpenAI's servers (cf. the
    /// files endpoint).
    ///
    /// # Returns:
    /// `FineTunesBuilder<'a, Create>` that can be used to set optional parameters and execute the request by
    /// awaiting `send()`.
    ///
    /// Awaiting `send()` will return `Result<reqwest::Response, OairsError>`. A successful response can be
    /// deserialized into a [`FineTuneInfo`].
    ///
    /// # Optional parameters:
    /// The following parameters are optional and can be set with the corresponding methods:
    /// * `validation_file`
    /// * `model`
    /// * `n_epochs`
    /// * `batch_size`
    /// * `learning_rate_multiplier`
    /// * `prompt_loss_weight`
    /// * `compute_classification_metrics`
    /// * `classification_n_classes`
    /// * `classification_positive_class`
    /// * `classification_betas`
    /// * `suffix`
    ///
    /// # Example:
    /// ```rust,no_run
    /// let key = std::env::var("OPENAI_API_KEY").unwrap();
    /// let client = Client::new(key);
    ///
    /// // Id of a file that has been previously uploaded
    /// let training_file_id = "file-ERwBFft4Iywvy7KeWcz0Mscs";
    /// let response = match client.create_fine_tune(training_file_id)
    ///     .n_epochs(1)
    ///     .post()
    ///     .await {
    ///         Ok(r) => r.json::<FineTuneInfo>().await.unwrap(),
    ///         Err(e) => panic!("{e}"),
    ///     };
    ///
    /// let ft_id = response.id;
    /// let filename = format!("ft_results/{}", ft_id);
    /// response.save_json(&filename).unwrap();
    /// ```
    pub fn create_fine_tune<'a>(
        &self,
        training_file_id: &'a str,
    ) -> FineTunesBuilder<'a, Sendable> {
        FineTunesBuilder::create(&self.key, training_file_id)
    }

    /// "Classifies if text violates OpenAI's Content Policy." -
    /// [OpenAI Docs](https://platform.openai.com/docs/api-reference/moderations/create)
    ///
    /// # Arguments
    /// * `input` - A `&str` or `String` that will be classified.
    ///
    /// # Optional parameters
    /// The following parameters are optional and can be set with the corresponding methods:
    /// * `model` - The [`ModerationModel`] to use for classification. Defaults to
    /// [`ModerationModel::TextModerationLatest`].
    ///
    /// # Returns
    /// `ModerationBuilder<Sendable>` that can be used to set the optional parameter, `model`, and
    /// execute the request by awaiting `send()`.
    ///
    /// Awaiting `send()` will return `Result<reqwest::Response, OairsError>`. A successful response
    /// can be deserialized into a [`ModerationResult`](self::response::ModerationResult).
    ///
    /// # Example
    /// ```rust,no_run
    /// let mut moderation_result = match client
    ///     .create_moderation("This is a test.")
    ///     .model(model)
    ///     .send()
    ///     .await
    /// {
    ///     Ok(res) => res.json::<ModerationResult>().await.unwrap(),
    ///     Err(e) => panic!("{e}"),
    /// };
    /// ```
    pub fn create_moderation<S: Into<String>>(&self, input: S) -> ModerationBuilder<Sendable> {
        let inputs = vec![input.into()];
        ModerationBuilder::create(&self.key, inputs)
    }

    /// Same as [`create_moderation`] but takes a vector of strings.
    ///
    /// # Arguments
    /// * `inputs` - A vector of strings to be classified.
    ///
    /// # Optional parameters
    /// The following parameters are optional and can be set with the corresponding methods:
    /// * `model` - The [`ModerationModel`] to use for classification. Defaults to
    /// [`ModerationModel::TextModerationLatest`].
    ///
    /// # Returns
    /// `ModerationBuilder<Sendable>` that can be used to set the optional parameter, `model`,
    /// and execute the request by awaiting `send()`.
    ///
    /// Awaiting `send()` will return `Result<reqwest::Response, OairsError>`. A successful
    /// response can be deserialized into a [`ModerationResult`](self::response::ModerationResult).
    ///
    /// # Example
    /// ```rust,no_run
    /// let prompts = vec!["This is a test.".to_string(), "this is another test.".to_string()];
    /// let mut moderation_result = match client
    ///     .create_moderations(prompts)
    ///     .model(model)
    ///     .send()
    ///     .await
    /// {
    ///     Ok(res) => res.json::<ModerationResult>().await.unwrap(),
    ///     Err(e) => panic!("{e}"),
    /// };
    /// ```
    pub fn create_moderations(&self, inputs: Vec<String>) -> ModerationBuilder<Sendable> {
        ModerationBuilder::create(&self.key, inputs)
    }

    /// "Creates an image given a prompt." -
    /// [OpenAI Docs](https://platform.openai.com/docs/api-reference/images/create)
    ///
    /// # Arguments
    /// * `prompt` - The prompt to use for the image.
    ///
    /// # Optional parameters
    /// The following parameters are optional and can be set with the corresponding methods:
    /// * [`n`] - The number of images to generate. Must be between `1` and `10`. Defaults to `1`. Panics
    /// if `n` is not in this range.
    /// * [`size`](images::ImageSize) -
    /// The [`ImageSize`](images::ImageSize) of the image to generate. Size vairants are
    /// [`ImageSize::Small`](images::ImageSize) (`256x256`),
    /// [`ImageSize::Medium`](images::ImageSize) (`512x512`),
    /// and [`ImageSize::Large`](images::ImageSize) (`1024x1024`). API defaults to
    /// [`ImageSize::Large`](images::ImageSize).
    /// * [`response_format`](images::response::ResponseFormat) - The
    /// [`ResponseFormat`](images::response::ResponseFormat) of the image to generate. Format variants are
    /// [`ResponseFormat::Url`](images::response::ResponseFormat) and
    /// [`ResponseFormat::Base64`](images::response::ResponseFormat). API defaults to
    /// [`ResponseFormat::Url`](images::response::ResponseFormat).
    /// * `user` - The unique id for a user, for detecting abuse.
    ///
    /// # Returns
    /// `ImageBuilder<ImageGen>` that can be used to set optional parameters and execute the request by
    /// awaiting `send()`.
    ///
    /// Awaiting `send()` will return `Result<reqwest::Response, OairsError>`. A successful response can
    /// be deserialized into an [`Image`](images::response::Image) that can be saved to a file using the `save_json()` method.
    ///
    /// # Example
    /// ```rust,no_run
    /// let key = get_key();
    /// let client = Client::new(key);
    ///
    /// let prompt = "An image of a cat.";
    /// let size = ImageSize::Small;
    /// let image = match client.create_image(prompt)
    ///     .response_format(ResponseFormat::Url)
    ///     .size(size)
    ///     .send()
    ///     .await {     
    ///        Ok(r) => r.json::<Image>().await.unwrap(), // Doesn't matter whether we requested b64_json or url
    ///        Err(e) => panic!("{e}"),
    ///   };
    ///
    /// let filename = format!("results/images/{}", image.created);
    /// image.save_json(&filename).unwrap();
    /// ```
    pub fn create_image<P: Into<String>>(&self, prompt: P) -> ImageBuilder<ImageGen> {
        ImageBuilder::create_image(&self.key, prompt)
    }

    /// Image format must be `RGBA`, `LA`, or `L`, (`RGB` will return an error from the API)."
    pub fn create_image_edit<I, P>(&self, image_path: I, prompt: P) -> ImageBuilder<ImageEdit>
    where
        I: Into<String>,
        P: Into<String>,
    {
        ImageBuilder::create_edit(&self.key, image_path, prompt)
    }

    /// Create a variation of an image.
    ///
    /// # Arguments
    /// * `image_path` - A `&str` or `String` representing path to the image to use for the
    ///  variation.
    ///
    /// # Optional parameters
    /// The following can be set with the corresponding methods:
    /// * `n` - The number of images to generate. Must be between `1` and `10`. Defaults to
    /// `1`. Panics if `n` is not in this range.
    ///
    /// * `size` - The [`ImageSize`](images::ImageSize) of the image to
    /// generate. Size vairants are [`ImageSize::Small`](images::ImageSize) (`256x256`),
    /// [`ImageSize::Medium`](images::ImageSize) (`512x512`), and
    /// [`ImageSize::Large`](images::ImageSize) (`1024x1024`). API defaults to
    /// [`ImageSize::Large`](images::ImageSize).
    ///
    /// * `response_format` - The
    /// [`ResponseFormat`](images::response::ResponseFormat) of the image to generate. Format
    /// variants are [`ResponseFormat::Url`](images::response::ResponseFormat) and
    /// [`ResponseFormat::Base64`](images::response::ResponseFormat). API defaults to
    /// [`ResponseFormat::Url`](images::response::ResponseFormat).
    ///
    /// * `user` - The unique id for a user, for detecting abuse.
    pub fn create_image_variation<I>(&self, image_path: I) -> ImageBuilder<ImageVariation>
    where
        I: Into<String>,
    {
        ImageBuilder::create_variation(&self.key, image_path)
    }

    /// Delete a file that belongs to your organization.
    /// [OpenAI API Docs](https://platform.openai.com/docs/api-reference/files/delete)
    ///
    /// # Arguments
    /// * `file_id` - The ID of the file to delete. Will have a format similar to `file-xxxxxxx...`.
    /// If you don't know the ID of the file you want to delete, you can use the [`list`] method to
    /// get a list of all files that belong to your organization.
    ///
    /// # Returns
    /// `Client<Delete>` that can be used to execute the request by awaiting `send()`. This is used for
    /// consistency/reliability with the rest of the library.
    ///
    /// `Result<reqwest::Response, OairsError>` is returned after `send()` is awaited.
    ///
    /// # Example
    /// ```rust,no_run
    /// let key = std::env::var("OPENAI_API_KEY").unwrap();
    /// let client = client::Client::new(key);
    ///
    /// let file_id = "file-eV13Rfwj2AWw8EVDQ0kEzdk4";
    /// let response = match client.delete_file(file_id).send().await {
    ///     Ok(r) => r.json::<DeleteResponse>().await.unwrap(),
    ///     Err(e) => panic!("{e}"),
    /// };
    /// // ...
    /// ```
    pub fn delete_file(self, file_id: &str) -> Client<Delete> {
        Client {
            key: self.key,
            url: Some(delete_file_url(file_id)),
            ..Default::default()
        }
    }

    /// "Delete a fine-tuned model. You must have the Owner role in your organization." -
    /// [OpenAI API docs](https://platform.openai.com/docs/api-reference/fine-tunes/delete-model)
    ///
    /// # Arguments
    /// * `model` - The model to delete. Will have a format similar to `curie:ft-personal-2022-12-31-20-10-18`
    ///
    /// # Returns
    /// `Client<Delete>` that can be used to execute the request by awaiting `send()`.
    ///
    /// `Result<reqwest::Response, OairsError>`. A Successful response can be deserialized into a
    /// [`DeleteResponse`]. An unsuccessful response can be deserialized into an [`InvalidRequest`].
    ///
    /// # Example
    /// ```rust,no_run
    /// let key = std::env::var("OPENAI_API_KEY").unwrap();
    /// let client = Client::new(key);
    ///
    /// let model_id = "curie:ft-personal-2022-12-31-20-10-18";
    /// let response = match client.delete_fine_tune_model(model_id).send().await {
    ///     Ok(r) => r,
    ///     Err(e) => panic!("{e}"),
    /// };
    ///
    /// if response.status().is_success() {
    ///    let delete_response = response.json::<DeleteResponse>().await.unwrap();
    /// // ...
    /// ```
    pub fn delete_fine_tune_model(self, model: &str) -> Client<Delete> {
        Client {
            key: self.key,
            url: Some(delete_ft_model_url(model)),
            ..Default::default()
        }
    }

    /// Get a list of files that you've uploaded to the server or that have been generated by OpenAI
    /// (e.g., as part of a fine-tune request).
    /// [OpenAI API Docs](https://platform.openai.com/docs/api-reference/files/list)
    ///
    /// # Returns
    /// `Client<Gettable>` that can be used to execute the request by awaiting `send()`.
    ///
    /// `Result<reqwest::Response, OairsError>` is returned by awaiting `send()`. A successful response
    /// can be deserialized using the [`FileList`] struct.
    ///
    /// # Example
    /// ```rust,no_run
    /// // ...
    /// let file_list = match client.list_files().send().await {
    ///     Ok(res) => res.json::<FileList>().await.unwrap();,
    ///     Err(e) => panic!("{e}"),
    /// };
    /// // ...
    /// ```
    pub fn list_files(&self) -> Client<Gettable> {
        Client {
            key: self.key.to_owned(),
            url: Some(list_files_url().to_string()),
            ..Default::default()
        }
    }

    /// "List your organization's fine-tuning jobs" -
    /// [OpenAI API docs](https://platform.openai.com/docs/api-reference/fine-tunes/list)
    ///
    /// # Returns
    /// `Client<Gettable>` that can be used to execute the request by awaiting `send()`.
    ///
    /// Awaiting `send()` will return `Result<reqwest::Response, OairsError>`. A Successful
    /// response can be deserialized into a [`FineTunesList`].
    ///
    /// # Example
    /// ```rust,no_run
    /// // ...
    /// let ft_list = match client.list_fine_tunes().send().await {
    ///    Ok(response) => response.json::<FineTunesList>().await.unwrap();,
    ///    Err(e) => panic!("{e}"),
    /// };
    /// // ...
    /// ```
    pub fn list_fine_tunes(&self) -> Client<Gettable> {
        Client {
            key: self.key.to_owned(),
            url: Some(list_fine_tunes_url().to_string()),
            ..Default::default()
        }
    }

    /// "Get fine-grained status updates for a fine-tune job." -
    /// [OpenAI API docs](https://platform.openai.com/docs/api-reference/fine-tunes/events)
    ///
    /// This method doesn't provide any information that won't be available in a
    /// response from a call to [`retrieve_fine_tune_info`](Client::retrieve_fine_tune_info).
    /// The difference between the two methods comes down to whether the job is in progress
    /// or not. If the job is in progress, you can optionally use this method to get a stream
    /// of events.
    ///
    /// Once the job has finished, you can use either method to get the information. The
    /// information *is* organized differently, but all of the same information can retrieved
    /// from either method.
    ///
    /// # Arguments
    /// * `fine_tune_id` - The ID of the fine-tune job to get events for. Will have a format
    /// similar to `ft-xxxxx...`
    ///
    /// # Optional Arguments
    /// The following can be set via chaining on the returned `ListEventsBuilder<Sendable>`:
    /// * `stream` - If `true`, the server will stream events as they happen.
    ///
    /// # Returns
    /// `ListEventsBuilder<Sendable>` that can be used to execute the request by awaiting `send()`.
    ///
    /// `Result<reqwest::Response, OairsError>`. A Successful response can be deserialized into
    /// an [`EventList`].
    ///
    /// # Example
    /// ```rust,no_run
    /// let key = std::env::var("OPENAI_API_KEY").unwrap();
    /// let client = Client::new(key);
    ///
    /// let fine_tune_id = "ft-2gotxnRxraEdN4YI7oJUb8It";
    /// let ft_object = match client.list_fine_tune_events(fine_tune_id).send().await {
    ///     Ok(response) => response.json::<EventList>().await.unwrap(),
    ///     Err(e) => panic!("{e}"),
    /// };
    /// // ...
    /// ```
    pub fn list_fine_tune_events(&self, fine_tune_id: &str) -> ListEventsBuilder<Sendable> {
        ListEventsBuilder::new(&self.key, fine_tune_id)
    }

    /// List all available models and their associated information.
    ///
    /// # Returns
    /// `Client<Gettable>` that can be used to execute the request by awaiting `send()`.
    ///
    /// Awaiting `send()` returns a [`Result`] with either a
    /// [`Response`](https://docs.rs/reqwest/0.11.14/reqwest/struct.Response.html) or an [`OairsError`].
    /// If response, it can be deserialized into a [`ModelsList`] struct. Both
    /// [`ModelsList`] and [`OairsError`] implement the `SaveJson` trait, which allows you to save
    /// the response to a `json` file with the `save_json()` method.
    ///
    /// # Example
    /// ```rust,no_run
    /// let key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");
    /// let client = Client::new(key);
    ///
    /// let response = match client.list_models().send().await {
    ///    Ok(r) => r.json::<ModelsList>().await.unwrap(),
    ///    Err(e) => panic!("{e}"),
    /// }
    /// ```
    /// # Example
    /// ```rust,no_run
    /// let models_list: ModelsList = response.json().await.unwrap();
    /// // All of the response structs implement a `SaveJson` trait, which allows you to save
    /// // the response to a `json` file.
    /// match models_list.save_json("list_models") {
    ///   Ok(_) => (),
    ///   Err(e) => panic!("{}", e),
    /// }
    pub fn list_models(&self) -> Client<Gettable> {
        Client {
            key: self.key.to_owned(),
            url: Some(list_models_url().to_string()),
            ..Default::default()
        }
    }

    /// Retrieve information about the specified file.
    ///
    /// # Arguments
    /// * file_id - The ID of the file (as a `&str`) to retrieve. Will be a format similar to
    /// "file-xxxxxxx...". If you don't know the ID of the file you want to retrieve, you can
    /// use the [`list_files`] method to get a list of all files that belong to your organization.
    ///
    /// # Returns
    /// `Client<Gettable>` that can be used to execute the request by awaiting `send()`.
    ///
    /// `Result<reqwest::Response, OairsError>` is returned after `send()` is awaited. A successful
    /// request can be deserialized into a [`FileInfo`](crate::files::response::FileInfo).
    ///
    /// # Example
    /// ```rust,no_run
    /// let key = get_key();
    /// let client = Client::new(key);
    ///
    /// let file_id = "file-BRwHsFt1LywSy7KeWcz3Mscv";
    /// let file_info = match client.retrieve_file(file_id).send().await {
    ///     Ok(res) => res.json::<FileInfo>().await.unwrap(),
    ///     Err(e) => panic!("{e}"),
    /// };
    ///
    /// // ...
    /// file_info.save_json(&filename).unwrap();
    /// ```
    pub fn retrieve_file(&self, file_id: &str) -> Client<Gettable> {
        Client {
            key: self.key.to_owned(),
            url: Some(retrieve_file_url(file_id)),
            ..Default::default()
        }
    }

    /// Retrieve the content of the specified file.
    ///
    /// # Arguments
    /// * file_id - A `&str` of the ID of the file to retrieve. Will have a format similar to
    /// "file-xxxxxxx...". If you don't know the ID of the file you want to retrieve, you can use
    /// the [`list_files`](crate::client::Client::list_files) method to get a list of all files
    /// that belong to your organization.
    ///
    /// # Returns
    /// `Client<Gettable>` that can be used to execute the request by awaiting `send()`.
    ///
    /// `Result<reqwest::Response, OairsError>` is returned after `send()` is awaited.
    ///
    /// ## Deserializing a Fine Tune File
    /// Currently there are two types of files that might exist for a given organization. The first
    /// is a fine-tune training file that would have a `.jsonl` extension and should have the following
    /// format:
    ///
    /// ```json
    /// {"prompt":"Example prompt with example separator\n\n###\n\n","completion":" Some completion"}
    /// {"prompt":"Example prompt with example separator\n\n###\n\n","completion":" Some completion"}
    /// ```
    ///
    /// This sort of file can be deserialized using the [`FineTuneFC`](crate::files::response::FineTuneFC)
    /// struct, but not in the usual manner of other response structs. Because the lines in a `JSONL`
    /// file are not comma separated, [`serde`](https://docs.rs/serde/1.0.158/serde/)/
    /// [`serde_json`](https://docs.rs/serde_json/1.0.94/serde_json/) cannot deserialize them directly
    /// (or not that I'm aware of). Instead, you can pass the
    /// [`Response`](https://docs.rs/reqwest/0.11.14/reqwest/struct.Response.html) to the
    /// [`from_response`](crate::files::response::FineTuneFC::from_response) method of
    /// the [`FineTuneFC`](crate::files::response::FineTuneFC) struct.
    ///
    /// ## Example
    /// ```rust,no_run
    /// let fine_tune_fc = match client.retrieve_file_content(file_id).send().await {
    ///    Ok(res) => FineTuneFC::from_response(res).await.unwrap(),
    ///    Err(e) => panic!("{e}"),
    /// };
    /// ```
    /// Again, due to the nature of it being `JSONL`, we don't have the usual `save_json` method
    /// available. However, there is a [`save_jsonl`](crate::files::response::FineTuneFC::save_jsonl)
    /// method that can be used to save the file contents.
    ///
    /// ## Example
    /// ```rust,no_run
    /// match fine_tune_fc.save_jsonl(&filename) {
    ///    Ok(_) => (),
    ///    Err(e) => panic!("{}", e),
    /// }
    /// ```
    /// ## Deserializing a Fine Tune Results File
    /// The other sort of file that might exist is one generated by OpenAI as part of a fine-tune
    /// request and is marked with the purpose "fine-tune-results". This will be a `CSV` file.
    ///
    /// Oairs provides two ways to deserialize a
    /// [`Response`](https://docs.rs/reqwest/0.11.14/reqwest/struct.Response.html)
    /// from the retrieve file content endpoint, when that file request is a fine-tune results file:
    ///
    /// 1. Deserialize to a [`FineTuneResultsFC`](crate::files::response::FineTuneResultsFC) struct -
    /// This can be done by passing the response to
    /// [`FineTuneResultsFC::from_response`](crate::files::response::FineTuneResultsFC::from_response).
    ///
    /// ## Example
    /// ```rust,no_run
    /// let key = std::env::var("OPENAI_API_KEY").unwrap();
    /// let client = client::Client::new(key);
    ///
    /// let file_id = "file-xewFBtsiD2hbU47Du1G37zJE";
    /// match client.retrieve_file_content(file_id).send().await {
    ///     Ok(res) => {
    ///        let ftr_content = FineTuneResultsFC::from_response(res);
    ///        let filename = "some/filename";
    ///        ftr_content.save_json(filename).unwrap();
    ///     },
    ///     Err(e) => panic!("{e}"),
    /// };
    /// ```
    ///
    /// 2. Deserialize to a Polars [`DataFrame`](https://docs.rs/polars/0.27.2/polars/prelude/struct.DataFrame.html) -
    /// This can be done by passing the response to the
    /// [`df_from_response`](crate::utils::df::df_from_response) function.
    /// You can save the dataframe as either a `CSV` or a `Parquet` file by using
    /// the utility functions [`write_csv`](crate::utils::df::write_csv) and
    /// [`write_parquet`](crate::utils::df::write_parquet).
    ///
    /// # Example
    /// ```rust,no_run
    /// // ...
    /// match client.retrieve_file_content(file_id).send().await {
    ///    Ok(res) => {
    ///       // Saving the DataFrame as a CSV or Parquet file
    ///       // requires we make the df mutable.
    ///       let mut df = df_from_response(res).await.unwrap();
    ///       // The extension of the filename will be set for you
    ///       // if you don't include it.
    ///       let filename = "some/filename";
    ///
    ///       // How to save as a CSV file
    ///       write_csv(&mut df, filename).unwrap();
    ///       // How to save as a Parquet file
    ///       write_parquet(&mut df, filename).unwrap();
    ///    },
    ///    Err(e) => panic!("{e}"),
    /// }
    /// ```
    pub fn retrieve_file_content(&self, file_id: &str) -> Client<Gettable> {
        Client {
            key: self.key.to_owned(),
            url: Some(retrieve_file_content_url(file_id)),
            ..Default::default()
        }
    }

    /// "Gets info about the fine-tune job." -
    /// [OpenAI API docs](https://platform.openai.com/docs/api-reference/fine-tunes/retrieve)
    ///
    /// This doesn't require that the "job" be in progress.
    ///
    /// # Arguments
    /// * fine_tune_id - A `&str` of the ID of the fine-tune job to retrieve. Will have a format
    /// similar to "ft-xxxxxxx...".
    ///
    /// # Returns
    /// `Client<Gettable>` that can be used to execute the request by awaiting `send()`.
    ///
    /// `Result<reqwest::Response, OairsError>`. A Successful response can be deserialized into
    /// a [`FineTuneInfo`]. As with all structs that can be deserialized from a successful
    /// response, it can be saved to a file using the [`save_json`] method.
    ///
    /// # Example
    /// ```rust,no_run
    /// let key = std::env::var("OPENAI_API_KEY").unwrap();
    /// let client = Client::new(key);
    ///
    /// let fine_tune_id = "ft-2gotxnRxraEdN4YI7oJUb8It";
    /// let ft_object = match client.retrieve_fine_tune_info(fine_tune_id).send().await {
    ///     Ok(r) => r.json::<FineTuneInfo>().await.unwrap(),
    ///     Err(e) => panic!("{e}"),
    /// };
    /// // ...
    /// ```
    pub fn retrieve_fine_tune_info(&self, fine_tune_id: &str) -> Client<Gettable> {
        Client {
            key: self.key.to_owned(),
            url: Some(retrieve_ft_info_url(fine_tune_id)),
            ..Default::default()
        }
    }

    /// Retrieve information about a model.
    ///
    ///
    /// # Arguments
    /// * `model` - The model to retrieve information about. The model needs to implement the
    /// `RetrievableModel` trait, which is implemented for the following model
    /// enums:
    ///   - [`EditModel`]
    ///   - [`ChatModel`]
    ///   - [`CompletionModel`]
    ///   - [`EmbeddingModel`]
    ///   - [`FineTuneModel`]
    ///
    /// If you need to retrieve information about a model that is not included in the above enums,
    /// you can use either the [`custom_model!`] or [`ft_model!`] macros to create a custom/fine-tuned
    /// model enum. These macros will implement the `RetrievableModel` trait for you.
    ///
    /// # Returns
    /// `Client<Gettable>` that can be used to execute the request by awaiting `send()`.
    ///
    /// Awaiting `send()` returns a [`Result`] with either a
    /// [`Response`](https://docs.rs/reqwest/0.11.14/reqwest/struct.Response.html) or an [`OairsError`].
    /// The response can be deserialized into a [`ModelObject`] struct. Both [`ModelObject`] and
    /// [`OairsError`] implement the `SaveJson` trait, which allows you to save the response to a
    /// `json` file with the `save_json()` method.
    ///
    /// # Examples
    /// ```rust,no_run
    /// let key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");
    /// let client = Client::new(key);
    ///
    /// let model = EditModel::CodeDavinci002;
    /// let model_obj = match client.retrieve_model(&model).send().await {
    ///   Ok(r) => r.json::<ModelObject>().await.unwrap(),
    ///   Err(e) => panic!("{e}"),
    /// };
    /// ```
    ///
    /// ## Retrieving an unenumerated model
    /// If you need to retrieve a model that hasn't been enumerated in the library, you can use the
    /// [`custom_model!`] or [`ft_model!`] macros to create a custom/fine-tuned model enum.
    /// ```rust,no_run
    /// // You can give them whatever name you want as the first argument, but the second argument
    /// // must match the model id OpenAI uses exactly. If you're unsure of what this is, use the
    /// // `list()` method to get a list of all available models.
    /// custom_model!(IfDavinciv2, "if-davinci-v2");
    /// ft_model!(CurieFtPersonal, "curie:ft-personal-2023-02-18-20-10-18");
    /// // ...
    /// let custom_model = CustomModel::IfDavinciv2;
    /// let ft_model = FineTunedModel::CurieFtPersonal;
    /// // They will implement the `RetrievableModel` trait, so you can pass them to `retrieve()`
    /// let model_info = match client.retrieve_model(&davinci).send().await {
    ///   Ok(response) => response.json::<ModelObject>().await.unwrap(),
    ///   Err(e) => panic!("{e}"),
    /// }
    /// ```
    pub fn retrieve_model<R>(&self, model: &R) -> Client<Gettable>
    where
        R: RetrievableModel,
    {
        Client {
            key: self.key.to_owned(),
            url: Some(retrieve_model_url(model.to_str())),
            ..Default::default()
        }
    }

    /// "Upload a file that contains document(s) to be used across various endpoints/features. Currently,
    /// the size of all the files uploaded by one organization can be up to `1 GB`." -
    /// [OpenAI API Docs](https://platform.openai.com/docs/api-reference/files/upload)
    ///
    /// # Arguments
    /// * `file` - A `&str` path to the file you want to upload. Since currently the only purpose for
    /// uploading a file is to use it for fine-tuning, the file should be `.jsonl`.
    /// * `purpose` - A `Purpose`. Currently, the only available purpose by the API is 'fine-tune'
    /// (hence, [`Purpose::FineTune`]). The enum constraint is to allow for future expansion of the API
    /// and more generally to ensure that the purpose is properly set.
    ///
    /// # Returns
    /// `FileBuilder<'a, Create>` that can be used to execute the request by awaiting `send()`.
    ///
    /// `Result<reqwest::Response, OairsError>` is returned by awaiting `send()`. A successful response
    /// can be deserialized using the [`FileInfo`](crate::files::response::FileInfo) struct.
    ///
    /// # Example
    /// ```rust,no_run
    /// let key = std::env::var("OPENAI_API_KEY").unwrap();
    /// let client = client::Client::new(key);
    ///
    /// let filepath = "some/path/file.jsonl";
    /// let purpose = Purpose::FineTune;
    ///
    /// let file_info = match client.upload_file(filepath, purpose).send().await {
    ///     Ok(response) => response.json::<FileInfo>().await.unwrap(),
    ///     Err(e) => panic!("{e}"),
    /// };
    /// ```
    pub fn upload_file<F: Into<String>>(&self, file: F, purpose: Purpose) -> Client<Sendable> {
        Client {
            key: self.key.clone(),
            url: Some(upload_file_url().to_string()),
            upload_filename: Some(file.into()),
            file_purpose: Some(purpose),
            ..Default::default()
        }
    }
}

impl Client<Cancel> {
    // Currently only used by `cancel_fine_tune` method.
    /// Executes the `POST` request. Returns a `Result` with either a `reqwest::Response` or an
    /// `OairsError`.
    pub async fn send(&self) -> Result<reqwest::Response, OairsError> {
        let url = self.url.clone().unwrap();
        handle_request(&self.key, &url, HttpMethod::Post, None, None).await
    }
}

impl<'a> Client<Gettable> {
    pub async fn send(&self) -> Result<reqwest::Response, OairsError> {
        let url = self.url.clone().unwrap();
        handle_request(&self.key, &url, HttpMethod::Get, None, None).await
    }
}

// For `upload_file` requests.
impl Client<Sendable> {
    /// Executes the `POST` request for a form. Returns a `Result` with either a
    /// [`Response`](https://docs.rs/reqwest/0.11.14/reqwest/struct.Response.html)
    /// or an `OairsError`.
    pub async fn send(&self) -> Result<reqwest::Response, OairsError> {
        let path = self.upload_filename.as_ref().unwrap();
        let purpose = self.file_purpose.as_ref().unwrap();

        let file_part = get_file_part(path)?;
        let form = reqwest::multipart::Form::new()
            .text("purpose", purpose.to_string())
            .part("file", file_part);

        let url = self.url.clone().unwrap();
        handle_request(&self.key, &url, HttpMethod::Post, None, Some(form)).await
    }
}

impl Client<Delete> {
    /// Executes the `DELETE` request. Returns a `Result` with either a
    /// [`Response`](https://docs.rs/reqwest/0.11.14/reqwest/struct.Response.html)
    ///  or an `OairsError`.
    pub async fn send(&self) -> Result<reqwest::Response, OairsError> {
        let url = self.url.clone().unwrap();
        handle_request(&self.key, &url, HttpMethod::Delete, None, None).await
    }
}

// Below: Some helper functions for handling the request and response

pub(crate) enum HttpMethod {
    Get,
    Post,
    Delete,
}

pub(crate) async fn handle_request(
    key: &str,
    url: &str,
    http_method: HttpMethod,
    json: Option<serde_json::Value>,
    form: Option<reqwest::multipart::Form>,
) -> Result<reqwest::Response, OairsError> {
    let client = match build_client(key) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };

    let request = match set_method(client, url.to_string(), http_method) {
        Ok(r) => r,
        Err(e) => return Err(e),
    };

    if let Some(json) = json {
        send(
            request
                .header("Content-Type", "application/json")
                .json(&json),
        )
        .await
    } else if let Some(form) = form {
        send(
            request
                .header("Content-Type", "multipart/form-data")
                .multipart(form),
        )
        .await
    } else {
        send(request).await
    }
}

// (Just following reqwest example for the most part)
pub(crate) fn build_client(key: &str) -> Result<reqwest::Client, OairsError> {
    let user_agent = concat!(env!("CARGO_PKG_NAME"), "/", env!("CARGO_PKG_VERSION"));

    let mut headers = header::HeaderMap::new();
    // Using an unwrap here because I assume the OpenAI API won't generate a key
    // that uses non-visible ASCII characters
    let mut auth_value = header::HeaderValue::from_str(key).unwrap();
    auth_value.set_sensitive(true);
    headers.insert(header::AUTHORIZATION, auth_value);

    match reqwest::Client::builder()
        .user_agent(user_agent)
        .default_headers(headers)
        .build()
    {
        Ok(c) => Ok(c),
        Err(e) => Err(builder_error(e)),
    }
}

pub(crate) fn set_method(
    client: reqwest::Client,
    url: String,
    http_method: HttpMethod,
) -> Result<reqwest::RequestBuilder, OairsError> {
    let request = match http_method {
        HttpMethod::Get => client.get(url),
        HttpMethod::Post => client.post(url),
        HttpMethod::Delete => client.delete(url),
    };

    Ok(request)
}

pub(crate) fn get_file_part(path: &str) -> Result<Part, OairsError> {
    let file = std::fs::read(path)?;
    let file_part = Part::bytes(file).file_name(path.to_string());

    Ok(file_part)
}

pub(crate) async fn send(
    request: reqwest::RequestBuilder,
) -> Result<reqwest::Response, OairsError> {
    let response = match request.send().await {
        Ok(r) => r,
        Err(e) => return Err(parse_reqwest_error(e)),
    };

    check_status(response).await
}

pub(crate) async fn check_status(
    response: reqwest::Response,
) -> Result<reqwest::Response, OairsError> {
    let sc = response.status();
    match sc {
        reqwest::StatusCode::OK => Ok(response),
        _ => Err(parse_api_error(response, sc).await),
    }
}
