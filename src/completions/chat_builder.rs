// TODO: Clean up type-state pattern!

use serde::ser::SerializeSeq;

use super::{response::ChatCompletion, *};
use crate::{
    client::{handle_request, HttpMethod},
    tokenizers::{tokenize, Tokenizer},
};

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct ChatBuilder<Buildable> {
    #[serde(skip)]
    key: String,
    #[serde(skip)]
    url: String,
    model: ChatModel,
    messages: Messages,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<Temperature>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<TopP>,
    n: u8,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u16>,
    /// Defaults to 0.0 if `presence_penalty` is not specified.
    presence_penalty: f32,
    /// Defaults to 0.0 if `frequency_penalty` is not specified.
    frequency_penalty: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    logit_bias: Option<HashMap<String, f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    user: Option<String>,
    #[serde(skip)]
    state: PhantomData<Buildable>,
}

impl ChatBuilder<Buildable> {
    pub fn create<K>(key: K, model: ChatModel, msgs: &Messages) -> ChatBuilder<Sendable>
    where
        K: Into<String>,
    {
        ChatBuilder {
            key: key.into(),
            url: chat_completion_url().to_string(),
            model,
            messages: msgs.to_owned(),
            n: 1,
            state: PhantomData::<Sendable>,
            ..Default::default()
        }
    }
}

impl ChatBuilder<Sendable> {
    /// The amount of randomness for the model to use when generating the
    /// completion. The valid range is 0 to 2. A value of 2 can lead to
    /// incoherent completions.
    /// [OpenAI API docs](https://platform.openai.com/docs/api-reference/chat/create#chat/create-temperature)
    pub fn temperature(&mut self, temperature: Temperature) -> &mut Self {
        self.temperature = Some(temperature);
        self
    }

    /// Alternative to using temperature. Not recommended to use both. cf.
    /// [OpenAI API Docs](https://platform.openai.com/docs/api-reference/chat/create#chat/create-top_p)
    pub fn top_p(&mut self, top_p: TopP) -> &mut Self {
        self.top_p = Some(top_p);
        self
    }

    /// Number of chat completions to generate per prompt.
    /// [OpenAI API docs](https://platform.openai.com/docs/api-reference/chat/create#chat/create-n)
    pub fn n(&mut self, n: u8) -> &mut Self {
        self.n = n;
        self
    }

    /// "If set, partial message deltas will be sent, like in ChatGPT. Tokens will be sent as data-only server-sent events
    /// as they become available, with the stream terminated by a `data: [DONE]` message."
    /// - [OpenAI API docs](https://platform.openai.com/docs/api-reference/chat/create#chat/create-stream)
    pub fn stream(&mut self, truth_value: bool) -> &mut Self {
        self.stream = truth_value;
        self
    }

    /// "Up to 4 sequences where the API will stop generating further tokens."
    /// - [OpenAI API docs](https://platform.openai.com/docs/api-reference/chat/create#chat/create-stop)
    pub fn stop(&mut self, stop: Vec<String>) -> &mut Self {
        self.stop = Some(stop);
        self
    }

    /// Maximum number of tokens to generate. Model may generate fewer than `max_tokens`.
    /// [OpenAI API Docs](https://platform.openai.com/docs/api-reference/chat/create#chat/create-max_tokens)
    pub fn max_tokens(&mut self, max_tokens: u16) -> &mut Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// "Number between -2.0 and 2.0. Positive values penalize new tokens based
    /// on whether they appear in the text so far, increasing the model's
    /// likelihood to talk about new topics."
    /// - [OpenAI API docs](https://platform.openai.com/docs/api-reference/chat/create#chat/create-presence_penalty)
    pub fn presence_penalty(&mut self, penalty: f32) -> &mut Self {
        if !(-2.0..=2.0).contains(&penalty) {
            panic!("Presence penalty must be between -2.0 and 2.0");
        }
        self.presence_penalty = penalty;
        self
    }

    /// "Number between -2.0 and 2.0. Positive values penalize new tokens based
    /// on their existing frequency in the text so far, decreasing the model's
    /// likelihood to repeat the same line verbatim."
    /// - [OpenAI API docs](https://platform.openai.com/docs/api-reference/chat/create#chat/create-frequency_penalty)
    pub fn frequency_penalty(&mut self, penalty: f32) -> &mut Self {
        if !(-2.0..=2.0).contains(&penalty) {
            panic!("Frequency penalty must be between -2.0 and 2.0");
        }
        self.frequency_penalty = penalty;
        self
    }

    /// Influence how likely or unlikey it is that the model will choose a
    /// given token. (Note: token != word. For example, the word "Ninny" is
    /// tokenized to \[36091, 3281\] where 36091 represents 'Nin' and 3281
    /// represents 'ny'. You can use the [`tokenizer`] module or the
    /// [tokenizer](https://platform.openai.com/tokenizer?view=bpe)
    /// tool provided by OpenAI to see how a word is tokenized.) Range should
    /// be from `100.0` to `-100.0`. However, selecting values at the extremes may
    /// lead to incoherent completions.
    ///
    /// cf. [documentation](https://platform.openai.com/docs/api-reference/chat/create#chat/create-logit_bias)
    ///
    /// # Example
    /// 100.0 does not make it certain that a token will occur, nor does -100.0
    /// make it certain that a token will *not* occur.
    /// ```rust,no_run
    /// // ...
    /// let mut try_ban = HashMap::new();
    /// try_ban.insert("9703", -100.0);  // 9703: "dog"
    /// try_ban.insert("22242", -100.0); // 22242: "dogs"
    /// try_ban.insert("32942", -100.0); // 32942: "Dog"
    ///
    /// use Msg::System;
    /// use Msg::User;
    ///
    /// let system = System("You are a helpful assistant.");
    /// let user = User("What is an animal that barks?");
    ///
    /// let model = ChatModel::default();
    /// let mut messages = Messages::default();
    /// messages.push(system);
    /// messages.push(user);
    ///
    /// let client = Client::new(key)
    ///     .chat_completion(model, &messages)
    ///     .max_tokens(200)
    ///     .logit_bias(try_ban)
    ///     .send()
    ///     .await
    /// {
    ///     Ok(response) => response, // Response was "\n\n- Dog"
    ///     Err(e) => panic!("{e}"),
    /// };
    /// // ...
    /// ```
    /// If we try to push it towards "cat", we get an incoherent response:
    /// ```rust,no_run
    /// // ...
    /// let mut try_ban = HashMap::new();
    /// try_ban.insert("9703", -100.0);  // 9703: "dog"
    /// try_ban.insert("22242", -100.0); // 22242: "dogs"
    /// try_ban.insert("32942", -100.0); // 32942: "Dog"
    /// try_ban.insert("9246", 100.0);   // 9246: "cat"
    ///
    /// // ...
    ///
    /// let client = Client::new(key)
    ///     .chat_completion(model, &messages)
    ///     .max_tokens(200)
    ///     .logit_bias(try_ban)
    ///     .send()
    ///     .await
    /// {
    ///     // Response was "catcatcatcatcatcatcatcatcat" ...
    ///     Ok(response) => response,
    ///     Err(e) => panic!("Error: {}", e),
    /// };
    /// ```
    pub fn logit_bias(&mut self, logit_bias: HashMap<String, f32>) -> &mut Self {
        self.logit_bias = Some(logit_bias);
        self
    }

    /// "A unique identifier representing your end-user, which can help OpenAI
    /// to monitor and detect abuse." -
    /// [OpenAI API docs](https://platform.openai.com/docs/api-reference/chat/create#chat/create-user)
    pub fn user(&mut self, user: String) -> &mut Self {
        self.user = Some(user);
        self
    }

    pub async fn send(&self) -> Result<reqwest::Response, OairsError> {
        let json = serde_json::to_value(self).unwrap();
        handle_request(&self.key, &self.url, HttpMethod::Post, Some(json), None).await
    }
}

// TODO: Implement this
// impl ChatBuilder<Sendable> {
//     pub fn save_with_input(&mut self) -> Result<(), OairsError> {
//         self.save_with_input = true;
//         Ok(())
//     }
// }

// TODOD: Can probaby do without this, but would need to customize deserialization of the API response
/// Used for assisting deserialization of the API response. Can also be used as
/// convenience for checking role type of a Msg.
/// * `Assistant` - designates the model.
/// * `System` - designates the 'narrator', setting the behavior of the model.
/// (Default)
/// * `User` - designates the user or developer, providing instruction or a
/// prompt.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    Assistant,
    #[default]
    System,
    User,
}

impl std::fmt::Display for Role {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Role::Assistant => write!(f, "assistant"),
            Role::System => write!(f, "system"),
            Role::User => write!(f, "user"),
        }
    }
}

impl std::str::FromStr for Role {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "assistant" => Ok(Role::Assistant),
            "system" => Ok(Role::System),
            "user" => Ok(Role::User),
            _ => Err(format!("{} is not a valid role", s)),
        }
    }
}

impl Role {
    pub fn to_str(&self) -> &str {
        match self {
            Role::Assistant => "assistant",
            Role::System => "system",
            Role::User => "user",
        }
    }

    /// Convert a byte slice to a Role.
    /// # Example
    /// ```rust,no_run
    /// // Assume we have a byte slice containing the field
    /// // b"{\"role\":\"assistant\"}" from an API stream
    /// let slice = ...;
    /// let (input, role) = nom_role(slice)?;
    /// let role = Role::from_slice(role)?;
    ///
    /// assert_eq!(role, Role::Assistant);
    /// ```
    pub fn from_slice(s: &[u8]) -> Result<Self, String> {
        match s {
            b"{\"role\":\"assistant\"}" => Ok(Role::Assistant),
            b"{\"role\":\"system\"}" => Ok(Role::System),
            b"{\"role\":\"user\"}" => Ok(Role::User),
            b"assistant" => Ok(Role::Assistant),
            b"system" => Ok(Role::System),
            b"user" => Ok(Role::User),
            _ => Err(format!("{:?} is not a valid role", s)),
        }
    }
}

/// For building a type that can be serialized into what the API expects of a
/// message. It's also possible to deserialize a message from the API into a
/// `Msg`. Cf. [`Messages`]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "role", content = "content", rename_all = "lowercase")]
pub enum Msg {
    Assistant(String),
    System(String),
    User(String),
    #[serde(skip_serializing)]
    Response {
        role: Role,
        content: String,
    },
}

impl std::fmt::Display for Msg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Msg::Assistant(s) => write!(f, "assistant: {}", s),
            Msg::System(s) => write!(f, "system: {}", s),
            Msg::User(s) => write!(f, "user: {}", s),
            Msg::Response { role, content } => write!(f, "{}: {}", role, content),
        }
    }
}

impl std::str::FromStr for Msg {
    type Err = String;

    /// The argument must be in the format of `"{"role": "role", "content":
    /// "content"}"` (white space doesn't matter), the sort of format we would
    /// need to pass to the API (or returned by the API).
    ///
    /// # Example
    /// ```rust,no_run
    /// let s = r#"{"role": "user", "content": "Hello, world!"}"#;
    /// let msg: Msg = Msg::from_str(s).unwrap();
    ///
    /// assert_eq!(msg.role(), "user");
    /// assert_eq!(msg.content(), "Hello, world!");
    /// assert_eq!(msg, Msg::User("Hello, world!".into()));
    /// ```
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut split = s.splitn(2, ',');
        // `role` will look like the following: "{\"role\": \"user\""
        let role = split.next().ok_or("No role found")?;
        // `content` will look like the following: "\"content\": \"Hello\"}"
        let content = split.next().ok_or("No content found")?.trim();

        let mut split = role.splitn(2, ':');
        let role = split.next().ok_or("No content found")?.trim();

        let mut split = content.splitn(2, ':');
        let content = split.next().ok_or("No content found")?.trim();

        // At this point, content value will be something like "\"Hello\"}"
        // We want to strip the quotes and the closing curly brace
        let content = content[1..content.len() - 2].to_string();

        match role {
            "assistant" => Ok(Msg::Assistant(content)),
            "\"assistant\"" => Ok(Msg::Assistant(content)),
            "system" => Ok(Msg::System(content)),
            "\"system\"" => Ok(Msg::System(content)),
            "user" => Ok(Msg::User(content)),
            "\"user\"" => Ok(Msg::User(content)),
            _ => Err(format!("{} is not a valid role", role)),
        }
    }
}

impl Msg {
    pub fn role(&self) -> Role {
        match self {
            Msg::Assistant(_) => Role::Assistant,
            Msg::System(_) => Role::System,
            Msg::User(_) => Role::User,
            Msg::Response { role, .. } => role.clone(),
        }
    }

    pub fn role_as_str(&self) -> &str {
        match self {
            Msg::Assistant(_) => "assistant",
            Msg::System(_) => "system",
            Msg::User(_) => "user",
            Msg::Response { role, .. } => role.to_str(),
        }
    }

    pub fn content(&self) -> &str {
        match self {
            Msg::Assistant(s) => s,
            Msg::System(s) => s,
            Msg::User(s) => s,
            Msg::Response { content, .. } => content,
        }
    }

    pub fn set_content(&mut self, content: String) {
        match self {
            Msg::Assistant(_) => *self = Msg::Assistant(content),
            Msg::System(_) => *self = Msg::System(content),
            Msg::User(_) => *self = Msg::User(content),
            Msg::Response { role, .. } => {
                *self = Msg::Response {
                    role: role.clone(),
                    content,
                }
            }
        }
    }

    /// Assumes no special tokens.
    pub fn tokens(&self) -> Result<Vec<usize>, OairsError> {
        match self {
            Msg::Assistant(s) => tokenize(s, Tokenizer::CL100KBase),
            Msg::System(s) => tokenize(s, Tokenizer::CL100KBase),
            Msg::User(s) => tokenize(s, Tokenizer::CL100KBase),
            Msg::Response { content, .. } => tokenize(content, Tokenizer::CL100KBase),
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize, SaveJson)]
pub struct Messages {
    data: Vec<Msg>,
    #[serde(skip)]
    save_with_tokens: bool,
    tokens: Vec<Vec<usize>>,
}

impl std::fmt::Display for Messages {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut s = String::new();
        for message in &self.data {
            s.push_str(&message.to_string());
            s.push_str(", ");
        }
        write!(f, "[{}]", s.trim())
    }
}

impl Serialize for Messages {
    fn serialize<Ser>(&self, serializer: Ser) -> Result<Ser::Ok, Ser::Error>
    where
        Ser: serde::Serializer,
    {
        match self.save_with_tokens {
            true => serializer.collect_seq(self.data.iter()),
            false => {
                // Need to produce array of "map/obj" API is expecting:
                let mut seq = serializer.serialize_seq(Some(self.data.len()))?;
                for element in &self.data {
                    seq.serialize_element(&element)?;
                }
                seq.end()
            }
        }
    }
}

impl Messages {
    pub fn new(msgs: Vec<Msg>) -> Messages {
        let mut messages = Messages::default();
        messages.extend(msgs);
        messages
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn push(&mut self, msg: Msg) {
        self.data.push(msg);
    }

    pub fn extend(&mut self, msgs: Vec<Msg>) {
        self.data.extend(msgs);
    }

    pub fn push_response(&mut self, chat_response: &ChatCompletion) {
        self.data.extend(
            chat_response
                .choices
                .iter()
                .map(|choice| choice.message.clone()),
        );
    }

    pub fn clear(&mut self) {
        self.data.clear();
    }

    pub fn save_with_tokens(&mut self, filename: &str) -> Result<(), std::io::Error> {
        // TODO: Handle potential error in tokenization
        self.data.iter().for_each(|msg| {
            let tokens = msg.tokens().unwrap();
            self.tokens.push(tokens)
        });
        self.save_with_tokens = true;
        self.save_json(filename)
    }
}
