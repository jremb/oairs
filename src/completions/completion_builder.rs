//! Structs and enums relevant to handling requests to the completions endpoint. [`CompletionBuilder`] is
//! responsible for building the request. [`Temperature`], [`TopP`] and [`LogProbs`] are necessary for
//! properly setting their respective parameters (the former two also being used by [`ChatBuilder`]).

use crate::client::handle_request;

use super::*;

/// Struct responsible for building a completion create request. Normally you would
/// not use this struct directly, but would interact with it through the `Client`
/// struct.
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
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct CompletionBuilder<State = Buildable> {
    #[serde(skip)]
    key: String,
    #[serde(skip)]
    url: String,
    model: CompletionModel,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    suffix: Option<String>,
    /// Defaults to 16 if `max_tokens` is not specified.
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u16>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<Temperature>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<TopP>,
    /// Default is 1 if `n` is not specified.
    n: u8,
    /// Defaults to `false` if `stream` is not specified.
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    logprobs: Option<u8>,
    // Defaults to `false` if `echo` is not specified.
    echo: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Vec<String>>,
    /// Defaults to 0.0 if `presence_penalty` is not specified.
    presence_penalty: f32,
    /// Defaults to 0.0 if `frequency_penalty` is not specified.
    frequency_penalty: f32,
    /// Defaults to 1 if `best_of` is not specified.
    best_of: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    logit_bias: Option<HashMap<String, f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    user: Option<String>,
    #[serde(skip)]
    state: PhantomData<State>,
}

impl CompletionBuilder<Buildable> {
    pub(crate) fn create<K: Into<String>>(
        key: K,
        model: CompletionModel,
    ) -> CompletionBuilder<Sendable> {
        CompletionBuilder {
            key: key.into(),
            url: completion_url().to_string(),
            model,
            n: 1,
            best_of: 1,
            ..Default::default()
        }
    }
}

impl<'a> CompletionBuilder<Sendable> {
    /// The text for which you wish to generate a completion. For generating
    /// completions from multiple prompts, use the `prompts()` method.
    pub fn prompt(&mut self, prompt: &'a str) -> &mut Self {
        self.prompt = Some(vec![prompt.to_string()]);
        self
    }

    // TODO: Ideally incompatible params would be prevented by further refining the type-state. Look into builder crates!
    /// The texts for which you wish to generate completions.
    ///
    /// # Panics
    /// If `suffix` is also set.
    pub fn prompts(&mut self, prompts: Vec<String>) -> &mut Self {
        self.prompt = Some(prompts);
        self
    }

    /// The [documentation](https://platform.openai.com/docs/api-reference/completions/create#completions/create-suffix)
    /// is not particularly clear on the parameter's function, IMO. You can think of
    /// it as a way to set the direction (*telos*) of the completion. Or you can
    /// think of it as asking the model how it would get from your prompt to
    /// something that (might) naturally end with your suffix. See the examples
    /// section which should better illustrate the function.
    ///
    /// # Panics
    /// If `echo` is also set to `true` or if multiple prompts have been set (via
    /// `prompts()`).
    ///
    /// # Examples
    /// ```rust
    /// ...
    /// let prompt = "What is the purpose of the `suffix` parameter in the OpenAI API's `completions` endpoint?";
    /// let temp = Temperature::new(0.7);
    /// let model = CompletionModel::TextDavinci003;
    /// let mut response = match client
    ///    .completion_create(model)
    ///    .prompt(prompt)
    ///    .max_tokens(200)
    ///    .temperature(temp)
    ///    .suffix("Fuzzy Wuzzy")
    ///    .send()        
    ///    .await
    /// {
    ///    Ok(r) => r,
    ///    Err(e) => panic!("{e}"),
    /// };
    ///
    ///dbg!(&response);
    ///
    /// ```
    /// The completion for the above was:
    /// <pre>
    /// The 'suffix' parameter allows users to define a set of words that are
    /// appended to each completed expression returned by the endpoint. This can be
    /// useful to add context or disambiguate entries. For example, if the original
    /// text was "fuzzy," the 'suffix' parameter could be set to "wuzzy," so that
    /// the completed expression returned is, "Fuzzy Wuzzy." This makes the generated
    /// text more clear and understandable to readers, as the sentence now reads,
    /// </pre>
    ///
    /// ```rust
    /// //...
    /// let prompt = "What is the purpose of the `suffix` parameter in the OpenAI API's `completions` endpoint?";
    /// let temp = Temperature::new(0.7);
    /// let mut response = match client
    ///    .completion_create(model)
    ///    .prompt(prompt)
    ///    .max_tokens(200)
    ///    .temperature(temp)
    ///    .suffix("Mufasa")
    ///    .send()        
    ///    .await
    /// {
    ///    Ok(r) => r,
    ///    Err(e) => panic!("{e}"),
    /// };
    ///
    /// dbg!(&response);
    ///
    /// ```
    /// The completion for the above was:
    /// <pre>
    /// The 'suffix' parameter is used to specify a specific string that must be
    /// included in the completion generated by the OpenAI API. This allows you to
    /// refine or limit the range of possible completions that the API might
    /// provide. For example, if you used the 'suffix' parameter to set the
    /// value to "Mufasa," this would only return completions that end with the
    /// string "Mufasa" such as "remember Mufasa" or "live like "
    /// </pre>
    ///
    /// ```rust,no_run
    /// // ...
    /// let model = Gpt3Model::TextDavinci003;
    /// let prompt = "Bob: Can anyone tell me why I'm standing here talking? Carol:";
    /// let temp = Temperature::new(0.7);
    ///let mut response = match client.completion_create(model)
    ///    .prompt(prompt)
    ///    .max_tokens(200)
    ///    .temperature(temp)
    ///    .suffix("Bob: Aliens, you say?")
    ///    .send()        
    ///    .await
    /// {
    ///    Ok(r) => r,
    ///    Err(e) => panic!("{e}"),
    /// };
    ///
    /// dbg!(&response);
    /// ```
    /// The completion for the above was:
    /// <pre>
    /// " You're standing here talking because aliens made you do it.
    /// Bob: Aliens? What aliens?
    /// Carol: The aliens who live on the dark side of the moon."
    /// </pre>
    pub fn suffix<S: Into<String> + std::fmt::Debug>(&mut self, suffix: S) -> &mut Self {
        if self.echo {
            panic!("Cannot set suffix if echo is set to true");
        }
        if self.prompt.is_some() && self.prompt.as_ref().unwrap().len() > 1 {
            panic!("Cannot set suffix if multiple prompts have been set");
        }
        self.suffix = Some(suffix.into());
        self
    }

    /// Maximum number of tokens to generate. Defaults to 16 if `max_tokens` is not
    /// specified. Model may generate fewer than `max_tokens`. Token count of the
    /// prompt + `max_tokens` cannot exceed model's context length (2048 for older
    /// models, 4096 for newer).
    /// [OpenAI API Docs](https://platform.openai.com/docs/api-reference/completions/create#completions/create-max_tokens)
    pub fn max_tokens(&mut self, max_tokens: u16) -> &mut Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// The amount of randomness for the model to use when generating the
    /// completion. The valid range is 0 to 2. A value of 2 can lead to incoherent
    /// completions.
    pub fn temperature(&mut self, temperature: Temperature) -> &mut Self {
        self.temperature = Some(temperature);
        self
    }

    /// Alternative to using temperature. Not recommended to use both.
    /// cf. [OpenAI API Docs](https://platform.openai.com/docs/api-reference/completions/create#completions/create-top_p)
    pub fn top_p(&mut self, top_p: TopP) -> &mut Self {
        self.top_p = Some(top_p);
        self
    }

    /// Number of completions to generate per prompt.
    pub fn n(&mut self, n: u8) -> &mut Self {
        self.n = n;
        self
    }

    /// "Whether to stream back partial progress. If set, tokens will be sent as
    /// data-only server-sent events as they become available, with the stream
    /// terminated by a `data: [DONE]` message." -
    /// [OpenAI API docs](https://platform.openai.com/docs/api-reference/completions/create#completions/create-stream)
    pub fn stream(&mut self, truth_value: bool) -> &mut Self {
        self.stream = truth_value;
        self
    }

    /// "Include the log probabilities on the `logprobs` most likely tokens, as well
    /// the chosen tokens. For example, if logprobs is [[LogProbs::Five]], the API
    /// will return a list of the 5 most likely tokens. The API will always return
    /// the `logprob` of the sampled token, so there may be up to `logprobs+1`
    /// elements in the  response." -
    /// [OpenAI API docs](https://platform.openai.com/docs/api-reference/completions/create#completions/create-logprobs)
    ///
    /// If `logprobs` is not specified, defaults to null.
    pub fn logprobs(&mut self, logprobs: LogProbs) -> &mut Self {
        self.logprobs = Some(logprobs.to_int());
        self
    }

    /// The prompt(s) will be appended to the beginning of the completion, followed
    /// by two newlines.
    pub fn echo(&mut self, truth_value: bool) -> &mut Self {
        self.echo = truth_value;
        self
    }

    /// "Up to 4 sequences where the API will stop generating further tokens. The
    /// returned text will not contain the stop sequence." -
    /// [OpenAI API docs](https://platform.openai.com/docs/api-reference/completions/create#completions/create-stop)
    pub fn stop(&mut self, stop: Vec<String>) -> &mut Self {
        self.stop = Some(stop);
        self
    }

    /// "Number between -2.0 and 2.0. Positive values penalize new tokens based on
    /// whether they appear in the text so far, increasing the model's likelihood to
    /// talk about new topics." -
    /// [OpenAI API docs](https://platform.openai.com/docs/api-reference/completions/create#completions/create-presence_penalty)
    pub fn presence_penalty(&mut self, penalty: f32) -> &mut Self {
        if !(-2.0..=2.0).contains(&penalty) {
            panic!("Presence penalty must be between -2.0 and 2.0");
        }
        self.presence_penalty = penalty;
        self
    }

    /// "Number between -2.0 and 2.0. Positive values penalize new tokens based on
    /// their existing frequency in the text so far, decreasing the model's
    /// likelihood to repeat the same line verbatim." -
    /// [OpenAI API docs](https://platform.openai.com/docs/api-reference/completions/create#completions/create-frequency_penalty)
    pub fn frequency_penalty(&mut self, penalty: f32) -> &mut Self {
        if !(-2.0..=2.0).contains(&penalty) {
            panic!("Frequency penalty must be between -2.0 and 2.0");
        }
        self.frequency_penalty = penalty;
        self
    }

    /// The model will generate *num* completions server-side and return *n*
    /// completions (where *n* refers to the parameter set by `n()`) with the
    /// highest log probability per token.
    ///
    /// cf. [documentation](https://platform.openai.com/docs/api-reference/completions/create#completions/create-best_of)
    pub fn best_of(&mut self, num: u32) -> &mut Self {
        self.best_of = num;
        self
    }

    /// Influence how likely or unlikey it is that the model will choose a given
    /// token. (Note: token != word. For example, the word "Ninny" is tokenized to
    /// \[36091, 3281\] where 36091 represents 'Nin' and 3281 represents 'ny'. You
    /// can use the [tokenizer](https://platform.openai.com/tokenizer?view=bpe)
    /// tool provided by OpenAI to see how a word is tokenized.)
    ///
    /// Range should be from 100.0 to -100.0. However, selecting values at the
    /// extremes may lead to incoherent completions.
    ///
    /// cf. [documentation](https://platform.openai.com/docs/api-reference/completions/create#completions/create-logit_bias)
    ///
    /// # Example
    /// 100.0 does not make it certain that a token will occur, nor does -100.0 make
    /// it certain that a token will *not* occur.
    /// ```rust,no_run
    /// // ...
    /// let mut try_ban = HashMap::new();
    /// try_ban.insert("9703", -100.0);  // 9703: "dog"
    /// try_ban.insert("22242", -100.0); // 22242: "dogs"
    /// try_ban.insert("32942", -100.0); // 32942: "Dog"
    ///
    /// let model = CompletionModel::TextDavinci003;
    /// let prompt = "What sort of animal barks?";
    ///
    /// let response = client.completion_create(model)
    ///     .prompt(prompt)
    ///     .max_tokens(100)
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
    /// let model = CompletionModel::TextDavinci003;
    /// let prompt = "What sort of animal barks?";
    ///
    /// let response = client.completion_create(model)
    ///     .prompt(prompt)
    ///     .max_tokens(100)
    ///     .logit_bias(try_ban)
    ///     .send()
    ///     .await
    /// {
    ///     Ok(response) => response, // Response was "catcatcatcatcatcatcatcatcatcatcatcat"
    ///     Err(e) => panic!("Error: {}", e),
    /// };
    /// ```
    pub fn logit_bias(&mut self, logit_bias: HashMap<String, f32>) -> &mut Self {
        self.logit_bias = Some(logit_bias);
        self
    }

    /// "A unique identifier representing your end-user, which can help OpenAI to
    /// monitor and detect abuse." -
    /// [OpenAI API docs](https://platform.openai.com/docs/api-reference/completions/create#completions/create-user)
    pub fn user<U: Into<String> + std::fmt::Debug>(&mut self, user: U) -> &mut Self {
        self.user = Some(user.into());
        self
    }

    pub async fn send(&self) -> Result<reqwest::Response, OairsError> {
        let json = serde_json::to_value(self).unwrap();
        handle_request(
            &self.key,
            &self.url,
            client::HttpMethod::Post,
            Some(json),
            None,
        )
        .await
    }
}

/// Used for the `logprobs` parameter to the completions endpoint.
///
/// "Include the log probabilities on the `logprobs` most likely tokens, as well the
/// chosen tokens. For example, if logprobs is [[LogProbs::Five]], the API will
/// return a list of the 5 most likely tokens. The API will always return the
/// `logprob` of the sampled token, so there may be up to `logprobs+1` elements in
/// the  response." -
/// [OpenAI API docs](https://platform.openai.com/docs/api-reference/completions/create#completions/create-logprobs)
#[derive(Serialize)]
pub enum LogProbs {
    Zero,
    One,
    Two,
    Three,
    Four,
    Five,
}
impl LogProbs {
    pub fn to_int(&self) -> u8 {
        match self {
            LogProbs::Zero => 0,
            LogProbs::One => 1,
            LogProbs::Two => 2,
            LogProbs::Three => 3,
            LogProbs::Four => 4,
            LogProbs::Five => 5,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct LogitBias {
    pub token: u32,
    pub value: i32,
}
impl LogitBias {
    pub fn new(token: u32, value: i32) -> Self {
        if !(-100..=100).contains(&value) {
            panic!("Bias value must be between -100.0 and 100.0");
        }
        LogitBias { token, value }
    }
}
impl std::fmt::Display for LogitBias {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{{\"{}\": {}}}", self.token, self.value)
    }
}
