use super::*;

pub use chat_response::*;
pub use completion_response::*;

/// Used by the ChatCompletion, `Completion` and `Embedding` structs.
#[derive(Debug, Serialize, Clone, Copy, Deserialize, SaveJson)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: Option<usize>,
    pub total_tokens: usize,
}

mod completion_response {
    use super::*;

    /// Struct representing a successful response from the OpenAI API completions endpoint.
    ///
    /// # Example
    /// ```rust
    /// let model = CompletionModel::TextDavinci003;
    /// // Response will be of type `Completion` if successful
    /// let response = match client.completion_create(model)
    ///     .prompt("This is a test.")
    ///     .max_tokens(10)
    ///     .send()
    ///     .await
    /// {
    ///        Ok(r) => r,
    ///        Err(e) => panic!("Error: {}", e),
    /// };
    /// // ...
    /// match response.save("some_filename") {
    ///    Ok(_) => println!("Saved!"),
    ///    Err(e) => panic!("Error: {}", e),
    /// }=
    /// ```
    #[derive(Debug, Serialize, Deserialize, SaveJson)]
    pub struct Completion {
        pub id: String,
        pub object: String,
        pub created: u64,
        pub model: String,
        pub choices: Vec<Choice>,
        pub usage: Usage,
    }

    /// Substruct of the [`Completion`] struct, used for deserializing the `choices` field in a response from the completions endpoint.
    #[derive(Debug, Serialize, Deserialize, SaveJson)]
    pub struct Choice {
        pub text: String,
        pub index: u32,
        pub logprobs: Option<LogProbsResult>,
        pub finish_reason: String,
    }

    #[derive(Debug, Serialize, Deserialize, SaveJson)]
    pub struct LogProbsResult {
        pub tokens: Vec<String>,
        pub token_logprobs: Vec<f32>,
        pub top_logprobs: Vec<HashMap<String, f32>>,
        pub text_offset: Vec<u32>,
    }
}

mod chat_response {
    use crate::tokenizers::{tokenize, Tokenizer};

    use super::*;

    /// For representing a successful response from the `...chat/completions`
    /// endpoint.
    #[derive(Debug, Serialize, Deserialize, SaveJson)]
    pub struct ChatCompletion {
        pub input: Option<String>,
        pub id: String,
        pub object: String,
        pub created: u64,
        pub model: Option<String>,
        pub choices: Vec<ChatChoice>,
        pub usage: Usage,
    }

    impl ChatCompletion {
        /// Returns only the first response message.
        pub fn response_message(&self) -> Msg {
            self.choices.iter().take(1).next().unwrap().message.clone()
        }

        /// Returns all response messages.
        pub fn get_messages(&self) -> Vec<Msg> {
            self.choices
                .iter()
                .map(|choice| choice.message.clone())
                .collect()
        }
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub struct ChatChoice {
        pub index: u8,
        #[serde(alias = "message", alias = "msg")]
        pub message: Msg,
        pub finish_reason: Option<String>,
    }

    /// For streaming
    #[derive(Debug, Serialize, Deserialize)]
    pub struct ChatCompletionChunk {
        pub id: String,
        pub object: String,
        pub created: u64,
        pub model: String,
        pub choices: Vec<StreamedChoice>,
    }

    impl ChatCompletionChunk {
        // A ChatCompletion contains a usage field, which is a map of the number of
        // tokens used for the prompt, the number of tokens used for the response,
        // and the total tokens used. In this library we represent this as a `Usage`
        // struct. But this field is absent in a streamed response (which is given
        // the object value of `chat.completion.chunk` by the API). Chat completion
        // chunks aren't very useful in isolation and saving the collection of
        // chunks that make up the whole would be inefficient.
        //
        // Thus, if we are going to provide a convenience method for users to
        // convert a ChatCompletionChunk into a ChatCompletion, we could either (1)
        // have them build the `Usage` field themselves, (2) have them hold onto
        // the original prompt, pass either it to this method, and we build it here
        // while also adding the original prompt to the optional `input` field, or
        // (3) have them calculate the token usage of the prompt and pass that to
        // this method. But if we do (3) we are already most of the way to (2),
        // which is most convenient for the user. And so that's what we do...
        //
        /// A Convenience method for converting a `ChatCompletionChunk` (identified
        /// as a `chat.completion.chunk` by the API) into a `ChatCompletion`
        /// (`chat.completion`). Since a `chat.completion.chunk` does not contain
        /// the `usage` field that is present in a `chat.completion`, and
        /// constructing that field requires the number of tokens used in the
        /// prompt, we require the user to pass in the original prompt and we use
        /// that to construct the `usage` field and fill in the optional `input`
        /// field as well.
        ///
        /// The field `object` will still be given the value `chat.completion.chunk`
        /// to indicate that this was a streamed response.
        ///
        /// # Arguments
        /// * response_message_content - The value of the `delta.message` field in
        /// the `chat.completion.chunk` object. Or what would be the value of the
        /// message.content field in a `chat.completion` object.
        /// * prompt - The original prompt (message) that was used to generate the
        /// `chat.completion.chunk` object.
        pub fn to_chat_response(
            self,
            response_message_content: String,
            prompt: String,
        ) -> ChatCompletion {
            // TODO: Handle potential error in tokenization
            let prompt_token_usage = tokenize(&prompt, Tokenizer::CL100KBase).unwrap().len();
            let response_token_usage = tokenize(&response_message_content, Tokenizer::CL100KBase)
                .unwrap()
                .len();
            let total_tokens_used = prompt_token_usage + response_token_usage;
            let usage = Usage {
                prompt_tokens: prompt_token_usage,
                completion_tokens: Some(response_token_usage),
                total_tokens: total_tokens_used,
            };

            // Get choices
            dbg!(&self);
            let streamed_choices = self.choices;
            let mut choices = Vec::new();
            // TODO Fix this after checking n + 1
            // TODO: Update, after trying to stream with n + 1 got server error half way through, may not be practical (or easy solution to this). For now, should discourage use of streaming with n > 1. (3/18/2023)
            for choice in streamed_choices {
                let msg = Msg::Assistant(response_message_content.clone());
                let index = choice.index;
                let finish_reason = choice.finish_reason;
                let choice = ChatChoice {
                    index,
                    message: msg,
                    finish_reason,
                };
                choices.push(choice);
            }

            ChatCompletion {
                input: Some(prompt),
                id: self.id,
                object: self.object,
                created: self.created,
                model: Some(self.model),
                choices,
                usage,
            }
        }
    }

    #[derive(Debug, Default, Serialize, Deserialize)]
    pub struct ChatCompletionChunks {
        pub data: Vec<ChatCompletionChunk>,
    }

    impl ChatCompletionChunks {
        pub fn new() -> Self {
            ChatCompletionChunks::default()
        }

        pub fn push(&mut self, part: ChatCompletionChunk) {
            self.data.push(part);
        }

        pub fn extend(&mut self, parts: Vec<ChatCompletionChunk>) {
            self.data.extend(parts);
        }

        pub fn is_empty(&self) -> bool {
            self.data.is_empty()
        }
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub struct StreamedChoice {
        pub delta: MessagePart,
        pub index: u8,
        pub finish_reason: Option<String>,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub struct MessagePart {
        #[serde(alias = "content", alias = "role")]
        pub msg: Option<String>,
    }
}
