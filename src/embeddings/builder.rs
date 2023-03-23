//! Contains the [`EmbeddingBuilder`] struct.

use super::*;

#[derive(Debug, Serialize, Deserialize)]
pub struct EmbeddingBuilder<State = Sendable> {
    #[serde(skip)]
    key: String,
    #[serde(skip)]
    url: &'static str,
    model: EmbeddingModel,
    input: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    user: Option<String>,
    #[serde(skip)]
    state: std::marker::PhantomData<State>,
}

impl<'a, Sendable> EmbeddingBuilder<Sendable> {
    pub fn new<K, T>(key: K, model: EmbeddingModel, inputs: &'a [T]) -> EmbeddingBuilder<Sendable>
    where
        K: Into<String>,
        T: Into<String> + std::fmt::Display,
    {
        Self {
            key: key.into(),
            url: URL.get(&Uri::Embeddings).unwrap(),
            model,
            input: inputs.iter().map(|i| i.to_string()).collect(),
            user: None,
            state: std::marker::PhantomData,
        }
    }

    pub fn user<U: Into<String> + std::fmt::Debug>(&mut self, user: U) -> &mut Self {
        self.user = Some(user.into());
        self
    }
}

impl_post!(EmbeddingBuilder<Sendable>, ContentType::Json);
