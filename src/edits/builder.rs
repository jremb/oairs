use super::*;

#[derive(Debug, Serialize, Deserialize)]
pub struct EditBuilder<State = Buildable> {
    #[serde(skip)]
    key: String,
    #[serde(skip)]
    url: &'static str,
    model: EditModel,
    #[serde(skip_serializing_if = "Option::is_none")]
    input: Option<String>,
    instruction: String,
    n: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<Temperature>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<TopP>,
    #[serde(skip)]
    state: std::marker::PhantomData<State>,
}

impl EditBuilder<Buildable> {
    pub fn create<K, I>(key: K, model: EditModel, instruction: I) -> EditBuilder<Sendable>
    where
        K: Into<String>,
        I: Into<String>,
    {
        EditBuilder {
            key: key.into(),
            url: URL.get(&Uri::Edits).unwrap(),
            model,
            input: None,
            instruction: instruction.into(),
            n: 1,
            temperature: None,
            top_p: None,
            state: std::marker::PhantomData,
        }
    }
}

impl EditBuilder<Sendable> {
    pub fn input<S: Into<String>>(&mut self, input: S) -> &mut Self {
        self.input = Some(input.into());
        self
    }

    pub fn n(&mut self, n: usize) -> &mut Self {
        self.n = n;
        self
    }

    pub fn temperature(&mut self, temperature: Temperature) -> &mut Self {
        self.temperature = Some(temperature);
        self
    }

    pub fn top_p(&mut self, top_p: TopP) -> &mut Self {
        self.top_p = Some(top_p);
        self
    }
}

impl_post!(EditBuilder<Sendable>, ContentType::Json);
