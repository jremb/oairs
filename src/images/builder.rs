use super::{response::ResponseFormat, *};

#[derive(Debug, Serialize, Deserialize)]
pub struct ImageBuilder<State = Buildable> {
    #[serde(skip)]
    key: String,
    #[serde(skip)]
    url: String,
    prompt: String,
    n: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    size: Option<ImageSize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<ResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    user: Option<String>,
    #[serde(skip)]
    state: std::marker::PhantomData<State>,
}

impl ImageBuilder<Buildable> {
    pub fn create<K, P>(key: K, prompt: P) -> ImageBuilder<Sendable>
    where
        K: Into<String>,
        P: Into<String>,
    {
        ImageBuilder {
            key: key.into(),
            url: format!("{}/generations", URL.get(&Uri::Images).unwrap()),
            prompt: prompt.into(),
            n: 1,
            size: None,
            response_format: None,
            user: None,
            state: std::marker::PhantomData,
        }
    }
}

impl ImageBuilder<Sendable> {
    pub fn n(&mut self, n: usize) -> &mut Self {
        if !(1..=10).contains(&n) {
            panic!("n must be between 1 and 10");
        }
        self.n = n;
        self
    }

    pub fn size(&mut self, size: ImageSize) -> &mut Self {
        self.size = Some(size);
        self
    }

    pub fn response_format(&mut self, format: ResponseFormat) -> &mut Self {
        self.response_format = Some(format);
        self
    }

    pub fn user(&mut self, user: String) -> &mut Self {
        self.user = Some(user);
        self
    }
}

impl_post!(ImageBuilder<Sendable>, ContentType::Json);

#[derive(Debug, Default, Deserialize)]
pub enum ImageSize {
    /// 256x256
    Small,
    /// 512x512
    Medium,
    #[default]
    /// 1024x1024 (default)
    Large,
}

impl std::fmt::Display for ImageSize {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.to_str())
    }
}

impl Serialize for ImageSize {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.to_str())
    }
}

impl ImageSize {
    pub fn to_str(&self) -> &str {
        match self {
            ImageSize::Small => "256x256",
            ImageSize::Medium => "512x512",
            ImageSize::Large => "1024x1024",
        }
    }
}
