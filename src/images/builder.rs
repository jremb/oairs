use reqwest::multipart::Form;

use crate::{
    client::{get_file_part, handle_request, HttpMethod},
    images::response::ResponseFormat,
};

use super::*;

#[doc(hidden)]
#[derive(Debug, Default, Serialize)]
struct ImageRequest {
    #[serde(skip)]
    key: String,
    #[serde(skip)]
    url: String,

    // Required for:
    //      .../images/generations
    //      .../images/edits
    //
    // But not for .../variations
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt: Option<String>,

    // Following group is optional for all .../images/... endpoints
    #[serde(skip_serializing_if = "Option::is_none")]
    n: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    size: Option<ImageSize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<ResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    user: Option<String>,
}

// impl_post!(ImageRequest);
impl ImageRequest {
    async fn send(&self) -> Result<reqwest::Response, OairsError> {
        let json = serde_json::to_value(self).unwrap();
        handle_request(&self.key, &self.url, HttpMethod::Post, Some(json), None).await
    }
}

// ========================== //
//     Type State Trackers    //
// ========================== //
#[doc(hidden)]
#[derive(Debug, Default, Serialize)]
pub struct ImageGen {}
#[doc(hidden)]
#[derive(Debug, Default, Serialize)]
pub struct ImageEdit {
    // file
    image: String,
    prompt: String,
    mask: Option<String>,
}
#[doc(hidden)]
#[derive(Debug, Default, Serialize)]
pub struct ImageVariation {
    // file
    image: String,
}
#[doc(hidden)]
#[derive(Debug, Default, Serialize)]
pub struct Keyed;

pub trait BuildableImage {}
impl BuildableImage for Keyed {}
impl BuildableImage for ImageGen {}
impl BuildableImage for ImageEdit {}
impl BuildableImage for ImageVariation {}

// ========================== //
//        ImageBuilder        //
// ========================== //

#[derive(Default)]
pub struct ImageBuilder<S> {
    key: String,
    url: String,
    // Fields common to all of the .../images/... endpoints
    // are stored here.
    state: Box<ImageRequest>,

    // other required and optional fields by .../edits and .../variations need
    // to be sent as a form, so we can store them in the state_data field
    state_data: S,
}

#[allow(dead_code)]
impl<S> ImageBuilder<S> {
    /// The number of images to generate. Must be between `1` and `10`.
    /// Panics if `n` is not in range.
    pub fn n(&mut self, n: usize) -> &mut Self {
        if !(1..=10).contains(&n) {
            panic!("n must be between 1 and 10");
        }
        self.state.n = Some(n);
        self
    }

    /// The [`ImageSize`] variant representing the size of the image to generate.
    ///
    /// Must be one of the following:
    /// - [`ImageSize::Small`]  (256x256)
    /// - [`ImageSize::Medium`] (512x512)
    /// - [`ImageSize::Large`]  (1024x1024)
    pub fn size(&mut self, size: ImageSize) -> &mut Self {
        self.state.size = Some(size);
        self
    }

    /// [`ResponseFormat`] variant representing the format of the response.
    ///
    /// Either [`ResponseFormat::Url`] or [`ResponseFormat::Base64`] json.
    pub fn response_format(&mut self, format: ResponseFormat) -> &mut Self {
        self.state.response_format = Some(format);
        self
    }

    /// Unique identifier that can be used to prevent abuse.
    pub fn user(&mut self, user: String) -> &mut Self {
        self.state.user = Some(user);
        self
    }
}

impl ImageBuilder<Keyed> {
    pub fn create_image<K, P>(key: K, prompt: P) -> ImageBuilder<ImageGen>
    where
        K: Into<String>,
        P: Into<String>,
    {
        ImageBuilder {
            state: Box::new(ImageRequest {
                key: key.into(),
                url: img_create_url(),
                prompt: Some(prompt.into()),
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    pub fn create_edit<K, I, P>(key: K, image: I, prompt: P) -> ImageBuilder<ImageEdit>
    where
        K: Into<String>,
        I: Into<String>,
        P: Into<String>,
    {
        ImageBuilder {
            key: key.into(),
            url: img_edit_url(),
            state_data: ImageEdit {
                image: image.into(),
                prompt: prompt.into(),
                mask: None,
            },
            ..Default::default()
        }
    }

    pub fn create_variation<K, I>(key: K, image: I) -> ImageBuilder<ImageVariation>
    where
        K: Into<String>,
        I: Into<String>,
    {
        ImageBuilder {
            key: key.into(),
            url: img_variation_url(),
            state_data: ImageVariation {
                image: image.into(),
            },
            ..Default::default()
        }
    }
}

impl ImageBuilder<ImageGen> {
    pub async fn send(&self) -> Result<reqwest::Response, OairsError> {
        self.state.send().await
    }
}

impl ImageBuilder<ImageEdit> {
    /// Executes a `POST` request, submitting a `form` to the API.
    pub async fn send(&self) -> Result<reqwest::Response, OairsError> {
        let file_part = get_file_part(&self.state_data.image)?;

        let mut form = Form::new()
            .part("image", file_part)
            .text("prompt", self.state_data.prompt.clone());

        if let Some(n) = self.state.n {
            form = form.text("n", n.to_string());
        }

        if let Some(m) = self.state_data.mask.clone() {
            form = form.text("mask", m);
        }

        if let Some(s) = self.state.size.clone() {
            form = form.text("size", s.to_string());
        }

        if let Some(r) = self.state.response_format.clone() {
            form = form.text("response_format", r.to_string());
        }

        if let Some(u) = self.state.user.clone() {
            form = form.text("user", u);
        }

        handle_request(&self.key, &self.url, HttpMethod::Post, None, Some(form)).await
    }
}

impl ImageBuilder<ImageVariation> {
    /// Executes a `POST` request, submitting a `form` to the API.
    pub async fn send(&self) -> Result<reqwest::Response, OairsError> {
        let file_part = get_file_part(&self.state_data.image)?;

        let mut form = Form::new().part("image", file_part);

        if let Some(n) = self.state.n {
            form = form.text("n", n.to_string());
        }

        if let Some(s) = self.state.size.clone() {
            form = form.text("size", s.to_string());
        }

        if let Some(r) = self.state.response_format.clone() {
            form = form.text("response_format", r.to_string());
        }

        if let Some(u) = self.state.user.clone() {
            form = form.text("user", u);
        }

        handle_request(&self.key, &self.url, HttpMethod::Post, None, Some(form)).await
    }
}

#[derive(Clone, Debug, Default, Deserialize)]
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
