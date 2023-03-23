use super::*;

/// The format of the response. Either `url` or `base64` json.
/// The default is `url`.
#[derive(Debug, Default, Deserialize)]
pub enum ResponseFormat {
    #[default]
    Url,
    Base64,
}

impl Serialize for ResponseFormat {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.to_str())
    }
}

impl ResponseFormat {
    pub fn to_str(&self) -> &str {
        match self {
            ResponseFormat::Url => "url",
            ResponseFormat::Base64 => "b64_json",
        }
    }
}

#[derive(Debug, Serialize, Deserialize, SaveJson)]
pub struct Image {
    created: u64,
    data: Vec<FormattedImage>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct FormattedImage {
    #[serde(alias = "b64_json", alias = "url")]
    pub frmt: String,
}
