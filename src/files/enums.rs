use super::*;

// Currently the only available purpose in the API, but if more are added later...
#[non_exhaustive]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Purpose {
    FineTune,
}

impl std::default::Default for Purpose {
    fn default() -> Self {
        Purpose::FineTune
    }
}

impl std::fmt::Display for Purpose {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Purpose::FineTune => write!(f, "fine-tune"),
        }
    }
}

impl Purpose {
    pub fn to_str(&self) -> &str {
        match self {
            Purpose::FineTune => "fine-tune",
        }
    }
}
