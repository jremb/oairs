pub use builder::*;

use super::*;

mod builder {
    use crate::client::handle_request;

    use super::*;

    #[derive(Debug, Default, Serialize, Deserialize)]
    pub struct ModerationBuilder<State = Buildable> {
        #[serde(skip)]
        key: String,
        #[serde(skip)]
        url: &'static str,
        model: ModerationModel,
        #[serde(alias = "input")]
        input: Vec<String>,
        #[serde(skip)]
        state: std::marker::PhantomData<State>,
    }

    impl ModerationBuilder<Buildable> {
        pub fn create<K>(key: K, inputs: Vec<String>) -> ModerationBuilder<Sendable>
        where
            K: Into<String>,
        {
            ModerationBuilder {
                key: key.into(),
                url: URL.get(&Uri::Moderations).unwrap(),
                input: inputs,
                ..Default::default()
            }
        }
    }

    impl ModerationBuilder<Sendable> {
        pub fn model(&mut self, model: ModerationModel) -> &mut Self {
            self.model = model;
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
}

pub mod moderations_response {
    use super::*;

    /// For deserializing the OpenAI API for a Moderation request. The response doesn't
    /// include the input. To get the input as a field, use the `with_input` method.
    /// To save the response with the input, use the `save_with_input` method.
    #[derive(Debug, Serialize, Deserialize, SaveJson)]
    pub struct ModerationResult {
        pub inputs: Option<Vec<String>>,
        pub id: String,
        pub model: String,
        pub results: Vec<Moderation>,
    }

    impl ModerationResult {
        pub fn with_input(&mut self, inputs: Vec<String>) -> &mut Self {
            self.inputs = Some(inputs);
            self
        }

        pub fn save_with_input(
            &mut self,
            path: &str,
            input: Vec<String>,
        ) -> Result<(), std::io::Error> {
            let response = self.with_input(input);
            response.save_json(path)
        }
    }

    #[derive(Debug, Serialize, Deserialize, SaveJson)]
    pub struct Moderation {
        pub categories: ModerationCategories,
        pub category_scores: ModerationScores,
        pub flagged: bool,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub struct ModerationCategories {
        hate: bool,
        #[serde(rename = "hate/threatening")]
        hate_threatening: bool,
        #[serde(rename = "self-harm")]
        self_harm: bool,
        sexual: bool,
        #[serde(rename = "sexual/minors")]
        sexual_minors: bool,
        violence: bool,
        #[serde(rename = "violence/graphic")]
        violence_graphic: bool,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub struct ModerationScores {
        hate: f64,
        #[serde(rename = "hate/threatening")]
        hate_threatening: f64,
        #[serde(rename = "self-harm")]
        self_harm: f64,
        sexual: f64,
        #[serde(rename = "sexual/minors")]
        sexual_minors: f64,
        violence: f64,
        #[serde(rename = "violence/graphic")]
        violence_graphic: f64,
    }
}
