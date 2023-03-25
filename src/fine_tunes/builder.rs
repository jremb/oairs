use crate::client::{handle_request, HttpMethod};

use super::*;

#[derive(Default, Serialize)]
pub struct FineTunesBuilder<'a, State = Buildable> {
    #[serde(skip)]
    key: String,
    #[serde(skip)]
    url: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    training_file: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    validation_file: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    n_epochs: Option<u16>,
    #[serde(skip_serializing_if = "Option::is_none")]
    batch_size: Option<u16>,
    #[serde(skip_serializing_if = "Option::is_none")]
    learning_rate_multiplier: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_loss_weight: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    compute_classification_metrics: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    classification_n_classes: Option<u16>,
    #[serde(skip_serializing_if = "Option::is_none")]
    classification_positive_class: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    classification_betas: Option<Vec<f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    suffix: Option<&'a str>,

    #[serde(skip)]
    state: PhantomData<State>,
}

impl<'a> FineTunesBuilder<'a, Buildable> {
    /// Fine-tune a model based on a training file from a file that you've uploaded to the server.
    ///
    /// # Arguments:
    /// * `training_file_id`: the id of a file that has *already been uploaded* to OpenAI's servers (cf. the files endpoint).
    ///
    /// # Returns:
    /// `FineTunesBuilder<'a, Create>` that can be used to set optional parameters and execute the request by awaiting `send()`.
    ///
    /// Awaiting `send()` will return `Result<reqwest::Response, OairsError>`. A successful response can be deserialized into a
    /// [`FineTuneObject`].
    ///
    /// # Optional parameters:
    /// The following parameters are optional and can be set with the corresponding methods:
    /// * `validation_file`
    /// * `model`
    /// * `n_epochs`
    /// * `batch_size`
    /// * `learning_rate_multiplier`
    /// * `prompt_loss_weight`
    /// * `compute_classification_metrics`
    /// * `classification_n_classes`
    /// * `classification_positive_class`
    /// * `classification_betas`
    /// * `suffix`
    ///
    /// # Example:
    /// ```rust,no_run
    /// let key = std::env::var("OPENAI_API_KEY").unwrap();
    /// let client = Client::new(key);
    ///
    /// // Id of a file that has been previously uploaded
    /// let training_file_id = "file-ERwBFft4Iywvy7KeWcz0Mscs";
    /// let response = match client.create_fine_tune(training_file_id)
    ///     .n_epochs(1)
    ///     .post()
    ///     .await {
    ///         Ok(r) => r.json::<FineTuneObject>().await.unwrap(),
    ///         Err(e) => panic!("{e}"),
    ///     };
    ///
    /// let ft_id = response.id;
    /// let filename = format!("ft_results/{}", ft_id);
    /// response.save_json(&filename).unwrap();
    /// ```
    pub(crate) fn create<K: Into<String>>(
        key: K,
        training_file_id: &'a str,
    ) -> FineTunesBuilder<'a, Sendable> {
        FineTunesBuilder {
            key: key.into(),
            url: create_ft_url().to_string(),
            training_file: Some(training_file_id),
            ..Default::default()
        }
    }
}

impl<'a> FineTunesBuilder<'a, Sendable> {
    /// The id of a file *that has already been uploaded* to OpenAI's servers (cf. the files endpoint).
    ///
    /// "If you provide this file, the data is used to generate validation metrics periodically during
    /// fine-tuning. These metrics can be viewed in the fine-tuning results file. Your train and
    /// validation data should be mutually exclusive.
    ///
    /// Your dataset must be formatted as a JSONL file, where each validation example is a JSON object
    /// with the keys 'prompt' and 'completion'. Additionally, you must upload your file with the [[`Purpose::FineTune`]]" -
    /// [OpenAI API docs](https://platform.openai.com/docs/api-reference/fine-tunes/create#fine-tunes/create-validation_file)
    pub fn validation_file(&mut self, id: &'a str) -> &mut Self {
        self.validation_file = Some(id);
        self
    }

    /// The model to fine-tune.
    ///
    /// Available models are variants of the [`FineTuneModel`] enum:
    ///  * `FineTuneModel::Babbage`
    ///  * `FineTuneModel::Curie` (default if not specified)
    ///  * `FineTuneModel::Davinci`
    pub fn model(&mut self, model: &'a FineTuneModel) -> &mut Self {
        self.model = Some(model.to_str());
        self
    }

    /// "The number of epochs to train the model for. An epoch refers to one full cycle through the
    /// training dataset." -
    /// [OpenAI API docs](https://platform.openai.com/docs/api-reference/fine-tunes/create#fine-tunes/create-n_epochs)
    pub fn n_epochs(&mut self, n: u16) -> &mut Self {
        self.n_epochs = Some(n);
        self
    }

    /// "The batch size to use for training. The batch size is the number of training examples used to
    /// train a single forward and backward pass.
    ///
    /// By default, the batch size will be dynamically configured to be ~ `0.2`% of the number of examples
    /// in the training set, capped at `256` - in general, we've found that larger batch sizes tend to
    /// work better for larger datasets." -
    /// [OpenAI API docs](https://platform.openai.com/docs/api-reference/fine-tunes/create#fine-tunes/create-batch_size)
    pub fn batch_size(&mut self, batch_size: u16) -> &mut Self {
        self.batch_size = Some(batch_size);
        self
    }

    /// "The learning rate multiplier to use for training. The fine-tuning learning rate is the original
    /// learning rate used for pretraining multiplied by this value.
    ///
    /// By default, the learning rate multiplier is the `0.05`, `0.1`, or `0.2` depending on final
    /// batch_size (larger learning rates tend to perform better with larger batch sizes). We recommend
    /// experimenting with values in the range `0.02` to `0.2` to see what produces the best results." -
    /// [OpenAI API docs](https://platform.openai.com/docs/api-reference/fine-tunes/create#fine-tunes/create-learning_rate_multiplier)
    pub fn learning_rate_multiplier(&mut self, learning_rate: f32) -> &mut Self {
        self.learning_rate_multiplier = Some(learning_rate);
        self
    }

    /// "The weight to use for loss on the prompt tokens. This controls how much the model tries to learn
    /// to generate the prompt (as compared to the completion which always has a weight of `1.0`), and can
    /// add a stabilizing effect to training when completions are short.
    ///
    /// If prompts are extremely long (relative to completions), it may make sense to reduce this weight
    /// so as to avoid over-prioritizing learning the prompt." -
    /// [OpenAI API docs](https://platform.openai.com/docs/api-reference/fine-tunes/create#fine-tunes/create-prompt_loss_weight)
    pub fn prompt_loss_weight(&mut self, loss_weight: f32) -> &mut Self {
        self.prompt_loss_weight = Some(loss_weight);
        self
    }

    /// "\[C\]alculate classification-specific metrics such as accuracy and F-1 score using the
    /// validation set at the end of every epoch. These metrics can be viewed in the results file.
    ///
    /// In order to compute classification metrics, you must provide a [`validation_file`]. Additionally,
    /// you must specify classification_n_classes for multiclass classification or
    /// classification_positive_class for binary classification." -
    /// [OpenAI API docs](https://platform.openai.com/docs/api-reference/fine-tunes/create#fine-tunes/create-compute_classification_metrics)
    pub fn compute_classification_metrics(&mut self, truth_value: bool) -> &mut Self {
        self.compute_classification_metrics = Some(truth_value);
        self
    }

    /// "The number of classes in a classification task.
    ///
    /// This parameter is required for multiclass classification." -
    /// [OpenAI API docs](https://platform.openai.com/docs/api-reference/fine-tunes/create#fine-tunes/create-classification_n_classes)
    pub fn classification_n_classes(&mut self, n: u16) -> &mut Self {
        self.classification_n_classes = Some(n);
        self
    }

    /// Provide the label which is a positive marker for your class.
    ///
    /// "The positive class in binary classification.
    ///
    /// This parameter is needed to generate precision, recall, and F1 metrics when doing binary
    /// classification." -
    /// [OpenAI API docs](https://platform.openai.com/docs/api-reference/fine-tunes/create#fine-tunes/create-classification_positive_class)
    pub fn classification_positive_class(&mut self, label: &'a str) -> &mut Self {
        self.classification_positive_class = Some(label);
        self
    }

    /// "If this is provided, we calculate F-beta scores at the specified beta values. The F-beta score
    /// is a generalization of F-1 score. This is only used for binary classification.
    ///
    /// With a beta of 1 (i.e. the F-1 score), precision and recall are given the same weight. A larger
    /// beta score puts more weight on recall and less on precision. A smaller beta score puts more weight
    /// on precision and less on recall." -
    /// [OpenAI API docs](https://platform.openai.com/docs/api-reference/fine-tunes/create#fine-tunes/create-classification_betas)
    pub fn classification_betas(&mut self, betas: Vec<f32>) -> &mut Self {
        self.classification_betas = Some(betas);
        self
    }

    /// "A string of up to `40` characters that will be added to your fine-tuned model name.
    ///
    /// For example, a suffix of \"custom-model-name\" would produce a model name like
    /// `ada:ft-your-org:custom-model-name-2022-02-15-04-21-04`." -
    /// [OpenAI API docs](https://platform.openai.com/docs/api-reference/fine-tunes/create#fine-tunes/create-suffix)
    pub fn suffix(&mut self, suffix: &'a str) -> &mut Self {
        if suffix.len() > 40 {
            panic!("Suffix must be no more than 40 characters");
        }
        self.suffix = Some(suffix);
        self
    }

    pub async fn send(&self) -> Result<reqwest::Response, OairsError> {
        let json = serde_json::to_value(&self).unwrap();
        handle_request(&self.key, &self.url, HttpMethod::Post, Some(json), None).await
    }
}

#[derive(Default, Serialize)]
pub struct ListEventsBuilder<State = Buildable> {
    #[serde(skip)]
    key: String,
    #[serde(skip)]
    url: String,
    stream: bool,
    #[serde(skip)]
    state: PhantomData<State>,
}

impl ListEventsBuilder<Buildable> {
    pub fn new(key: &str, ft_id: &str) -> ListEventsBuilder<Sendable> {
        ListEventsBuilder {
            key: key.to_string(),
            url: list_ft_events_url(ft_id).to_string(),
            stream: false,
            state: PhantomData::<Sendable>,
        }
    }
}

impl ListEventsBuilder<Sendable> {
    pub fn stream(&mut self, stream: bool) -> &mut Self {
        self.stream = stream;
        self
    }
}

impl_get!(ListEventsBuilder<Sendable>);
