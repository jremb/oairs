//! Not every model is represented as an Enum variant. This is due to several factors: some
//! ambiguity in the API documentation, where it's not entirely clear which models are accepted
//! by which endpoints, some models being marked for depreciation, and custom/fine-tuned models.
//!
//! In order to accommodate these cases, there are two macros, each of which (currently) does the
//! same thing, but which suggest a different use case conceptually: [`custom_model!`] and
//! [`ft_model!`]. See the documentation for those macros for more information.

use super::*;

// Some of the models have a default, where I think there's an obvious choice. Default is
// implement instead of deriving it because the enums are non-exhaustive.

// ========================== //
//        AudioModel          //
// ========================== //

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub enum AudioModel {
    Whisper1,
}

impl std::fmt::Display for AudioModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            AudioModel::Whisper1 => write!(f, "whisper-1"),
        }
    }
}

impl std::str::FromStr for AudioModel {
    type Err = OairsError;

    #[track_caller]
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "whisper-1" => Ok(AudioModel::Whisper1),
            _ => Err(OairsError::new(
                format!("No AudioModel variant: {s}"),
                ErrorType::DeserializationError,
                Some(s.to_string()),
                None,
            )),
        }
    }
}

impl Serialize for AudioModel {
    fn serialize<Ser>(&self, serializer: Ser) -> Result<Ser::Ok, Ser::Error>
    where
        Ser: Serializer,
    {
        serializer.serialize_str(self.to_str())
    }
}

impl RetrievableModel for AudioModel {
    fn to_str(&self) -> &str {
        match self {
            AudioModel::Whisper1 => "whisper-1",
        }
    }
}

// ========================== //
//        EditModel           //
// ========================== //

/// For models that can be used by the `.../v1/edits` endpoint.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub enum EditModel {
    CodeDavinci002, // codex
    TextDavinciEdit001,
    // TODO compatible with edits?
    CodeCushman001, // codex
}

impl std::fmt::Display for EditModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            EditModel::CodeDavinci002 => write!(f, "code-davinci-002"),
            EditModel::CodeCushman001 => write!(f, "code-cushman-001"),
            EditModel::TextDavinciEdit001 => write!(f, "text-davinci-edit-001"),
        }
    }
}

impl std::str::FromStr for EditModel {
    type Err = OairsError;

    #[track_caller]
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "code-davinci-002" => Ok(EditModel::CodeDavinci002),
            "code-cushman-001" => Ok(EditModel::CodeCushman001),
            "text-davinci-edit-001" => Ok(EditModel::TextDavinciEdit001),
            _ => Err(OairsError::new(
                format!("No ModelEditsv1 variant: {s}"),
                ErrorType::DeserializationError,
                Some(s.to_string()),
                None,
            )),
        }
    }
}

impl Serialize for EditModel {
    fn serialize<Ser>(&self, serializer: Ser) -> Result<Ser::Ok, Ser::Error>
    where
        Ser: serde::Serializer,
    {
        serializer.serialize_str(self.to_str())
    }
}

impl RetrievableModel for EditModel {
    fn to_str(&self) -> &str {
        match *self {
            EditModel::CodeDavinci002 => "code-davinci-002",
            EditModel::CodeCushman001 => "code-cushman-001",
            EditModel::TextDavinciEdit001 => "text-davinci-edit-001",
        }
    }
}

impl EditModel {
    /// For convenience of, e.g., iterating over all models: `for m in EditModel::ALL.iter()`
    /// or to get vector of all models: `EditModel::ALL.to_vec()`.
    pub const ALL: [EditModel; 3] = [
        EditModel::CodeDavinci002,
        EditModel::CodeCushman001,
        EditModel::TextDavinciEdit001,
    ];
}

// ========================== //
//      ChatModel             //
// ========================== //

/// For models that can be used by the `.../v1/chat/completions` endpoint.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Deserialize)]
pub enum ChatModel {
    GptTurbo,
    GptTurbo0301,
    Gpt4,
    Gpt40314,
}

impl std::default::Default for ChatModel {
    fn default() -> Self {
        ChatModel::GptTurbo
    }
}

impl std::fmt::Display for ChatModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChatModel::GptTurbo => write!(f, "gpt-3.5-turbo"),
            ChatModel::GptTurbo0301 => write!(f, "gpt-3.5-turbo-0301"),
            ChatModel::Gpt4 => write!(f, "gpt-4"),
            ChatModel::Gpt40314 => write!(f, "gpt-4-0314"),
        }
    }
}

impl std::str::FromStr for ChatModel {
    type Err = OairsError;

    #[track_caller]
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "gpt-3.5-turbo" => Ok(ChatModel::GptTurbo),
            "gpt-3.5-turbo-0301" => Ok(ChatModel::GptTurbo0301),
            "gpt-4" => Ok(ChatModel::Gpt4),
            "gpt-4-0314" => Ok(ChatModel::Gpt40314),
            _ => Err(OairsError::new(
                format!("No ModelChatCompletionsv1 variant: {s}"),
                ErrorType::DeserializationError,
                Some(s.to_string()),
                None,
            )),
        }
    }
}

impl Serialize for ChatModel {
    fn serialize<Ser>(&self, serializer: Ser) -> Result<Ser::Ok, Ser::Error>
    where
        Ser: serde::Serializer,
    {
        serializer.serialize_str(self.to_str())
    }
}

impl RetrievableModel for ChatModel {
    fn to_str(&self) -> &str {
        match self {
            ChatModel::GptTurbo => "gpt-3.5-turbo",
            ChatModel::GptTurbo0301 => "gpt-3.5-turbo-0301",
            ChatModel::Gpt4 => "gpt-4",
            ChatModel::Gpt40314 => "gpt-4-0314",
        }
    }
}

impl ChatModel {
    /// For convenience of, e.g., iterating over all models: `for m in ChatModel::ALL.iter()`
    /// or to get vector of all models: `EditModel::ALL.to_vec()`
    pub const ALL: [ChatModel; 4] = [
        ChatModel::GptTurbo,
        ChatModel::GptTurbo0301,
        ChatModel::Gpt4,
        ChatModel::Gpt40314,
    ];
}

// ========================== //
//     CompletionModel     //
// ========================== //

/// For models that can be used by the `.../v1/completions` endpoint.
/// Default is `CompletionModel::TextDavinci003`.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
pub enum CompletionModel {
    #[default]
    TextDavinci003,
    TextDavinci002,
    TextCurie001,
    TextBabbage001,
    TextAda001,
    Davinci,
    Curie,
    Babbage,
    Ada,
}

impl std::fmt::Display for CompletionModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompletionModel::TextDavinci003 => write!(f, "text-davinci-003"),
            CompletionModel::TextDavinci002 => write!(f, "text-davinci-002"),
            CompletionModel::TextCurie001 => write!(f, "text-curie-001"),
            CompletionModel::TextBabbage001 => write!(f, "text-babbage-001"),
            CompletionModel::TextAda001 => write!(f, "text-ada-001"),
            CompletionModel::Davinci => write!(f, "davinci"),
            CompletionModel::Curie => write!(f, "curie"),
            CompletionModel::Babbage => write!(f, "babbage"),
            CompletionModel::Ada => write!(f, "ada"),
        }
    }
}

impl std::str::FromStr for CompletionModel {
    type Err = OairsError;

    #[track_caller]
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "text-davinci-003" => Ok(CompletionModel::TextDavinci003),
            "text-davinci-002" => Ok(CompletionModel::TextDavinci002),
            "text-curie-001" => Ok(CompletionModel::TextCurie001),
            "text-babbage-001" => Ok(CompletionModel::TextBabbage001),
            "text-ada-001" => Ok(CompletionModel::TextAda001),
            "davinci" => Ok(CompletionModel::Davinci),
            "curie" => Ok(CompletionModel::Curie),
            "babbage" => Ok(CompletionModel::Babbage),
            "ada" => Ok(CompletionModel::Ada),
            _ => Err(OairsError::new(
                format!("No ModelCompletionsv1 variant: {s}"),
                ErrorType::DeserializationError,
                Some(s.to_string()),
                None,
            )),
        }
    }
}

impl Serialize for CompletionModel {
    fn serialize<Ser>(&self, serializer: Ser) -> Result<Ser::Ok, Ser::Error>
    where
        Ser: serde::Serializer,
    {
        serializer.serialize_str(self.to_str())
    }
}

impl RetrievableModel for CompletionModel {
    fn to_str(&self) -> &str {
        match self {
            CompletionModel::TextDavinci003 => "text-davinci-003",
            CompletionModel::TextDavinci002 => "text-davinci-002",
            CompletionModel::TextCurie001 => "text-curie-001",
            CompletionModel::TextBabbage001 => "text-babbage-001",
            CompletionModel::TextAda001 => "text-ada-001",
            CompletionModel::Davinci => "davinci",
            CompletionModel::Curie => "curie",
            CompletionModel::Babbage => "babbage",
            CompletionModel::Ada => "ada",
        }
    }
}

impl CompletionModel {
    /// For convenience of, e.g., iterating over all models: `for m in CompletionModel::ALL.iter()`
    pub const ALL: [CompletionModel; 9] = [
        CompletionModel::TextDavinci003,
        CompletionModel::TextDavinci002,
        CompletionModel::TextCurie001,
        CompletionModel::TextBabbage001,
        CompletionModel::TextAda001,
        CompletionModel::Davinci,
        CompletionModel::Curie,
        CompletionModel::Babbage,
        CompletionModel::Ada,
    ];
}

// ========================== //
//     ModerationModel     //
// ========================== //

/// For models that can be used by the `.../v1/moderations` endpoint.
/// Default is `ModerationModel::TextModerationLatest`.
#[non_exhaustive]
#[derive(Clone, Default, Debug, PartialEq, Deserialize)]
pub enum ModerationModel {
    #[default]
    TextModerationLatest,
    TextModerationStable,
}

impl std::fmt::Display for ModerationModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModerationModel::TextModerationLatest => write!(f, "text-moderation-latest"),
            ModerationModel::TextModerationStable => write!(f, "text-moderation-stable"),
        }
    }
}

impl std::str::FromStr for ModerationModel {
    type Err = OairsError;

    #[track_caller]
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "text-moderation-latest" => Ok(ModerationModel::TextModerationLatest),
            "text-moderation-stable" => Ok(ModerationModel::TextModerationStable),
            _ => Err(OairsError::new(
                format!("No ModelModerationsv1 variant: {s}"),
                ErrorType::DeserializationError,
                Some(s.to_string()),
                None,
            )),
        }
    }
}

impl Serialize for ModerationModel {
    fn serialize<Ser>(&self, serializer: Ser) -> Result<Ser::Ok, Ser::Error>
    where
        Ser: serde::Serializer,
    {
        serializer.serialize_str(self.to_str())
    }
}

impl ModerationModel {
    pub fn to_str(&self) -> &str {
        match self {
            ModerationModel::TextModerationLatest => "text-moderation-latest",
            ModerationModel::TextModerationStable => "text-moderation-stable",
        }
    }
}

// ========================== //
//       EmbeddingModel       //
// ========================== //

/// For models that can be used by the `.../v1/embeddings` endpoint. The default
/// is `EmbeddingModel::TextEmbeddingAda002`.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
pub enum EmbeddingModel {
    TextEmbeddingAda002,
    // Listed as compatible with the /v1/embeddings endpoint in the
    // docs (https://platform.openai.com/docs/models/model-endpoint-compatability)
    TextSearchAdaDoc001,
    // First gen embedding models
    TextDavinciEmbedding001,
    TextCurieEmbedding001,
    TextBabbageEmbedding001,
    TextAdaEmbedding001,
}

impl std::default::Default for EmbeddingModel {
    fn default() -> Self {
        EmbeddingModel::TextEmbeddingAda002
    }
}

impl std::fmt::Display for EmbeddingModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EmbeddingModel::TextEmbeddingAda002 => write!(f, "text-embedding-ada-002"),
            EmbeddingModel::TextSearchAdaDoc001 => write!(f, "text-search-ada-doc-001"),
            EmbeddingModel::TextDavinciEmbedding001 => write!(f, "text-davinci-embedding-001"),
            EmbeddingModel::TextCurieEmbedding001 => write!(f, "text-curie-embedding-001"),
            EmbeddingModel::TextBabbageEmbedding001 => write!(f, "text-babbage-embedding-001"),
            EmbeddingModel::TextAdaEmbedding001 => write!(f, "text-ada-embedding-001"),
        }
    }
}

impl std::str::FromStr for EmbeddingModel {
    type Err = OairsError;

    #[track_caller]
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "text-embedding-ada-002" => Ok(EmbeddingModel::TextEmbeddingAda002),
            "text-search-ada-doc-001" => Ok(EmbeddingModel::TextSearchAdaDoc001),
            "text-davinci-embedding-001" => Ok(EmbeddingModel::TextDavinciEmbedding001),
            "text-curie-embedding-001" => Ok(EmbeddingModel::TextCurieEmbedding001),
            "text-babbage-embedding-001" => Ok(EmbeddingModel::TextBabbageEmbedding001),
            "text-ada-embedding-001" => Ok(EmbeddingModel::TextAdaEmbedding001),
            _ => Err(OairsError::new(
                format!("No ModelEmbeddingsv1 variant: {s}"),
                ErrorType::DeserializationError,
                Some(s.to_string()),
                None,
            )),
        }
    }
}

impl Serialize for EmbeddingModel {
    fn serialize<Ser>(&self, serializer: Ser) -> Result<Ser::Ok, Ser::Error>
    where
        Ser: serde::Serializer,
    {
        serializer.serialize_str(self.to_str())
    }
}

impl RetrievableModel for EmbeddingModel {
    fn to_str(&self) -> &str {
        match self {
            EmbeddingModel::TextEmbeddingAda002 => "text-embedding-ada-002",
            EmbeddingModel::TextSearchAdaDoc001 => "text-search-ada-doc-001",
            EmbeddingModel::TextDavinciEmbedding001 => "text-davinci-embedding-001",
            EmbeddingModel::TextCurieEmbedding001 => "text-curie-embedding-001",
            EmbeddingModel::TextBabbageEmbedding001 => "text-babbage-embedding-001",
            EmbeddingModel::TextAdaEmbedding001 => "text-ada-embedding-001",
        }
    }
}

impl EmbeddingModel {
    /// For convenience of, e.g., iterating over all models: `for m in EmbeddingModel::ALL.iter()`
    /// or to get vector of all models: `EditModel::ALL.to_vec()`
    pub const ALL: [EmbeddingModel; 6] = [
        EmbeddingModel::TextEmbeddingAda002,
        EmbeddingModel::TextSearchAdaDoc001,
        EmbeddingModel::TextDavinciEmbedding001,
        EmbeddingModel::TextCurieEmbedding001,
        EmbeddingModel::TextBabbageEmbedding001,
        EmbeddingModel::TextAdaEmbedding001,
    ];
}

// ========================== //
//       FineTuneModel        //
// ========================== //

/// For models that can be used by the `.../v1/fine-tunes` endpoint. The default
/// is `Davinci`, as this will generally provide the best restuls. However, note that it is
/// also the most expensive to run.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub enum FineTuneModel {
    Ada,
    Babbage,
    Curie,
    Davinci,
}

impl std::default::Default for FineTuneModel {
    fn default() -> Self {
        FineTuneModel::Davinci
    }
}

impl std::fmt::Display for FineTuneModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FineTuneModel::Ada => write!(f, "ada"),
            FineTuneModel::Babbage => write!(f, "babbage"),
            FineTuneModel::Curie => write!(f, "curie"),
            FineTuneModel::Davinci => write!(f, "davinci"),
        }
    }
}

impl std::str::FromStr for FineTuneModel {
    type Err = OairsError;

    #[track_caller]
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "ada" => Ok(FineTuneModel::Ada),
            "babbage" => Ok(FineTuneModel::Babbage),
            "curie" => Ok(FineTuneModel::Curie),
            "davinci" => Ok(FineTuneModel::Davinci),
            _ => Err(OairsError::new(
                format!("No ModelFineTunesv1 variant: {s}"),
                ErrorType::DeserializationError,
                Some(s.to_string()),
                None,
            )),
        }
    }
}

impl Serialize for FineTuneModel {
    fn serialize<Ser>(&self, serializer: Ser) -> Result<Ser::Ok, Ser::Error>
    where
        Ser: serde::Serializer,
    {
        serializer.serialize_str(self.to_str())
    }
}

impl FineTuneModel {
    /// For convenience of, e.g., iterating over all models: `for m in FineTuneModel::ALL.iter()`
    /// or to get vector of all models: `EditModel::ALL.to_vec()`
    pub const ALL: [FineTuneModel; 4] = [
        FineTuneModel::Ada,
        FineTuneModel::Babbage,
        FineTuneModel::Curie,
        FineTuneModel::Davinci,
    ];
}

impl RetrievableModel for FineTuneModel {
    fn to_str(&self) -> &str {
        match self {
            FineTuneModel::Ada => "ada",
            FineTuneModel::Babbage => "babbage",
            FineTuneModel::Curie => "curie",
            FineTuneModel::Davinci => "davinci",
        }
    }
}

pub trait RetrievableModel {
    fn to_str(&self) -> &str;
}
