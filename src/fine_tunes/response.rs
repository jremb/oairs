use crate::files::response::FileInfo;

use super::*;

/// Describes a fine-tune object, which can be a response
/// from a request to fine-tune a model or part of a response from a list
/// fine-tunes or list fine-tune events request.
#[derive(Debug, Serialize, Deserialize, SaveJson)]
pub struct FineTuneInfo {
    pub object: String,
    pub id: String,
    pub model: String,
    pub created_at: u64,
    // events field will be present in a request to list fine
    // tune events, but not in a request to list fine tunes.
    pub events: Option<Vec<Event>>,
    // Can be null if failed
    pub fine_tuned_model: Option<String>,
    pub hyperparams: Option<Hyperparams>,
    pub organization_id: String,
    pub status: String,
    pub training_files: Vec<FileInfo>,
    pub validation_files: Option<Vec<FileInfo>>,
    pub result_files: Option<Vec<FileInfo>>,
    pub updated_at: u64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Event {
    pub object: String,
    pub created_at: u64,
    pub level: String,
    pub message: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Hyperparams {
    pub n_epochs: u32,
    pub batch_size: Option<u32>,
    pub prompt_loss_weight: f32,
    pub learning_rate_multiplier: Option<f32>,
}

#[derive(Debug, Serialize, Deserialize, SaveJson)]
pub struct FineTunesList {
    pub object: String, // Will be "list"
    pub data: Vec<FineTuneInfo>,
}

impl FineTunesList {
    pub fn load_from_file(path: &str) -> Result<Self, OairsError> {
        let file = std::fs::File::open(path)?;
        let reader = std::io::BufReader::new(file);
        let list = match serde_json::from_reader(reader) {
            Ok(list) => list,
            Err(e) => Err(OairsError::new(
                e.to_string(),
                ErrorType::DeserializationError,
                Some(path.to_string()),
                None,
            ))?,
        };
        Ok(list)
    }
}

/// Struct for deserializing a [`Response`](reqwest::Response) from a request to list
/// fine-tune events.
#[derive(Debug, Serialize, Deserialize, SaveJson)]
pub struct EventList {
    pub object: String, // Will be "list"
    pub data: Vec<Event>,
}
