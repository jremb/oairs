use std::str::FromStr;

use tokio::{
    fs::{File, OpenOptions},
    io::AsyncWriteExt,
};

use super::*;

/// In response to a delete file request or a delete fine-tune model request.
#[derive(Debug, Serialize, Deserialize, SaveJson)]
pub struct DeleteResponse {
    pub id: String,
    pub object: String,
    pub deleted: bool,
}

#[derive(Debug, Serialize, Deserialize, SaveJson)]
pub struct FileList {
    data: Vec<FileInfo>,
    object: String,
}

#[derive(Debug, Serialize, Deserialize, SaveJson)]
pub struct FileInfo {
    pub object: String, // "file"
    pub id: String,
    pub purpose: String,
    pub filename: String,
    pub bytes: u64,
    pub created_at: u64,
    pub status: String, // e.g., "deleted"
    // Not sure what might actually go here as I haven't
    // been able to get anything back in my testing other than
    // `null`
    pub status_destails: Option<String>,
}

impl FileInfo {
    /// Method for retrieving the id of a FileResponse.
    pub fn id(&self) -> &str {
        &self.id
    }
}

/// Struct for retrieving file content of a fine-tune training file.
#[derive(Debug, Serialize)]
pub struct FineTuneFC {
    pub data: Vec<PromptCompletion>,
}

impl Default for FineTuneFC {
    fn default() -> Self {
        Self::new()
    }
}

impl FineTuneFC {
    pub fn new() -> Self {
        FineTuneFC { data: Vec::new() }
    }

    pub fn from_string(s: String) -> Self {
        let ft_vec = s
            .lines()
            .map(|l| {
                dbg!(&l);
                let ftc: PromptCompletion = serde_json::from_str(l).unwrap();
                ftc
            })
            .collect::<Vec<PromptCompletion>>();

        FineTuneFC { data: ft_vec }
    }

    pub async fn from_response(response: reqwest::Response) -> Result<Self, OairsError> {
        let s = response.text().await.unwrap();
        Ok(Self::from_string(s))
    }

    pub async fn save_jsonl(&self, path: &str) -> Result<(), OairsError> {
        let path = if !path.ends_with(".jsonl") {
            format!("{}.jsonl", path)
        } else {
            path.to_string()
        };

        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .await?;
        for line in &self.data {
            let s = serde_json::to_string(line).unwrap();
            file.write_all(s.as_bytes()).await?;
        }
        Ok(())
    }
}

/// Represents a single prompt-response in a `jsonl` fine-tune training file.
#[derive(Debug, Serialize, Deserialize, SaveJson)]
pub struct PromptCompletion {
    pub prompt: String,
    pub completion: String,
}

impl FromStr for PromptCompletion {
    type Err = OairsError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut lines = s.lines();

        let prompt = match lines.next() {
            Some(p) => p,
            None => {
                return Err(OairsError::new(
                    String::from("Unable to parse prompt from file content."),
                    ErrorType::ParseError,
                    None,
                    None,
                ))
            }
        };
        let completion = match lines.next() {
            Some(c) => c,
            None => {
                return Err(OairsError::new(
                    String::from("Unable to parse completion from file content."),
                    ErrorType::ParseError,
                    None,
                    None,
                ))
            }
        };

        Ok(PromptCompletion {
            prompt: prompt.to_string(),
            completion: completion.to_string(),
        })
    }
}

#[derive(Debug, Serialize, Deserialize, SaveJson)]
pub struct FineTuneResultsFC {
    pub step: Vec<u64>,
    pub elapsed_tokens: Vec<u64>,
    pub elapsed_examples: Vec<u64>,
    pub training_loss: Vec<f64>,
    pub training_sequence_accuracy: Vec<f64>,
    pub training_token_accuracy: Vec<f64>,
}

impl Default for FineTuneResultsFC {
    fn default() -> Self {
        Self::new()
    }
}

impl FineTuneResultsFC {
    pub fn new() -> Self {
        FineTuneResultsFC {
            step: Vec::new(),
            elapsed_tokens: Vec::new(),
            elapsed_examples: Vec::new(),
            training_loss: Vec::new(),
            training_sequence_accuracy: Vec::new(),
            training_token_accuracy: Vec::new(),
        }
    }

    fn from_string(s: String) -> Self {
        let mut step = Vec::new();
        let mut elapsed_tokens = Vec::new();
        let mut elapsed_examples = Vec::new();
        let mut training_loss = Vec::new();
        let mut training_sequence_accuracy = Vec::new();
        let mut training_token_accuracy = Vec::new();

        let mut in_header_row = true;
        for line in s.lines() {
            if in_header_row {
                in_header_row = false;
                continue;
            }
            let mut iter = line.split(',');
            let s = match iter.next() {
                Some(s) => s.parse::<u64>().unwrap(),
                None => continue,
            };
            let et = iter.next().unwrap().parse::<u64>().unwrap();
            let ee = iter.next().unwrap().parse::<u64>().unwrap();
            let tl = iter.next().unwrap().parse::<f64>().unwrap();
            let tsa = iter.next().unwrap().parse::<f64>().unwrap();
            let tta = iter.next().unwrap().parse::<f64>().unwrap();

            step.push(s);
            elapsed_tokens.push(et);
            elapsed_examples.push(ee);
            training_loss.push(tl);
            training_sequence_accuracy.push(tsa);
            training_token_accuracy.push(tta);
        }

        FineTuneResultsFC {
            step,
            elapsed_tokens,
            elapsed_examples,
            training_loss,
            training_sequence_accuracy,
            training_token_accuracy,
        }
    }

    /// Convert a `reqwest::Response` to a `FineTuneResultsFC` struct.
    pub async fn from_response(response: reqwest::Response) -> Result<Self, OairsError> {
        let s = response.text().await.unwrap();
        Ok(Self::from_string(s))
    }
}
