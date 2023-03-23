use super::*;

/// Convenience struct for deserializing a successful response from a request to the models endpoint.
/// method.
///
/// Implements the `SaveJson` trait, allowing you to save the struct to a file with the [`save_json`]
/// method.
/// # Example
/// ```rust
/// // We are assuming the `response` is from a successful execution of
/// // `client.models().list().send().await?;`
/// let models_list: ModelsList = response.json().await?;
/// models_list.save_json("models.json")?;
/// ```
#[derive(Debug, Serialize, Deserialize, SaveJson)]
pub struct ModelsList {
    data: Vec<ModelObject>,
    object: String,
}

/// Convenience struct for deserializing a successful response from the models retrieve endpoint.
/// method. Also used as a substruct of the [`ModelsList`].
///
/// Implements the `SaveJson` trait, allowing you to save the struct to a file with the [`save_json`]
/// method.
/// # Example
/// ```
/// // We are assuming the `response` is from a successful execution of
/// // `client.models().retrieve(&model).send().await?;`
/// let model_data: ModelData = response.json().await?;
/// model_data.save_json("model.json")?;
/// ```
#[derive(Debug, Serialize, Deserialize, SaveJson)]
pub struct ModelObject {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
    pub permission: Vec<ModelPermissions>,
    pub root: String,
    pub parent: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelPermissions {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub allow_create_engine: bool,
    pub allow_sampling: bool,
    pub allow_logprobs: bool,
    pub allow_search_indices: bool,
    pub allow_view: bool,
    pub allow_fine_tuning: bool,
    pub organization: String,
    pub group: Option<String>,
    pub is_blocking: bool,
}
