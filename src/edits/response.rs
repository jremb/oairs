use super::*;

#[derive(Debug, Serialize, Deserialize, SaveJson)]
pub struct Edit {
    object: String,
    created: u64,
    choices: Vec<EditChoice>,
    usage: Usage,
}

// TODO: Use serde to collapse some of these choice structs in various modules.
#[derive(Debug, Serialize, Deserialize)]
pub struct EditChoice {
    pub index: u8,
    pub text: String,
    pub finish_reason: Option<String>,
}
