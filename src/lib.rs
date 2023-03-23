#![doc = include_str!("../README.md")]

use std::{collections::HashMap, marker::PhantomData};

use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize, Serializer};

pub mod client;
pub mod completions;
pub mod edits;
pub mod embeddings;
pub mod error;
pub mod files;
pub mod fine_tunes;
pub mod images;
pub mod macros;
pub mod models;
pub mod moderations;
pub mod tokenizers;
pub mod utils;

use crate::{
    client::{build_client, ContentType},
    completions::{response::Usage, Temperature, TopP},
    error::*,
    files::Purpose,
    macros::*,
    models::{
        ChatModel, CompletionModel, EditModel, EmbeddingModel, FineTuneModel, ModerationModel,
        RetrievableModel,
    },
    moderations::ModerationBuilder,
    utils::write_parquet,
};

use save_json::SaveJson;

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub(crate) enum Uri {
    Audio,
    ChatCompletion,
    Completions,
    Edits,
    Embeddings,
    Files,
    FineTunes,
    Images,
    Models,
    Moderations,
}

// There are a few places where the use of url fields isn't very efficient, involving a clone due to path params.
// This feels poorly done, but at least easier to make changes.
pub(crate) static URL: Lazy<HashMap<Uri, String>> = Lazy::new(|| {
    let mut map = HashMap::new();
    map.insert(Uri::Audio, "https://api.openai.com/v1/audio".to_string());
    map.insert(
        Uri::ChatCompletion,
        "https://api.openai.com/v1/chat/completions".to_string(),
    );
    map.insert(
        Uri::Completions,
        "https://api.openai.com/v1/completions".to_string(),
    );
    map.insert(Uri::Edits, "https://api.openai.com/v1/edits".to_string());
    map.insert(
        Uri::Embeddings,
        "https://api.openai.com/v1/embeddings".to_string(),
    );
    map.insert(Uri::Files, "https://api.openai.com/v1/files".to_string());
    map.insert(
        Uri::FineTunes,
        "https://api.openai.com/v1/fine-tunes".to_string(),
    );
    map.insert(Uri::Images, "https://api.openai.com/v1/images".to_string());
    map.insert(Uri::Models, "https://api.openai.com/v1/models".to_string());
    map.insert(
        Uri::Moderations,
        "https://api.openai.com/v1/moderations".to_string(),
    );
    map
});

pub(crate) fn retrieve_file_url(file_id: &str) -> String {
    format!("{}/{}", URL.get(&Uri::Files).unwrap(), file_id,)
}

pub(crate) fn delete_file_url(file_id: &str) -> String {
    retrieve_file_url(file_id)
}

pub(crate) fn retrieve_file_content_url(file_id: &str) -> String {
    format!("{}/content", retrieve_file_url(file_id),)
}

pub(crate) fn create_ft_url() -> String {
    URL.get(&Uri::FineTunes).unwrap().to_string()
}

pub(crate) fn list_ft_events_url(ft_id: &str) -> String {
    format!("{}/{}/events", create_ft_url(), ft_id,)
}

// region: type-state trackers

// used to track the endpoint-state of Client and some builder structs
// in other files.

#[derive(Default)]
pub struct Unkeyed;

#[derive(Default)]
pub struct Keyed;

#[derive(Default)]
pub struct Gettable;

#[derive(Default)]
pub struct Sendable;

#[derive(Default)]
pub struct Buildable;

#[derive(Default)]
pub struct Cancel;

/// Used for both delte file and delete fine-tune model
#[derive(Debug, Default)]
pub struct Delete;

// endregion

#[cfg(test)]
mod url_test {
    use super::*;

    #[test]
    fn url_file_delete() {
        let file_id = "file_id";
        let url = delete_file_url(file_id);

        assert_eq!(url, "https://api.openai.com/v1/files/file_id");
    }

    #[test]
    fn url_file_retrieve() {
        let file_id = "file_id";
        let url = retrieve_file_url(file_id);

        assert_eq!(url, "https://api.openai.com/v1/files/file_id");
    }

    #[test]
    fn url_file_retrieve_content() {
        let file_id = "file_id";
        let url = retrieve_file_content_url(file_id);

        assert_eq!(url, "https://api.openai.com/v1/files/file_id/content");
    }

    #[test]
    fn url_ft_create() {
        let url = create_ft_url();
        assert_eq!(url, "https://api.openai.com/v1/fine-tunes");
    }

    #[test]
    fn url_ft_list_events() {
        let ft_id = "ft_id";
        let url = list_ft_events_url(ft_id);
        assert_eq!(url, "https://api.openai.com/v1/fine-tunes/ft_id/events");
    }
}
