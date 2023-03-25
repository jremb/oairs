#![doc = include_str!("../README.md")]

use std::{
    borrow::{Borrow, Cow},
    collections::HashMap,
    marker::PhantomData,
};

use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize, Serializer};

pub mod audio;
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
    client::build_client,
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

impl Uri {
    pub(crate) fn get(&self) -> &str {
        URL.get(self).unwrap()
    }
}

// There are a few places where the use of url fields isn't very efficient, involving a clone due
// to path params. This feels poorly done, but at least easier to make changes.
pub(crate) static URL: Lazy<HashMap<Uri, &str>> = Lazy::new(|| {
    let mut map = HashMap::new();
    map.insert(Uri::Audio, "https://api.openai.com/v1/audio");
    map.insert(
        Uri::ChatCompletion,
        "https://api.openai.com/v1/chat/completions",
    );
    map.insert(Uri::Completions, "https://api.openai.com/v1/completions");
    map.insert(Uri::Edits, "https://api.openai.com/v1/edits");
    map.insert(Uri::Embeddings, "https://api.openai.com/v1/embeddings");
    map.insert(Uri::Files, "https://api.openai.com/v1/files");
    map.insert(Uri::FineTunes, "https://api.openai.com/v1/fine-tunes");
    map.insert(Uri::Images, "https://api.openai.com/v1/images");
    map.insert(Uri::Models, "https://api.openai.com/v1/models");
    map.insert(Uri::Moderations, "https://api.openai.com/v1/moderations");
    map
});

// TODO: Need to read about Cow and see if it would be a better choice, but the following works for now.

// https://api.openai.com/v1/fine-tunes/{fine_tune_id}/cancel
pub(crate) fn cancel_ft_url<'a>(ft_id: &str) -> String {
    format!("{}/{}/cancel", create_ft_url(), ft_id)
}

pub(crate) fn chat_completion_url<'a>() -> &'a str {
    Uri::ChatCompletion.get()
}

pub(crate) fn completion_url<'a>() -> &'a str {
    Uri::Completions.get()
}

pub(crate) fn create_ft_url<'a>() -> &'a str {
    Uri::FineTunes.get()
}

pub(crate) fn delete_file_url<'a>(file_id: &str) -> String {
    retrieve_file_url(file_id)
}

pub(crate) fn delete_ft_model_url<'a>(model: &str) -> String {
    format!("{}/{}", Uri::Models.get(), model)
}

pub(crate) fn img_create_url<'a>() -> String {
    format!("{}/generations", URL.get(&Uri::Images).unwrap())
}

pub(crate) fn img_edit_url<'a>() -> String {
    format!("{}/edits", URL.get(&Uri::Images).unwrap())
}

pub(crate) fn img_variation_url<'a>() -> String {
    format!("{}/variations", URL.get(&Uri::Images).unwrap())
}

pub(crate) fn list_files_url<'a>() -> &'a str {
    Uri::Files.get()
}

pub(crate) fn list_fine_tunes_url<'a>() -> &'a str {
    Uri::FineTunes.get()
}

pub(crate) fn list_models_url<'a>() -> &'a str {
    Uri::Models.get()
}

pub(crate) fn list_ft_events_url(ft_id: &str) -> String {
    format!("{}/{}/events", create_ft_url(), ft_id,)
}

pub(crate) fn retrieve_file_url(file_id: &str) -> String {
    format!("{}/{}", Uri::Files.get(), file_id)
}

pub(crate) fn retrieve_file_content_url(file_id: &str) -> String {
    format!("{}/content", retrieve_file_url(file_id))
}

pub(crate) fn retrieve_ft_info_url(ft_id: &str) -> String {
    format!("{}/{}", create_ft_url(), ft_id,)
}

pub(crate) fn retrieve_model_url(model: &str) -> String {
    format!("{}/{}", Uri::Models.get(), model)
}

pub(crate) fn upload_file_url<'a>() -> &'a str {
    Uri::Files.get()
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
    fn url_cancel_ft() {
        let ft_id = "ft_id";
        let url = cancel_ft_url(ft_id);

        let expected = "https://api.openai.com/v1/fine-tunes/ft_id/cancel";

        assert_eq!(url, expected);
    }

    #[test]
    fn url_file_delete() {
        let file_id = "file_id";
        let url = delete_file_url(file_id);

        let expected = "https://api.openai.com/v1/files/file_id";

        assert_eq!(url, expected);
    }

    #[test]
    fn url_file_retrieve() {
        let file_id = "file_id";
        let url = retrieve_file_url(file_id);

        let expected = "https://api.openai.com/v1/files/file_id";

        assert_eq!(url, expected);
    }

    #[test]
    fn url_file_retrieve_content() {
        let file_id = "file_id";
        let url = retrieve_file_content_url(file_id);

        let expected = "https://api.openai.com/v1/files/file_id/content";

        assert_eq!(url, expected);
    }

    #[test]
    fn url_ft_create() {
        let url = create_ft_url();

        let expected = "https://api.openai.com/v1/fine-tunes";

        assert_eq!(url, expected);
    }

    #[test]
    fn url_ft_list_events() {
        let ft_id = "ft_id";
        let url = list_ft_events_url(ft_id);

        let expected = "https://api.openai.com/v1/fine-tunes/ft_id/events";

        assert_eq!(url, expected);
    }
}
