static MODEL_PREFIX_TO_ENCODING: Lazy<HashMap<&'static str, &'static str>> = Lazy::new(|| {
    let mut map = HashMap::new();
    // chat
    map.insert("gpt-3.5-turbo-", "cl100k_base"); // e.g, gpt-3.5-turbo-0301, -0401, etc.
    map.insert("gpt-4-", "cl100k_base"); // e.g, gpt-4-0314,  etc.
    map
});

static MODEL_TO_ENCODING: Lazy<HashMap<&'static str, &'static str>> = Lazy::new(|| {
    let mut map = HashMap::new();
    // chat
    map.insert("gpt-3.5-turbo-", "cl100k_base");
    map.insert("gpt-4-", "cl100k_base");
    // text
    map.insert("text-davinci-003", "p50k_base");
    map.insert("text-davinci-002", "p50k_base");
    map.insert("text-davinci-001", "r50k_base");
    map.insert("text-curie-001", "r50k_base");
    map.insert("text-babbage-001", "r50k_base");
    map.insert("text-ada-001", "r50k_base");
    map.insert("davinci", "r50k_base");
    map.insert("curie", "r50k_base");
    map.insert("babbage", "r50k_base");
    map.insert("ada", "r50k_base");
    // code
    map.insert("code-davinci-002", "p50k_base");
    map.insert("code-davinci-001", "p50k_base");
    map.insert("code-cushman-002", "p50k_base");
    map.insert("code-cushman-001", "p50k_base");
    map.insert("davinci-codex", "p50k_base");
    map.insert("cushman-codex", "p50k_base");
    // edit
    map.insert("text-davinci-edit-001", "p50k_edit");
    map.insert("code-davinci-edit-001", "p50k_edit");
    // embeddings
    map.insert("text-embedding-ada-002", "cl100k_base");
    // old embeddings
    map.insert("text-similarity-davinci-001", "r50k_base");
    map.insert("text-similarity-curie-001", "r50k_base");
    map.insert("text-similarity-babbage-001", "r50k_base");
    map.insert("text-similarity-ada-001", "r50k_base");
    map.insert("text-search-davinci-doc-001", "r50k_base");
    map.insert("text-search-curie-doc-001", "r50k_base");
    map.insert("text-search-babbage-doc-001", "r50k_base");
    map.insert("text-search-ada-doc-001", "r50k_base");
    map.insert("code-search-babbage-code-001", "r50k_base");
    map.insert("code-search-ada-code-001", "r50k_base");
    // open source
    map.insert("gpt2", "gpt2");
    map
});


/// Returns the encoding used by a model.
///
/// TODO use hashmap 
pub fn encoding_for_model(model_name: &str) -> Option<&str> {
    if let Some(encoding) = MODEL_TO_ENCODING
        .iter()
        .find(|(model, _)| *model == model_name)
    {
        return Some(encoding.1);
    }
    if let Some(encoding) = MODEL_PREFIX_TO_ENCODING
        .iter()
        .find(|(model_prefix, _)| model_name.starts_with(*model_prefix))
    {
        return Some(encoding.1);
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoding_for_model() {
        assert_eq!(encoding_for_model("gpt-3.5-turbo"), Some("cl100k_base"));
        assert_eq!(
            encoding_for_model("gpt-3.5-turbo-0301"),
            Some("cl100k_base")
        );
        assert_eq!(encoding_for_model("text-davinci-003"), Some("p50k_base"));
        assert_eq!(
            encoding_for_model("code-search-ada-code-001"),
            Some("r50k_base")
        );
        assert_eq!(encoding_for_model("foo"), None);
    }
}