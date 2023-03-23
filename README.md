# Oairs

A Rust library for working with the OpenAI API.

Endpoints not currently covered:

* <https://api.openai.com/v1/images/edits> (create image edit)
* <https://api.openai.com/v1/images/variations> (create image variation)
* <https://api.openai.com/v1/audio/transcription> (create transcription)
* <https://api.openai.com/v1/audio/translations> (Create translation)

Currently only tested on Windows.

## Use

The main point of entry for working with the API is the `Client` struct. The typical initialization is intended to be as follows:

```rust
// Argument should be &str or String of your API key
let client = Client::new(...);
```

This returns `Client<Unkeyed>` which you can use to access each resource or endpoint. Any parameters that are required by the API will be required arguments to the method:

```rust
let model = CompletionModel::TextDavinci003;
client.completion(model);
```

If the endpoint has optional parameters, they are set by chaining methods.

```rust
client.completion(model)
    .prompt("Hello, world")
    .temperature(temp)
    .stream(true)
    // ...
```

Every request is only executed upon `await`ing `send()`:

```rust
// ...
let model = ChatModel::Gpt40314;
let mut msgs = Messages::new();

use Msg::System;
use Msg::User;
let system = System("You are a helpful assistant.".to_string());
let user = User("Hello, world.".to_string());

msgs.push(system);
msgs.push(user);

client.chat_completion(model, msgs)
    .send()
    .await
// ...
```

The return type of every executed request will be `Result<reqwest::Response, OairsError>`. Every `Response` can be deserialized by a struct. (As for why I've chosen to return the `Response` rather than a deserialized struct directly, see the excurses below.)

```rust
// ...
let model = EmbeddingModel::TextEmbeddingAda002;
let inputs = vec!["This is a test.".to_string(), "This is another test.".to_string()];

let mut embedding_response = match client.create_embeddings(model, &inputs).send().await {
  Ok(response) => response.json::<Embedding>().await.unwrap(),
  Err(e) => panic!("{e}"),
};
```

Each method's docstring contains information and examples on deserialization.

Every "deserializable" struct for a `reqwest::Response` can be saved to JSON via its `save_json(...)` method.

```rust
let model = EditModel::CodeDavinci002;
let model_obj = match client.retrieve_model(&model).send().await {
  Ok(response) => response.json::<ModelObject>().await.unwrap(),
  Err(e) => panic!("{e}"),
};

// The `.json` extension will be appended automatically if not already present
// The method attempts to create any parent directories if they are not found
// to exist. Certain characters that present problems on Windows filenames (e.g., `:`)
// will return an error.
match model_obj.save_json("some/filename") {
    Ok(_) => (),
    Err(e) => eprintln!("Error saving file: {}", e),
}
// ...
```

The `OairsError` also has this method available:

```rust
// ...
    Err(e) => e.save_json("errors/some_filename.json"),
// ...
```

At some point I may publish this crate, in which case you can refer to the documentation for more details. In the meantime, much of that documentation already exists in the docstrings.

## Credit

Naturally every dependency listed in the `Cargo.toml` file deserves credit. Special acknowledgement should be given to the [`tiktoken`](<https://github.com/openai/tiktoken>) and [`tiktoken-rs`](<https://github.com/zurawiki/tiktoken-rs>) crates from which I heavily copied (and slightly modified) for the tokenizer module (see the module comments for more information).

Other crates I relied heavily upon:
[`reqwest`](<https://github.com/seanmonstar/reqwest>), [`serde`](<https://github.com/serde-rs/serde>), and [`serde_json`](<https://github.com/mullr/serde-json>).

And the following crates which made adding certain things much easier: [`nom`](<https://github.com/rust-bakery/nom>), [`polars`](<https://github.com/pola-rs/polars>)

## Excurses On returning `reqwest::Response`

This was a personal project that I've used to try and learn Rust. I haven't focused on speed, but making the library easy to use. Nor do I have the general experience or specific Rust knowledge to know much about optimization. Thus, it may (or may not) be slower than other similar libraries. Along the lines of ease-of-use, I've tried to provide a lot of convenience methods (and plan to add more). Thus, I haven't taken much time to ask myself "Should this be left to the user?" or stuck only to creating an interface for the endpoints. And so it may (or may not) be larger than other similar libraries. Additionally, Rust being a new language for me, it will almost certainly fail to be idiomatic in places. And finally, due to the compounding factors of it being a project in its early stages and my ignorance, the updates will probably often involve breaking changes.

Initially I made the return type of every request the appropriate struct from a deserialized response. To make it easier to debug issues and get a better grasp of which fields I could expect from a response (which wasn't always obvious from the API documentation) I switched over to always returning the `reqwest::Response`. An error would only be returned in the case that `reqwest` itself failed. I ultimately decided to stick with this basic idea because it allows the macros handling the various send methods to be somewhat simplified and also gives users the option to have more information about the response if they want it (e.g., checking headers for rate-limits).

The only slight reversion to the earlier iteration (in this regard) is that I've screened the status of the responses ahead of time, so that if the `Result` type is `Ok`, the user can be assured that the response status is too (i.e., `response.status().is_success()` should always return `true` on that branch) and skip straight for deserialization. (However, given the early stage of the project, I may have left some gaps.) IMO, this gives a good balance between giving the user some more control over the response data while also not requiring them to add a lot of lines of code if they don't care about that.

If a user wants, they can still check the headers and confirm the status:

```rust
// ...
let response = match client.list_fine_tunes().send().await {
    Ok(r) => r,
    Err(e) => panic!("{e}"),
};

if response.status().is_success() {
    let ft_list = response.json::<FineTunesList>.await.unwrap();
    // or
    let s = response.text().await.unwrap();
    // or ...
} else {
    // handle bad status
}
```
