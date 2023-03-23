//! # save_json
//! This crate provides a convenience trait, `SaveJson`, that can be derived for structs that
//! implement `Serialize`. The trait will implement a `save_json` method for the struct, which
//! takes a `&str` path and uses the `serde_json::to_writer_pretty` method to serialize the struct.
//!
//! If the path does not end with ".json" it will be appended. If the path includes parents that do not
//! exist it will try to create them.
//!
//! Fails if it cannot create the path (or its parents) or if serialization fails on
//! [the specified conditions](https://docs.rs/serde_json/latest/serde_json/fn.to_writer_pretty.html#errors).
//!  
//! # Arguments
//! * `path` - `&str` that represents the path to the file to be created.
//!
//! # Example
//! The following example assumes that we are using the `oairs` library to send a request to the
//! OpenAI API. The `response` is a `reqwest::Response` with the `json` feature.
//! ```rust,no_run
//! #[derive(Debug, Serialize, Deserialize, SaveJson)]
//! pub struct Completion {
//!     pub id: String,
//!     pub object: String,
//!     pub created: u64,
//!     pub model: String,
//!     pub choices: Vec<CompletionChoice>,
//!     pub usage: Usage,
//! }
//! // ...
//!
//! let completion: Completion = response.json().await?;
//!
//! let path = "some/path/completion.json";
//! match completion.save_json(&path) {
//!    Ok(_) => println!("Saved completion to {}", path),
//!    Err(e) => println!("Failed to save completion to {}: {}", path, e),
//! };
//! ```

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput};

#[proc_macro_derive(SaveJson)]
pub fn derive(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);
    impl_save_json_macro_derive(&ast)
}

// TODO: Check this with FileResponse
fn impl_save_json_macro_derive(ast: &syn::DeriveInput) -> TokenStream {
    let name = &ast.ident;
    quote!{
        impl #name {
            pub fn save_json(&self, path: &str) -> Result<(), std::io::Error> {

                let path = if path.ends_with(".json") {
                    path.to_string()
                } else {
                    format!("{}.json", path)
                };
                let path = std::path::Path::new(&path);
                let parent = path.parent().unwrap();
                if !parent.exists() {
                    match std::fs::create_dir_all(parent) {
                        Ok(_) => (),
                        Err(e) => {
                            // By default, we get the error 'failed to create whole tree', which
                            // probably isn't very helpful to many users when the error will simply
                            // point to the function call.
                            let e = format!(
                                "Path's parent directory, {}, does not exist and could not be created. Possibly invalid characters or non-existent? {}",
                                parent.display(),
                                e
                            );
                            return Err(
                                std::io::Error::new(std::io::ErrorKind::Other, e)
                            )
                        },
                    };
                }

                // In my testing, `:` in the filename itself won't be caught as an error by
                // File::create. However, (on Windows at least) a path like "te:st2.json" will create a 
                // file named `te` and return a handle to a path that doesn't exist, which won't be caught 
                // by serde_json. The user will end up with an empty file named `te` and no indication
                // that an error has occured.
                let filename = path.file_name().unwrap().to_str().unwrap();
                if filename.contains(":") {
                    return Err(
                            std::io::Error::new
                            (std::io::ErrorKind::InvalidInput, "filename contains invalid character: :")
                        );
                }

                let file = match std::fs::File::create(&path) {
                    Ok(f) => f,
                    Err(e) => return Err(e),
                };

                match serde_json::to_writer_pretty(&file, &self) {
                    Ok(_) => Ok(()),
                    Err(e) => {
                        println!("Error: {}", e);
                        let e = std::io::Error::new(std::io::ErrorKind::Other, e.to_string());
                        return Err(e)
                    },
                }
            }
        }
    }.into()
}
