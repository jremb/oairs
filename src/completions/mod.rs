//! Contains the structs for sending and recieving data from the
//! `https://api.openai.com/v1/completions` endpoint as well as the
//! `https://api.openai.com/v1/chat/completions` endpoint.

// Given the API documentation, it might be less confusing to find these as
// their own seperate modules: `completions` and `chat`. However, they are
// both completions, ultimately, and they each have a few structs in common.
// Putting them in their own seperate modules would require duplicating
// the common structs (otherwise it would be confusing if, e.g.,, users had to
// import a `Temperature` struct from the chat module for use in a completion).
// Either way, I don't see it as a major issue. I decided to combine them into
// one module for now.

mod chat_builder;
mod completion_builder;
pub mod response;

pub use self::chat_builder::*;
pub use self::completion_builder::*;

use super::*;

// Both a chat-completion and a completion rely upon the following structs.
// A response from either also relies upon a `Usage` struct, which is defined in
// `response.rs` of this module.

/// Used to set the amount of randomness for a model when generating a
/// completion. The valid range is 0 to 2. A value of 2 can lead to incoherent
/// completions. Panics if `temperature` is not between 0.0 and 2.0.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Temperature(f32);
impl Temperature {
    pub fn new(temperature: f32) -> Self {
        if !(0.0..=2.0).contains(&temperature) {
            panic!("Temperature must be between 0.0 and 2.0");
        }
        Temperature(temperature)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopP(f32);
impl TopP {
    pub fn new(top_p: f32) -> Self {
        if !(0.0..=1.0).contains(&top_p) {
            panic!("TopP must be between 0.0 and 1.0");
        }
        TopP(top_p)
    }
}
