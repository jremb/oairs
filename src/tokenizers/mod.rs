//! The code in this module, aside from `tokenizer.rs`, was copied and lightly modified from <https://github.com/zurawiki/tiktoken-rs>
//! which is itself adapted to some degree from here <https://github.com/dust-tt/dust> and the original code here
//! <https://github.com/openai/tiktoken>. See the LICENSE file for the licenses of the respective projects.
//!
//! The code in `tokenizer.rs` is my own, as is probably obvious from its more amateur style. I have
//! applied some comments from the zurawiki crate into `tokenizer.rs` to document the TokenizerType variants. These comments
//! were not originally made by zurawiki about the enum variants.
//!
//! The rationale for not simply using the zurawiki crate is twofold: (1) I could not get the crate to succesfully
//! compile (DLL error) and (2, the zurawiki crate included some dependencies and code, particularly relating to PYO3,
//! that seemed unnecessary for the Oairs library.
//!
//! NOTE: All of the docstrings and comments in the original files have been left as is, unless the relevant portion of
//! code was removed or modified.
use std::collections::HashSet;

mod openai_public;
mod singletons;
mod vendor_tiktoken;

pub mod tokenizer;

use self::openai_public::{cl100k_base, p50k_base, p50k_edit, r50k_base};
pub use self::tokenizer::*;
use super::error::{ErrorType, OairsError};
