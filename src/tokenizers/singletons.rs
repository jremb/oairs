use std::sync::Arc;

use once_cell::sync::OnceCell;
use parking_lot::Mutex;

use crate::tokenizers::openai_public::{cl100k_base, p50k_base, p50k_edit, r50k_base};
use crate::tokenizers::vendor_tiktoken::CoreBPE;

/// Returns a singleton instance of the r50k_base tokenizer. (also known as `gpt2`)
/// Use for GPT-3 models like `davinci`
///
/// This function will only initialize the tokenizer once, and then return a reference the tokenizer
pub fn r50k_base_singleton() -> Arc<Mutex<CoreBPE>> {
    static R50K_BASE: OnceCell<Arc<Mutex<CoreBPE>>> = OnceCell::new();
    R50K_BASE
        .get_or_init(|| Arc::new(Mutex::new(r50k_base().unwrap())))
        .clone()
}

/// Returns a singleton instance of the p50k_base tokenizer.
/// Use for Code models, `text-davinci-002`, `text-davinci-003`
///
/// This function will only initialize the tokenizer once, and then return a reference the tokenizer.
pub fn p50k_base_singleton() -> Arc<Mutex<CoreBPE>> {
    static P50K_BASE: OnceCell<Arc<Mutex<CoreBPE>>> = OnceCell::new();
    P50K_BASE
        .get_or_init(|| Arc::new(Mutex::new(p50k_base().unwrap())))
        .clone()
}

/// Returns a singleton instance of the p50k_edit tokenizer.
/// Use for edit models like `text-davinci-edit-001`, `code-davinci-edit-001`
///
/// This function will only initialize the tokenizer once, and then return a reference the tokenizer.
pub fn p50k_edit_singleton() -> Arc<Mutex<CoreBPE>> {
    static P50K_EDIT: OnceCell<Arc<Mutex<CoreBPE>>> = OnceCell::new();
    P50K_EDIT
        .get_or_init(|| Arc::new(Mutex::new(p50k_edit().unwrap())))
        .clone()
}

/// Returns a singleton instance of the cl100k_base tokenizer.
/// Use for ChatGPT models, `text-embedding-ada-002`
///
/// This function will only initialize the tokenizer once, and then return a reference the tokenizer
pub fn cl100k_base_singleton() -> Arc<Mutex<CoreBPE>> {
    static CL100K_BASE: OnceCell<Arc<Mutex<CoreBPE>>> = OnceCell::new();
    CL100K_BASE
        .get_or_init(|| Arc::new(Mutex::new(cl100k_base().unwrap())))
        .clone()
}
