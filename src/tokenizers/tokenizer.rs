use super::*;

// The following enums and struct provide an abstraction over the various
// encoders/decoders and special tokens that, to my mind, make them easier to
// work with. (I.e., easier to remember `tokenize` and let your IDE help with
// tokenizer types than to remember `r50k_base`, `p50k_base`, `p50k_edit`, and
// `cl100k_base`.)

/// Representations of the encoders/decoders used by different models. Not all
/// variants recognize the same [`SpecialToken`]s.
///
/// You can use the [`Tokenizer::recognized_special`] method to get a
/// [`SpecialTokens`] struct, which simply acts as a wrapper around a [`HashSet`]
/// of [`SpecialToken`]s. This is useful for checking whether a given
/// [`SpecialToken`] is recognized by a given [`Tokenizer`].
#[non_exhaustive]
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Tokenizer {
    /// zurawiki crate: "Use for GPT-3 models like `davinci`"
    R50KBase,
    /// zurawiki crate: "Use for Code models, `text-davinci-002`, `text-davinci-003`"
    P50KBase,
    /// zurawiki crate: "Use for Code models, `text-davinci-002`, `text-davinci-003`"
    P50KEdit,
    /// For text-embedding-ada-002 and chat models (e.g., gpt-3.5-turbo etc.)
    #[default]
    CL100KBase,
}

impl Tokenizer {
    pub fn to_str(&self) -> &str {
        match self {
            Tokenizer::R50KBase => "r50k_base",
            Tokenizer::P50KBase => "p50k_base",
            Tokenizer::P50KEdit => "p50k_edit",
            Tokenizer::CL100KBase => "cl100k_base",
        }
    }

    /// Returns [`SpecialTokens`] that are recognized by the variant.
    pub fn recognized_special(&self) -> SpecialTokens {
        match self {
            Tokenizer::CL100KBase => SpecialTokens::from_vec(SpecialToken::ALL.to_vec()),
            Tokenizer::P50KEdit => SpecialTokens::from_vec(
                SpecialToken::ALL
                    .iter()
                    .filter_map(|x| {
                        if x == &SpecialToken::EndOfPrompt {
                            None
                        } else {
                            Some(*x)
                        }
                    })
                    .collect::<Vec<SpecialToken>>(),
            ),
            _ => SpecialTokens::from_vec(vec![SpecialToken::EndOfText]),
        }
    }
}

/// Representations of special tokens for the encoders.
///
/// >"Special tokens are artificial tokens used to unlock capabilities from a
/// >model, such as fill-in-the-middle. So we want to be careful about
/// >accidentally encoding special tokens, since they can be used to trick a
/// >model into doing something we don't want it to do."
/// [Source](<https://github.com/openai/tiktoken/blob/main/tiktoken/core.py>)
///
/// Not every encoder supports every special token. You can use
/// [`Tokenizer::recognized_special`] to get [`SpecialTokens`] recognized by the
/// [`Tokenizer`] variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SpecialToken {
    EndOfText,
    FimPrefix,
    FimMiddle,
    FimSuffix,
    EndOfPrompt,
}

impl std::fmt::Display for SpecialToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SpecialToken::EndOfText => write!(f, "<|endoftext|>"),
            SpecialToken::FimPrefix => write!(f, "<|fim_prefix|>"),
            SpecialToken::FimMiddle => write!(f, "<|fim_middle|>"),
            SpecialToken::FimSuffix => write!(f, "<|fim_suffix|>"),
            SpecialToken::EndOfPrompt => write!(f, "<|endofprompt|>"),
        }
    }
}

impl SpecialToken {
    pub fn to_str(&self) -> &str {
        match self {
            SpecialToken::EndOfText => "<|endoftext|>",
            SpecialToken::FimPrefix => "<|fim_prefix|>",
            SpecialToken::FimMiddle => "<|fim_middle|>",
            SpecialToken::FimSuffix => "<|fim_suffix|>",
            SpecialToken::EndOfPrompt => "<|endofprompt|>",
        }
    }

    pub const ALL: [SpecialToken; 5] = [
        SpecialToken::EndOfText,
        SpecialToken::FimPrefix,
        SpecialToken::FimMiddle,
        SpecialToken::FimSuffix,
        SpecialToken::EndOfPrompt,
    ];
}

/// Wrapper around a HashSet of [`SpecialToken`]s that makes use of
/// [`tokenize_custom`] more convenient while also making improper use more difficult.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct SpecialTokens {
    tokens: HashSet<SpecialToken>,
}

impl SpecialTokens {
    pub fn new() -> Self {
        Self {
            tokens: HashSet::new(),
        }
    }

    pub fn from_vec(tokens: Vec<SpecialToken>) -> Self {
        Self {
            tokens: tokens.into_iter().collect(),
        }
    }

    pub fn insert(&mut self, token: SpecialToken) {
        self.tokens.insert(token);
    }

    pub fn contains(&self, token: SpecialToken) -> bool {
        self.tokens.contains(&token)
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn iter(&self) -> std::collections::hash_set::Iter<SpecialToken> {
        self.tokens.iter()
    }

    pub fn extend(&mut self, other: Vec<SpecialToken>) {
        self.tokens.extend(other.iter());
    }

    fn to_str_set(&self) -> HashSet<&str> {
        self.tokens.iter().map(|s| s.to_str()).collect()
    }
}

fn load_bpe(tokenizer: Tokenizer) -> Result<vendor_tiktoken::CoreBPE, OairsError> {
    let bpe = match tokenizer {
        Tokenizer::R50KBase => r50k_base(),
        Tokenizer::P50KBase => p50k_base(),
        Tokenizer::P50KEdit => p50k_edit(),
        Tokenizer::CL100KBase => cl100k_base(),
    };

    match bpe {
        Ok(bpe) => Ok(bpe),
        Err(e) => Err(OairsError::new(
            format!("Failed to load tokenizer: {}", e),
            ErrorType::Tokenizer,
            Some(tokenizer.to_str().into()),
            None,
        )),
    }
}

/// Calls the [`encode_ordinary`](vendor_tiktoken::CoreBPE::encode_ordinary) method of
/// the [`CoreBPE`](vendor_tiktoken::CoreBPE) struct. Any text matching a
/// [`SpecialToken`] will be encoded as an ordinary an token.
///
/// >"Special tokens are artificial tokens used to unlock capabilities from a
/// >model, such as fill-in-the-middle. So we want to be careful about
/// >accidentally encoding special tokens, since they can be used to trick a
/// >model into doing something we don't want it to do."
/// [Source](<https://github.com/openai/tiktoken/blob/main/tiktoken/core.py>)
///
/// The [`tokenize_with_special`] and [`tokenize_custom`] functions can be used to
/// encode special tokens. [`tokenize_custom`] allows you to specify which special
/// tokens should be treated as such and it also performs a check that will return an
/// error if you attempt to recognize a special token that the tokenizer can't
/// recognize. If you don't need this check and want every recognizable special token
/// to be treated as such, use the [`tokenize_with_special`] function (any
/// unrecognized special token will be treated as an ordinary token).
pub fn tokenize(text: &str, tokenizer: Tokenizer) -> Result<Vec<usize>, OairsError> {
    match load_bpe(tokenizer) {
        Ok(bpe) => {
            let tokens = bpe.encode_ordinary(text);
            Ok(tokens)
        }
        Err(e) => Err(e),
    }
}

/// Calls the `encode` method of the `CoreBPE` struct.
///
/// If you have no special tokens (e.g., `<|endoftext|>`), it makes no practical
/// difference whether you call `tokenize`, `tokenize_ordinary`, or
/// `tokenize_with_special`.
pub fn tokenize_custom(
    text: &str,
    tokenizer: Tokenizer,
    allowed_special: SpecialTokens,
) -> Result<Vec<usize>, OairsError> {
    let recognized_special = tokenizer.recognized_special();
    for token in allowed_special.iter() {
        if !recognized_special.contains(*token) {
            return Err(OairsError::new(
                format!(
                    "Special token {:?} is not recognized by tokenizer {:?}.",
                    token, tokenizer
                ),
                ErrorType::Tokenizer,
                Some(tokenizer.to_str().into()),
                None,
            ));
        }
    }

    match load_bpe(tokenizer) {
        Ok(bpe) => {
            let tokens = bpe.encode(text, allowed_special.to_str_set());
            Ok(tokens)
        }
        Err(e) => Err(e),
    }
}

/// Calls the `encode_with_special_tokens` method of the `CoreBPE` struct. All
/// recognizable special tokens will be treated as such. Any unrecognized special
/// tokens will be treated as ordinary tokens.
///
/// If you want to specify a limited set of special tokens to recognize or you want to
/// verify that a tokenizer is capable of recognizing a special token, use the
/// [`tokenize_custom`]
pub fn tokenize_with_special(text: &str, tokenizer: Tokenizer) -> Result<Vec<usize>, OairsError> {
    match load_bpe(tokenizer) {
        Ok(bpe) => {
            let tokens = bpe.encode_with_special_tokens(text);
            Ok(tokens)
        }
        Err(e) => Err(e),
    }
}

#[cfg(test)]
mod tokenizer_tests {
    use crate::tokenizers::openai_public::cl100k_base;

    use super::*;

    // The following test is simply a copy of the example code at this repo:
    // https://github.com/zurawiki/tiktoken-rs
    #[test]
    fn tokens_t0() {
        let bpe = cl100k_base().unwrap();
        let tokens = bpe.encode_with_special_tokens("This is an example");

        let actual = tokens.len();
        let expected = 4;

        assert_eq!(actual, expected);
    }

    // The rest of the tests compare the results with passing the same string to the
    // tiktoken Python library.

    #[test]
    fn tokens_t1() {
        let s = "This is a test string to see how it tokenizes.";
        let tokens = tokenize(s, Tokenizer::CL100KBase).unwrap();

        let actual = tokens.len();
        let expected = 12;

        assert_eq!(actual, expected);

        let expected_tokens = vec![
            2028, 374, 264, 1296, 925, 311, 1518, 1268, 433, 4037, 4861, 13,
        ];
        assert_eq!(tokens, expected_tokens)
    }

    #[test]
    fn tokens_t2() {
        let s = "The one who shuts his ears to the cry of the poor will himself also call out and not be answered.";
        let tokens = tokenize(s, Tokenizer::CL100KBase).unwrap();

        let actual = tokens.len();
        let expected = 22;

        assert_eq!(actual, expected);

        let expected_tokens = vec![
            791, 832, 889, 89678, 813, 25212, 311, 279, 16106, 315, 279, 8009, 690, 5678, 1101,
            1650, 704, 323, 539, 387, 19089, 13,
        ];

        assert_eq!(tokens, expected_tokens)
    }
}
