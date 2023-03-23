use super::*;

// region: client macros

// TODO: If async in traits gets stabilized, consider using that instead.
#[allow(unused_macros)]
macro_rules! impl_del {
    ($typ:ident < $( $gen:tt ),+ >) => {
        impl<'a> $typ<$($gen),*> {
            /// Executes the `DELETE` request. Returns a `Result` with either a `reqwest::Response` or an
            /// `OairsError`.
            pub async fn send(&self) -> Result<reqwest::Response, OairsError> {
                let client = match build_client(&self.key) {
                    Ok(c) => c,
                    Err(e) => return Err(e),
                };

                match client.delete(self.url.to_string()).send().await {
                    Ok(r) => Ok(r),
                    Err(e) => Err(parse_reqwest_error(e)),
                }
            }
        }
    };
}
pub(crate) use impl_del;

#[allow(unused_macros)]
macro_rules! impl_get {
    ($typ:ident < $( $gen:tt ),+ >) => {
        impl<'a> $typ<$($gen),*> {
            /// Executes the `GET` request. Returns a `Result` with either a `reqwest::Response` or an
            /// `OairsError`.
            pub async fn send(&self) -> Result<reqwest::Response, OairsError> {
                let client = match build_client(&self.key) {
                    Ok(c) => c,
                    Err(e) => return Err(e),
                };

                let response = match client.get(self.url.to_string()).send().await {
                    Ok(r) => r,
                    Err(e) => return Err(parse_reqwest_error(e)),
                };

                let sc = response.status();
                match sc {
                    reqwest::StatusCode::OK => Ok(response),
                    _ => Err(parse_api_error(response, sc).await),
                }
            }
        }
    };
}
pub(crate) use impl_get;

macro_rules! impl_post {
    // A POST that only has a path parameter (e.g., fine-tunes cancel endpoint)
    ($strct:ident < $( $type_state:tt ),+ >) => {
        impl<'a> $strct<$($type_state),*> {
            /// Executes the `POST` request. Returns a `Result` with either a `reqwest::Response` or an
            /// `OairsError`.
            pub async fn send(&self) ->  Result<reqwest::Response, OairsError> {
                let client = match build_client(&self.key) {
                    Ok(c) => c,
                    Err(e) => return Err(e),
                };

                let url = self.url.clone().unwrap();

                match client
                    .post(&url)
                    .send()
                    .await
                {
                    Ok(r) => Ok(r),
                    Err(e) => Err(parse_reqwest_error(e)),
                }
            }
        }
    };
    // POST that has a request body. There is some arbitrariness in the pattern matching here
    // to distinguish a post that has no request body vs. one with a body
    ($strct:ident < $( $type_state:tt ),+ >, $cont_type:expr) => {
        impl<'a> $strct<$($type_state),*> {
            /// Executes the `POST` request. Returns a `Result` with either a `reqwest::Response` or an
            /// `OairsError`.
            pub async fn send(&self) -> Result<reqwest::Response, OairsError> {
                let client = match build_client(&self.key) {
                    Ok(c) => c,
                    Err(e) => return Err(e),
                };

                let response = match client
                    .post(self.url.to_string())
                    .header("Content-Type", $cont_type.to_str())
                    .json(self)
                    .send()
                    .await
                {
                    Ok(r) => r,
                    Err(e) => return Err(parse_reqwest_error(e)),
                };

                let sc = response.status();

                match sc {
                    reqwest::StatusCode::OK => Ok(response),
                    _ => Err(parse_api_error(response, sc).await),
                }
            }
        }
    }
}
pub(crate) use impl_post;

// Currently only one endpoint uses a form, therefore, no reason for a generic macro.
// However, if more endpoints use forms in the future, this macro can be used to
// generate the code for them. ...
#[allow(unused_macros)]
macro_rules! impl_post_form {
    ($strct:ident < $( $type_state:tt ),+ >) => {
        impl<'a> $strct<$($type_state),*> {
            /// Executes the `POST` request. Returns a `Result` with either a `reqwest::Response` or an
            /// `OairsError`.
            pub async fn post_form(&self) -> Result<reqwest::Response, OairsError> {
                let path = self.file;
                let purpose = self.purpose.clone();

                let file = match std::fs::read(path) {
                    Ok(f) => f,
                    Err(e) => {
                        return Err(
                            OairsError::new(
                                format!("Error reading file: {e}"),
                                ErrorType::FileError,
                                Some(format!("{}, {}", path, purpose.to_string())),
                                None,
                        ))
                    }
                };

                let path = path.to_string();
                let file_part = reqwest::multipart::Part::bytes(file).file_name(path);

                let form = reqwest::multipart::Form::new()
                    .text("purpose", purpose.to_string())
                    .part("file", file_part);

                let client = match build_client(&self.key) {
                    Ok(c) => c,
                    Err(e) => return Err(e),
                };

                match client
                    .post(&self.url)
                    .header("Content-Type", "multipart/form-data")
                    .multipart(form)
                    .send()
                    .await
                {
                    Ok(r) => Ok(r),
                    Err(e) => Err(parse_reqwest_error(e)),
                }
            }
        }
    }
}
pub(crate) use impl_post_form;

// endregion: client macros

// region: model macros

/// Generates a `CustomModel` enum which implements the `RetrievableModel` trait. Pass in the identifier
/// you want to use as the enum variant, followed by a comma, followed by the string literal that the
/// API expects for that model. You can repeat this pattern however many times you want.
///
/// This macro is intended to be used for models that are not enumerated in the model enums, but
/// which might be generally available to all users via the `retrieve_model()` method. For Fine-Tuned
/// models, use the `ft_model!` macro. Both macros currently generate the same code, with different
/// enum identifiers, but this may change in the future!
///
/// WARNING: Using the model in a request will fail if the string literal doesn't match the model id that
/// the API expects! If you're not sure what the string literal should be, use the `list_models()` method
/// to retrieve a list of models and their ids.
///
/// # Example
/// Assume that the OpenAI's API expects the model id `if-davinci-v2`. No variant for this model is
/// currently enumerated in the model enums, so we can use the `custom_model!` macro to create a
/// `CustomModel` enum variant that implements the `RetrievableModel` trait and can be used with the
/// `retrieve_model()` method.
///
/// ```rust
/// custom_model!(IfDavinciV2, "if-davinci-v2");
///
/// let ifd2 = CustomModel::IfDavinciV2;
/// // The RetrievableModel trait implements `to_str()` for the enum variants
/// let ifd2_str = ifd2.to_str();
/// assert_eq!(ifd2_str, "if-davinci-v2");
/// ```
/// Repeat the identifier and string literal pattern, `ident, literal`, as many times as you need:
/// Assume that the OpenAI's API expects the model ids `if-davinci-v2` and `babbage:2020-05-03`
///
/// ```rust
/// custom_model!(IfDavinciV2, "if-davinci-v2", Babbage_2020_05_03, "babbage:2020-05-03");
///
/// let ifd2 = CustomModel::IfDavinciV2;
/// let ifd2_str = ifd2.to_str();
/// assert_eq!(ifd2_str, "if-davinci-v2");
///
/// let babbage = CustomModel::Babbage_2020_05_03;
/// let babbage_str = babbage.to_str();
/// assert_eq!(babbage_str, "babbage:2020-05-03");
/// ```
#[macro_export]
macro_rules! custom_model {
    ($name:ident, $str_rep:literal) => {

        #[derive(Debug, Clone, Serialize, Deserialize)]
        pub enum CustomModel {
            $name,
        }

        impl RetrievableModel for CustomModel {
            fn to_str(&self) -> &str {
                match self {
                    CustomModel::$name => $str_rep,
                }
            }
        }
    };
    ($($name:ident, $str_rep:literal),*) => {

        #[derive(Debug, Clone, Serialize, Deserialize)]
        pub enum CustomModel {
            $($name),*
        }

        impl RetrievableModel for CustomModel {
            fn to_str(&self) -> &str {
                match self {
                    $(CustomModel::$name => $str_rep,)*
                }
            }
        }
    };
}
pub use custom_model;

/// Generates a `FineTunedModel` enum which implements the `RetrievableModel` trait. Pass in the identifier
/// you want to use as the enum variant, followed by a comma, followed by the string literal that the
/// API expects for that model (the model's id). You can repeat this pattern however many times you want.
///
/// This macro is intended to be used for fine-tuned models, which are created by the user and
/// which are not enumerated in the model enums. For models that might be generally available to all
/// users, use the `custom_model!` macro. Both macros currently generate the same code, with different
/// enum identifiers, but this may change in the future!
///
/// # Example
/// Assume your fine-tuned model has the following id: `curie:ft-personal-2023-02-18-20-10-18`
///
/// ```rust
/// ft_model!(CurieFtPersonal, "curie:ft-personal-2023-02-18-20-10-18");
///
/// let ft_curie = FineTunedModel::CurieFtPersonal;
/// // The RetrievableModel trait implements `to_str()` for the enum variants
/// let ft_curie_str = ft_model.to_str();
/// assert_eq!(ft_model_str, "curie:ft-personal-2023-02-18-20-10-18");
/// ```
///
/// Repeat the identifier and string literal pattern as many times as you need.
/// Assume your fine-tuned models have the following ids: `curie:ft-personal-2023-02-18-20-10-18`,
/// `davinci:ft-personal-2023-02-18-20-10-18`.
///
/// ```rust
/// ft_model!(CurieFtPersonal, "curie:ft-personal-2023-02-18-20-10-18", DavinciFtPersonal, "davinci:ft-personal-2023-02-18-20-10-18");
///
/// let ft_curie = FineTunedModel::CurieFtPersonal;
/// let ft_curie_str = ft_curie.to_str();
/// assert_eq!(ft_curie_str, "curie:ft-personal-2023-02-18-20-10-18");
///
/// let ft_davinci = FineTunedModel::DavinciFtPersonal;
/// let ft_davinci_str = ft_davinci.to_str();
/// assert_eq!(ft_davinci_str, "davinci:ft-personal-2023-02-18-20-10-18");
/// ```
#[macro_export]
macro_rules! ft_model {
    ($name:ident, $str_rep:literal) => {

        #[derive(Debug, Clone, Serialize, Deserialize)]
        pub enum FineTunedModel {
            $name,
        }

        impl RetrievableModel for FineTunedModel {
            fn to_str(&self) -> &str {
                match self {
                    FineTunedModel::$name => $str_rep,
                }
            }
        }
    };
    ($($name:ident, $str_rep:literal),*) => {

        #[derive(Debug, Clone, Serialize, Deserialize)]
        pub enum FineTunedModel {
            $($name),*
        }

        impl FineTunedModel {
            pub fn to_str(&self) -> &str {
                match self {
                    $(FineTunedModel::$name => $str_rep,)*
                }
            }
        }

        impl RetrievableModel for FineTunedModel {
            fn to_str(&self) -> &str {
                match self {
                    $(FineTunedModel::$name => $str_rep,)*
                }
            }
        }
    };
}
pub use ft_model;

// endregion: macros
