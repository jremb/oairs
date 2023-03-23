// TODO: Need to look closer at what's considered idiomatic Rust error handling. May need a lot of refactoring?

use reqwest::StatusCode;

use super::*;

#[derive(Debug, Clone, Serialize, Deserialize, SaveJson)]
pub struct OairsError {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
    pub param: Option<String>,
    pub code: Option<String>,
}

impl std::fmt::Display for OairsError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{:#?}", self)
    }
}

impl From<std::num::ParseIntError> for OairsError {
    fn from(e: std::num::ParseIntError) -> Self {
        OairsError::new(e.to_string(), ErrorType::Other, None, None)
    }
}

impl From<std::io::Error> for OairsError {
    fn from(e: std::io::Error) -> Self {
        OairsError::new(e.to_string(), ErrorType::FileError, None, None)
    }
}

impl OairsError {
    pub fn new(
        message: String,
        error_type: ErrorType,
        param: Option<String>,
        code: Option<String>,
    ) -> OairsError {
        OairsError {
            message,
            error_type: error_type.to_string(),
            param,
            code,
        }
    }
}

#[non_exhaustive]
#[derive(Debug)]
pub enum ErrorType {
    // OpenAI API Errors (<https://platform.openai.com/docs/guides/error-codes/error-codes>)
    ApiError,
    Timeout,
    RateLimit,
    APIConnection,
    InvalidRequest, // This should probably be reworked
    Authentication,
    ServiceUnavailable,
    // Errors that might arise from dependencies (or my use thereof!)
    DeserializationError,
    FileError,
    ReqwestError,
    SerializationError,
    Tokenizer,
    PolarsError,
    // Errors that might arise from this library
    ParseError,
    SaveError,
    ParamError,
    // Catch-all that should be factored out as more specific errors are added
    Other,
}

impl ErrorType {
    pub fn to_str(&self) -> &str {
        match self {
            ErrorType::ApiError => "API Error",
            ErrorType::Timeout => "Timeout",
            ErrorType::RateLimit => "Rate limit reached",
            ErrorType::APIConnection => "Issue connecting to API",
            ErrorType::InvalidRequest => "Invalid Request",
            ErrorType::Authentication => "Authentication",
            ErrorType::ServiceUnavailable => "Service Unavailable",
            ErrorType::ParseError => "Parse Error",
            ErrorType::SaveError => "Save Error",
            ErrorType::DeserializationError => "Deserialization Error",
            ErrorType::FileError => "File Error",
            ErrorType::ReqwestError => "Reqwest Error",
            ErrorType::SerializationError => "Serialization Error",
            ErrorType::Tokenizer => "Tokenizer Error",
            ErrorType::PolarsError => "Polars Error",
            ErrorType::ParamError => "Parameter Error",
            ErrorType::Other => "Other Error",
        }
    }
}

impl std::fmt::Display for ErrorType {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.to_str())
    }
}

// This represents how the OpenAI API returns an error in the case of an invalid request.
#[derive(Debug, Serialize, Deserialize, SaveJson)]
pub struct InvalidRequest {
    pub error: OairsError,
}

impl std::fmt::Display for InvalidRequest {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{:#?}", self)
    }
}

/// The OpenAI API returns a JSON object with an `error` field that contains the fields we've defined
/// in the `OairsError` struct. This function basically attempts to unwrap that into a simple `Error` struct.
pub(crate) fn handle_request_fail(body: &str, status_code: StatusCode) -> OairsError {
    match serde_json::from_str::<InvalidRequest>(body) {
        Ok(ir) => {
            if ir.error.code.is_none() {
                OairsError {
                    code: Some(status_code.as_u16().to_string()),
                    ..ir.error
                }
            } else {
                ir.error
            }
        }
        Err(e) => OairsError::new(
            format!("Error attempting to deserialize body: {}", e),
            ErrorType::DeserializationError,
            None,
            Some(status_code.to_string()),
        ),
    }
}

pub(crate) fn builder_error(e: reqwest::Error) -> OairsError {
    let status_code = if e.status().is_some() {
        Some(e.status().unwrap().to_string())
    } else {
        None
    };
    OairsError::new(e.to_string(), ErrorType::ReqwestError, None, status_code)
}

pub(crate) fn parse_reqwest_error(e: reqwest::Error) -> OairsError {
    if e.status().is_some() {
        let status = e.status().unwrap();
        let body = e.to_string();
        handle_request_fail(&body, status)
    } else {
        OairsError::new(
            e.to_string(),
            ErrorType::Other,
            Some("cf. message".to_string()),
            None,
        )
    }
}

pub(crate) async fn parse_api_error(
    response: reqwest::Response,
    status_code: reqwest::StatusCode,
) -> OairsError {
    let headers = response.headers().clone();

    let invalid_request = response.json::<InvalidRequest>().await.unwrap();
    let message = invalid_request.error.message;
    let param = invalid_request.error.param;
    let api_code = if invalid_request.error.code.is_some() {
        invalid_request.error.code.unwrap()
    } else {
        "".to_string()
    };
    let code = format!("{} {}", status_code, api_code);

    match status_code {
        reqwest::StatusCode::TOO_MANY_REQUESTS => {
            let remaining_requests = headers.get("x-ratelimit-remaining-requests");
            if let Some(remaining) = remaining_requests {
                let remaining = remaining.to_str().unwrap();
                let remaining = remaining.parse::<u32>().unwrap();
                if remaining == 0 {
                    let reset = headers.get("x-ratelimit-reset-requests").unwrap();
                    OairsError::new(
                        format!("Rate limit exceeded. API message: {}", message),
                        ErrorType::RateLimit,
                        Some(reset.to_str().unwrap().to_string()),
                        Some(code),
                    )
                } else {
                    OairsError::new(
                    format!(
                        "Status code 429 indicates that either the engine is currently overloaded or you have exceeded \
                        your current quota. Check your plan and billing. If you have not exceeded your quota, please try \
                        again later. If issue persists, please contact OpenAI support. API message: {}", message
                    ),
                    ErrorType::RateLimit,
                    param,
                    Some(code),
                )
                }
            } else {
                // Can't be more specific
                OairsError::new(
                    format!("429 - Rate Limit. API message: {}", message),
                    ErrorType::RateLimit,
                    param,
                    Some(code),
                )
            }
        }
        reqwest::StatusCode::SERVICE_UNAVAILABLE => OairsError::new(
            format!("Service Unavailable. API message: {}", message),
            ErrorType::ServiceUnavailable,
            param,
            Some(code),
        ),
        reqwest::StatusCode::UNAUTHORIZED => {
            let message = if message == "Invalid authorization header" {
                "Invalid authorization header. Did you forget to enter your API key?".to_string()
            } else {
                message
            };
            OairsError::new(message, ErrorType::Authentication, param, Some(code))
        }
        reqwest::StatusCode::NOT_FOUND => {
            OairsError::new(message, ErrorType::InvalidRequest, param, Some(code))
        }
        _ => OairsError::new(message, ErrorType::Other, param, Some(code)),
    }
}
