use nom::{
    bytes::streaming::{tag, take_until},
    IResult,
};

pub use chat_parsers::*;

// It may be that much of this is applicable to parsing a completion stream too. In which
// case, will change mod layout.
pub mod chat_parsers {
    use super::*;

    /// Uses [`nom`] to attempt to parse out the `role` field from a server-sent stream.
    ///
    /// # Returns
    /// `(input, role)` where `role` will be a `&[u8]` representation of the role object
    /// (or `b"{\"role\": \"assistant\"}"`) if successful. This can be used to create a `Role` object
    /// if converted to a utf-8 str.
    ///
    /// # Fails
    /// Returns an `Err::Incomplete` if the `role` field is not present.
    ///
    /// # Example
    /// ```rust,no_run
    /// let slice = ...;
    /// let (remaining, role) = nom_role(slice).unwrap();
    /// let role = Role::from_utf8(role).unwrap();
    pub fn nom_role(input: &[u8]) -> IResult<&[u8], &[u8]> {
        let (input, _) = take_until("{\"role")(input)?;
        take_until(",")(input)
    }

    /// Uses [`nom`] to attempt to parse out only the `role` field from a server-sent stream.
    ///
    /// # Returns
    /// `(input, role_value)` where `role` will be a `&[u8]` representation of the value for the
    /// `role` field (e.g., `assistant`).
    ///
    /// # Fails
    /// Returns an `Err::Incomplete` if the `role` field is not present.
    pub fn nom_role_value(input: &[u8]) -> IResult<&[u8], &[u8]> {
        let (input, _) = take_until("role")(input)?;
        let (role_start, _) = tag("role\":\"")(input)?;
        take_until("\"}")(role_start)
    }

    /// Uses [`nom`] to attempt to parse out the `content` object from a server-sent stream.
    ///
    /// # Returns
    /// `(input, content)` where `content` will be a `&[u8]` representation of the content object
    /// (or `b"{\"content\": \"Hello\"}"`) if successful.
    ///
    /// # Fails
    /// Returns an `Err::Incomplete` if the `content` field is not present.
    /// 
    /// # Example
    /// ```rust
    /// // Part of a server-sent event:
    /// let slice = b"data: {\"id\":\"chatcmpl-6sKy7POpwtespFLK7OTQCCwNeRIZv\",\"object\":\"chat.completion.chunk\",\"created\":1678408335,\"model\":\"gpt-3.5-turbo-0301\",\"choices\":[{\"delta\":{\"content\":\"Hello\"},\"index\":0,\"finish_reason\":null}]}";
    /// let (_, content) = nom_content(slice).unwrap();
    /// assert_eq!(content, b"{\"content\": \"Hello\"}");
    /// ```
    pub fn nom_content(input: &[u8]) -> IResult<&[u8], &[u8]> {
        let (input, _) = take_until("{\"content")(input)?;
        take_until(",")(input)
    }

    pub fn nom_content_value(input: &[u8]) -> IResult<&[u8], &[u8]> {
        let (input, _) = take_until("content")(input)?;
        let (content_start, _) = tag("content\":\"")(input)?;
        take_until("\"}")(content_start)
    }

    /// `input` is assumed to be `&[u8]` from the `get_delta` method.
    ///
    /// # Returns
    /// Returns either the value of the `role` field or the `content` field
    /// in a `delta` object, whichever is present.
    pub fn nom_delta_value(input: &[u8]) -> IResult<&[u8], &[u8]> {
        let (input, _) = take_until(":\"")(input)?;
        let (content_start, _) = tag(":\"")(input)?;
        take_until("\"")(content_start)
    }

    /// Uses [`nom`] to attempt to parse out `[DONE]` from a server-sent stream. This
    /// will consume any bytes that precede `[DONE]` in the `input`, so it is important to
    /// ensure that you don't need to process any data that might accompany a chunk with
    /// `b"[DONE]"` in it.
    ///
    /// # Returns
    /// `(remaining, [DONE])` where `[DONE]` is represented as `&[u8]` and `remaining` will be any
    /// remaining slice from the input (typically `b"\n\n"`).
    ///
    /// # Fails
    /// Returns an `Err::Incomplete` if `[DONE]` is not present.
    pub fn nom_til_done(input: &[u8]) -> IResult<&[u8], &[u8]> {
        let (input, _) = take_until("[DONE]")(input)?;
        tag("[DONE]")(input)
    }

    /// Convenicence function to determine if a chunk of bytes contains `[DONE]`, without needing
    /// to unwrap the result yourself. NOTE: assumes Err::Incomplete
    pub fn nom_is_done(input: &[u8]) -> bool {
        match nom_til_done(input) {
            Ok(_) => true,
            Err(_e) => false,
        }
    }

    /// Uses [`nom`] to attempt to parse out a single chat completion chunk from a server-sent 
    /// event.
    /// 
    /// # Returns
    /// `(remaining, completion_chunk)` where `completion_chunk` will be a `&[u8]` representation
    /// of the completion chunk object and `remaining` will be any remaining slice from the input. 
    /// 
    /// # Fails
    /// 
    /// Returns an `Err<Error<&[u8]>>` with the message 'Parsing requires more data' if no chat 
    /// completion chunk in input. (You can use this to determine when to break out of a loop)
    /// 
    /// # Example
    /// ```rust
    /// // A server-sent event with two chat completion chunks:
    /// let server_sent_event = b"data: {\"id\":\"chatcmpl-6sHKShQDKQCki9wzsdyALS3ublOkh\",\"object\":\"chat.completion.chunk\",\"created\":1678394344,\"model\":\"gpt-3.5-turbo-0301\",\"choices\":[{\"delta\":{},\"index\":0,\"finish_reason\":\"stop\"}]}\n\ndata: {\"id\":\"chatcmpl-6sHKShQDKQCki9wzsdyALS3ublOkh\",\"object\":\"chat.completion.chunk\",\"created\":1678394344,\"model\":\"gpt-3.5-turbo-0301\",\"choices\":[{\"delta\":{},\"index\":0,\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n";
    /// let (remaining, completion_chunk) = nom_chat_completion_chunk(server_sent_event).unwrap();
    /// 
    /// let expected_remaining = b"\n\ndata: {\"id\":\"chatcmpl-6sHKShQDKQCki9wzsdyALS3ublOkh\",\"object\":\"chat.completion.chunk\",\"created\":1678394344,\"model\":\"gpt-3.5-turbo-0301\",\"choices\":[{\"delta\":{},\"index\":0,\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n";
    /// let expected_chunk = b"{\"id\":\"chatcmpl-6sHKShQDKQCki9wzsdyALS3ublOkh\",\"object\":\"chat.completion.chunk\",\"created\":1678394344,\"model\":\"gpt-3.5-turbo-0301\",\"choices\":[{\"delta\":{},\"index\":0,\"finish_reason\":\"stop\"}]}";
    ///
    /// assert_eq!(remaining, expected_remaining);
    /// assert_eq!(completion_chunk, expected_chunk);
    /// ```
    pub fn nom_chat_completion_chunk(input: &[u8]) -> IResult<&[u8], &[u8]> {
        let (obj_start, _) = take_until("{")(input)?;
        let (input, obj) = take_until("\n")(obj_start)?;

        Ok((input, obj))
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    // region: helpers

    fn no_delta<'a>() -> &'a [u8; 406] {
        b"data: {\"id\":\"chatcmpl-6sHKShQDKQCki9wzsdyALS3ublOkh\",\"object\":\"chat.completion.chunk\",\"created\":1678394344,\"model\":\"gpt-3.5-turbo-0301\",\"choices\":[{\"delta\":{},\"index\":0,\"finish_reason\":\"stop\"}]}\n\ndata: {\"id\":\"chatcmpl-6sHKShQDKQCki9wzsdyALS3ublOkh\",\"object\":\"chat.completion.chunk\",\"created\":1678394344,\"model\":\"gpt-3.5-turbo-0301\",\"choices\":[{\"delta\":{},\"index\":0,\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n"
    }

    fn some_delta<'a>() -> &'a [u8; 423] {
        b"data: {\"id\":\"chatcmpl-6sKy7POpwtespFLK7OTQCCwNeRIZv\",\"object\":\"chat.completion.chunk\",\"created\":1678408335,\"model\":\"gpt-3.5-turbo-0301\",\"choices\":[{\"delta\":{\"role\":\"assistant\"},\"index\":0,\"finish_reason\":null}]}\n\ndata: {\"id\":\"chatcmpl-6sKy7POpwtespFLK7OTQCCwNeRIZv\",\"object\":\"chat.completion.chunk\",\"created\":1678408335,\"model\":\"gpt-3.5-turbo-0301\",\"choices\":[{\"delta\":{\"content\":\"Hello\"},\"index\":0,\"finish_reason\":null}]}\n\n"
    }

    fn complete_response<'a>() -> &'a [u8; 2251] {
        b"{\"id\":\"chatcmpl-6sNfOBbKza2mImLMAWeoC37QEjDeY\",\"object\":\"chat.completion.chunk\",\"created\":1678418706,\"model\":\"gpt-3.5-turbo-0301\",\"choices\":[{\"delta\":{\"role\":\"assistant\"},\"index\":0,\"finish_reason\":null}]}\n\n{\"id\":\"chatcmpl-6sNfOBbKza2mImLMAWeoC37QEjDeY\",\"object\":\"chat.completion.chunk\",\"created\":1678418706,\"model\":\"gpt-3.5-turbo-0301\",\"choices\":[{\"delta\":{\"content\":\"Is\"},\"index\":0,\"finish_reason\":null}]}\n\n{\"id\":\"chatcmpl-6sNfOBbKza2mImLMAWeoC37QEjDeY\",\"object\":\"chat.completion.chunk\",\"created\":1678418706,\"model\":\"gpt-3.5-turbo-0301\",\"choices\":[{\"delta\":{\"content\":\" there\"},\"index\":0,\"finish_reason\":null}]}\n\n{\"id\":\"chatcmpl-6sNfOBbKza2mImLMAWeoC37QEjDeY\",\"object\":\"chat.completion.chunk\",\"created\":1678418706,\"model\":\"gpt-3.5-turbo-0301\",\"choices\":[{\"delta\":{\"content\":\" something\"},\"index\":0,\"finish_reason\":null}]}\n\n{\"id\":\"chatcmpl-6sNfOBbKza2mImLMAWeoC37QEjDeY\",\"object\":\"chat.completion.chunk\",\"created\":1678418706,\"model\":\"gpt-3.5-turbo-0301\",\"choices\":[{\"delta\":{\"content\":\" I\"},\"index\":0,\"finish_reason\":null}]}\n\n{\"id\":\"chatcmpl-6sNfOBbKza2mImLMAWeoC37QEjDeY\",\"object\":\"chat.completion.chunk\",\"created\":1678418706,\"model\":\"gpt-3.5-turbo-0301\",\"choices\":[{\"delta\":{\"content\":\" can\"},\"index\":0,\"finish_reason\":null}]}\n\n{\"id\":\"chatcmpl-6sNfOBbKza2mImLMAWeoC37QEjDeY\",\"object\":\"chat.completion.chunk\",\"created\":1678418706,\"model\":\"gpt-3.5-turbo-0301\",\"choices\":[{\"delta\":{\"content\":\" assist\"},\"index\":0,\"finish_reason\":null}]}\n\n{\"id\":\"chatcmpl-6sNfOBbKza2mImLMAWeoC37QEjDeY\",\"object\":\"chat.completion.chunk\",\"created\":1678418706,\"model\":\"gpt-3.5-turbo-0301\",\"choices\":[{\"delta\":{\"content\":\" you\"},\"index\":0,\"finish_reason\":null}]}\n\n{\"id\":\"chatcmpl-6sNfOBbKza2mImLMAWeoC37QEjDeY\",\"object\":\"chat.completion.chunk\",\"created\":1678418706,\"model\":\"gpt-3.5-turbo-0301\",\"choices\":[{\"delta\":{\"content\":\" with\"},\"index\":0,\"finish_reason\":null}]}\n\n{\"id\":\"chatcmpl-6sNfOBbKza2mImLMAWeoC37QEjDeY\",\"object\":\"chat.completion.chunk\",\"created\":1678418706,\"model\":\"gpt-3.5-turbo-0301\",\"choices\":[{\"delta\":{\"content\":\"?\"},\"index\":0,\"finish_reason\":null}]}\n\n{\"id\":\"chatcmpl-6sNfOBbKza2mImLMAWeoC37QEjDeY\",\"object\":\"chat.completion.chunk\",\"created\":1678418706,\"model\":\"gpt-3.5-turbo-0301\",\"choices\":[{\"delta\":{},\"index\":0,\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n"
    }

    // endregion

    #[test]
    fn test_nom_chat_completion_chunk() {
        let mut slice = no_delta().as_ref();
        match nom_chat_completion_chunk(slice) {
            Ok((remaining, chunk)) => {
                let expected_remaining = b"\n\ndata: {\"id\":\"chatcmpl-6sHKShQDKQCki9wzsdyALS3ublOkh\",\"object\":\"chat.completion.chunk\",\"created\":1678394344,\"model\":\"gpt-3.5-turbo-0301\",\"choices\":[{\"delta\":{},\"index\":0,\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n";
                let expected_chunk = b"{\"id\":\"chatcmpl-6sHKShQDKQCki9wzsdyALS3ublOkh\",\"object\":\"chat.completion.chunk\",\"created\":1678394344,\"model\":\"gpt-3.5-turbo-0301\",\"choices\":[{\"delta\":{},\"index\":0,\"finish_reason\":\"stop\"}]}";

                assert_eq!(remaining, expected_remaining);
                assert_eq!(chunk, expected_chunk);
            }
            Err(e) => {
                panic!("{e}");
            }
        }

        let expecting_first = b"{\"id\":\"chatcmpl-6sHKShQDKQCki9wzsdyALS3ublOkh\",\"object\":\"chat.completion.chunk\",\"created\":1678394344,\"model\":\"gpt-3.5-turbo-0301\",\"choices\":[{\"delta\":{},\"index\":0,\"finish_reason\":\"stop\"}]}";
        let expecting_second = b"{\"id\":\"chatcmpl-6sHKShQDKQCki9wzsdyALS3ublOkh\",\"object\":\"chat.completion.chunk\",\"created\":1678394344,\"model\":\"gpt-3.5-turbo-0301\",\"choices\":[{\"delta\":{},\"index\":0,\"finish_reason\":\"stop\"}]}";
        
        let mut c = 0;
        loop
        {
            match nom_chat_completion_chunk(slice) {
                Ok((remaining, chunk)) => {
                    slice = remaining;
                    if c == 0 {
                        assert_eq!(chunk, expecting_first);
                    } else if c == 1 {
                        assert_eq!(chunk, expecting_second);
                    } else {
                        panic!("Should not have gotten here");
                    }
                }
                Err(e) => {
                    if e.to_string() == "Parsing requires more data" {
                        break;
                    }
                    panic!("{e}")
                }
            }
            c += 1;
        }
    }

    #[test]
    fn test_nom_is_done() {
        let slice = no_delta().as_ref();
        assert!(nom_is_done(slice));

        let slice = some_delta().as_ref();
        assert!(!nom_is_done(slice));
    }

    #[test]
    fn test_nom_content() {
        let slice = some_delta().as_ref();
        let (_, content) = nom_content(slice).unwrap();
        assert_eq!(content, b"{\"content\":\"Hello\"}");

        let slice = complete_response().as_ref();
        let (_, content) = nom_content(slice).unwrap();
        assert_eq!(content, b"{\"content\":\"Is\"}");

        let slice = no_delta().as_ref();
        match nom_content(slice) {
            Ok(_) => {
                panic!("Should not have gotten here");
            }
            Err(e) => {
                let expected = "Parsing requires more data";
                assert_eq!(e.to_string(), expected);
            }
        }
    }
}
