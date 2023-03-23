use nom::{
    bytes::streaming::{tag, take_until},
    IResult,
};

pub use chat_parsers::*;

// It may be that much of this is applicable to parsing a completion stream too. In which
// case, will change mod layout.
pub mod chat_parsers {
    use super::*;

    /// Uses `nom` to attempt to parse out the `role` field from a server-sent stream.
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

    /// Uses `nom` to attempt to parse out only the `role` field from a server-sent stream.
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

    /// Uses `nom` to attempt to parse out the `content` field from a server-sent stream.
    ///
    /// # Returns
    /// `(input, content)` where `content` will be a `&[u8]` representation of the content object
    /// (or `b"{\"content\": \"Hello\"}"`) if successful.
    ///
    /// # Fails
    /// Returns an `Err::Incomplete` if the `content` field is not present.
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

    /// Uses `nom` to attempt to parse out `[DONE]` from a server-sent stream. This
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

    pub fn nom_chat_completion_chunk(input: &[u8]) -> IResult<&[u8], &[u8]> {
        let (obj_start, _) = take_until("{")(input)?;
        let (input, obj) = take_until("\n")(obj_start)?;

        Ok((input, obj))
    }
}

#[cfg(test)]
mod tests {

    use crate::completions::Role;

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
        let slice = no_delta().as_ref();
        match nom_chat_completion_chunk(slice) {
            Ok((remaining, chunk)) => {
                println!("remaining: {:?}", remaining);
                println!("chunk: {:?}", chunk);
                assert_eq!(remaining, b"");
                assert_eq!(chunk, b"{\"id\":\"chatcmpl-6sKy7POpwtespFLK7OTQCCwNeRIZv\",\"object\":\"chat.completion.chunk\",\"created\":1678408335,\"model\":\"gpt-3.5-turbo-0301\",\"choices\":[{\"delta\":{\"role\":\"assistant\"},\"index\":0,\"finish_reason\":null}]}\n\n");
            }
            Err(e) => {
                dbg!(e);
            }
        }
    }

    #[test]
    fn test_nom_unit() {
        let mut slice = complete_response().as_ref();
        let mut msg = vec![];
        let mut last_obj: Option<&[u8]> = None;
        let loop_count = 0;
        loop {
            println!("loop_count: {}", loop_count);
            match nom_chat_completion_chunk(slice) {
                Ok((remaining, chunk)) => {
                    slice = remaining;
                    last_obj = Some(chunk);
                    match nom_content_value(&chunk) {
                        Ok((_, content)) => {
                            let content_str = String::from_utf8(content.to_vec()).unwrap();
                            msg.push(content_str);
                        }
                        Err(e) => {
                            if e.is_incomplete() {
                                ()
                            } else {
                                dbg!(e);
                            }
                        }
                    }
                }
                Err(e) => {
                    if e.is_incomplete() {
                        break;
                    } else {
                        dbg!(e);
                    }
                }
            };
        }
        let last_obj_str = String::from_utf8(last_obj.unwrap().to_vec()).unwrap();
        println!("last_obj: {:?}", last_obj_str);

        let msg = msg.join("");
        println!("msg: {:?}", msg);
    }

    #[test]
    fn test_nom_response_chunk() {
        let mut slice = complete_response().as_ref();
        let mut count = 0;
        loop {
            let mut right;
            match nom_chat_completion_chunk(slice) {
                Ok((l, r)) => {
                    slice = l;
                    right = r;
                    let right_str = String::from_utf8(right.to_vec()).unwrap();
                    println!("right: {:?}", right_str);
                }
                Err(e) => {
                    if e.is_incomplete() {
                        break;
                    } else {
                        dbg!(e);
                    }
                }
            };
            // let left = String::from_utf8(slice.to_vec()).unwrap();

            // println!("left: {:?}", left);

            println!("");

            // count += 1;
            // if count == 10 {
            //     break;
            // }
        }
    }

    #[test]
    fn test_nom_content_til_done() {
        let mut slice = some_delta().as_ref();
        loop {
            match nom_content_value(slice) {
                Ok(r) => {
                    slice = r.0;
                    let content = match String::from_utf8(r.1.to_vec()) {
                        Ok(s) => s,
                        Err(e) => {
                            return {
                                dbg!(e.to_string());
                            };
                        }
                    };
                    println!("{}", content);
                }
                Err(e) => {
                    if e.is_incomplete() {
                        break;
                    } else {
                        dbg!(e);
                    }
                }
            };
        }
        match nom_til_done(slice) {
            Ok(r) => {
                slice = r.0;
                let content = match String::from_utf8(r.1.to_vec()) {
                    Ok(s) => s,
                    Err(e) => {
                        return {
                            dbg!(e.to_string());
                        };
                    }
                };
                println!("{}", content);
            }
            Err(e) => {
                if e.is_incomplete() {
                    let e_string = match String::from_utf8(e.to_string().into_bytes()) {
                        Ok(s) => s,
                        Err(e) => {
                            return {
                                dbg!(e.to_string());
                            };
                        }
                    };
                    println!("incomplete: {e}");
                } else {
                    dbg!(e);
                }
            }
        }
        // let slice = String::from_utf8(slice.to_vec()).unwrap();
        // println!("\nleftover slice: {:?}", slice);
    }

    #[test]
    fn test_nom_content() {
        let input = some_delta();
        let (input, content) = match nom_content(input) {
            Ok((input, content)) => (input, content),
            Err(e) => panic!("nom_content failed: {:?}", e),
        };

        let content = String::from_utf8(content.to_vec()).unwrap();
        dbg!(&content);
    }

    #[test]
    fn test_nom_content_value() {
        let input = some_delta();
        let (input, content_value) = match nom_content_value(input) {
            Ok((input, content_value)) => (input, content_value),
            Err(e) => panic!("nom_content_value failed: {:?}", e),
        };

        let content = String::from_utf8(content_value.to_vec()).unwrap();
        dbg!(&content);
    }

    #[test]
    fn test_nom_role() {
        let input = complete_response();
        let (input, role) = match nom_role(input) {
            Ok((input, role)) => (input, role),
            Err(e) => panic!("nom_role failed: {:?}", e),
        };

        let role: Role = Role::from_slice(role).unwrap();
        dbg!(&role);
    }

    #[test]
    fn test_nom_role_value() {
        let input = complete_response();

        let (input, role_value) = match nom_role_value(input) {
            Ok((input, role)) => (input, role),
            Err(e) => panic!("nom_role failed: {:?}", e),
        };

        let role = String::from_utf8(role_value.to_vec()).unwrap();
        dbg!(&role);
    }
}
