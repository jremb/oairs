use std::borrow::Cow;

use polars::{
    prelude::{DataFrame, NamedFrom},
    series::Series,
};
use serde::ser::SerializeMap;

use super::*;

/// Struct for deserializing a successful call to the embeddings endpoint.
#[derive(Debug, Serialize, Clone, Deserialize, SaveJson)]
pub struct Embedding {
    pub data: Vec<EmbeddingObject>,
    pub model: String,
    pub object: String,
    pub usage: Usage,
}

impl Embedding {
    /// Method for extracting the embeddings array from an `EmbeddingsResponse`,
    /// returning a `Vec<Vec<f64>>`. Note that this method does not consume self
    /// and, therefore, clones the inner `Vec<f64>`s.
    pub fn extract_embeddings(&self) -> Vec<Vec<f64>> {
        self.data
            .iter()
            .map(|e| e.embedding.clone())
            .collect::<Vec<Vec<f64>>>()
    }

    /// NOTE: The order of the inputs must match the order from which the
    /// embeddings were created, otherwise the output will be incorrect!
    pub fn append_input<'a, T>(&mut self, inputs: Vec<T>)
    where
        T: Into<Cow<'a, str>>,
    {
        self.data
            .iter_mut()
            .zip(inputs.into_iter())
            .for_each(|(e, i)| {
                e.input = Some(i.into().to_string());
            });
    }

    /// Appends the input to the `EmbeddingObject`s in `self.data` and then
    /// saves the `Embedding` to a JSON file at the specified `path`. Note that
    /// the order of the inputs must match the order from which the embeddings
    /// were created, otherwise the output will be incorrect!
    pub fn save_with_inputs<'a, T>(
        &mut self,
        path: &str,
        inputs: Vec<T>,
    ) -> Result<(), std::io::Error>
    where
        T: Into<Cow<'a, str>>,
    {
        self.append_input(inputs);
        self.save_json(path)
    }

    // TODO: Consider whether this can be made more efficient.
    /// Method for producing a Polars `DataFrame` from the embedding field of
    /// each `EmbeddingObject` (in the `data` field of `Embedding`). Consumes
    /// self. For a non-consuming method, you can use `extract_embeddings` and
    /// then  convert the `Vec<Vec<f64>>` to a `DataFrame`.
    pub fn embeddings_to_df(self) -> Result<DataFrame, OairsError> {
        let series = self
            .data
            .into_iter()
            .enumerate()
            .map(|(idx, e)| {
                let name = e.clone().input.unwrap_or_else(|| format!("input_{}", idx));
                e.to_series(Some(name))
            })
            .collect::<Vec<_>>();

        let df = DataFrame::new(series)
            .map_err(|e| OairsError::new(e.to_string(), ErrorType::PolarsError, None, None))?;

        Ok(df)
    }

    /// Converts EmbeddingResponse into a Polars DataFrame and saves it as a
    /// parquet file.
    pub fn save_parquet(self, path: &str) -> Result<u64, OairsError> {
        let mut df = self.embeddings_to_df()?;
        write_parquet(&mut df, path)
    }

    /// Appends the input to the `EmbeddingObject`s in `self.data` and then
    /// converts the `Embedding` into a Polars DataFrame and saves it as a
    /// parquet file. The inputs will be the column names for the respective
    /// embedding.
    pub fn save_parquet_with_input(
        mut self,
        path: &str,
        inputs: Vec<String>,
    ) -> Result<u64, OairsError> {
        self.append_input(inputs);
        let mut df = self.embeddings_to_df()?;
        write_parquet(&mut df, path)
    }
}

#[derive(Debug, Clone, Deserialize, SaveJson)]
pub struct EmbeddingObject {
    pub input: Option<String>,
    pub embedding: Vec<f64>,
    pub index: u32,
    pub object: String,
}

// Because I don't want the output to be cluttered with
// {
// "input": Some(
//     ...
// )
// ...
// }
impl Serialize for EmbeddingObject {
    fn serialize<Ser>(&self, serializer: Ser) -> Result<Ser::Ok, Ser::Error>
    where
        Ser: serde::Serializer,
    {
        let mut map = serializer.serialize_map(Some(4))?;
        if let Some(input) = &self.input {
            map.serialize_entry("input", input)?;
        }
        map.serialize_entry("embedding", &self.embedding)?;
        map.serialize_entry("index", &self.index)?;
        map.serialize_entry("object", &self.object)?;
        map.end()
    }
}

impl EmbeddingObject {
    pub fn to_series(self, name: Option<String>) -> Series {
        let name = name.unwrap_or_else(|| format!("{}", self.index));
        Series::new(name.as_ref(), self.embedding)
    }
}
