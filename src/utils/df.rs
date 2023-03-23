use std::path::PathBuf;

use polars::prelude::{
    CsvReader, CsvWriter, DataFrame, LazyCsvReader, LazyFrame, ParquetReader, ParquetWriter,
    SerReader, SerWriter,
};

use super::*;

pub fn load_csv(path: PathBuf) -> Result<LazyFrame, OairsError> {
    match LazyCsvReader::new(&path).has_header(true).finish() {
        Ok(lf) => Ok(lf),
        Err(e) => {
            return Err(OairsError::new(
                e.to_string(),
                ErrorType::FileError,
                Some(path.to_string_lossy().to_string()),
                None,
            ))
        }
    }
}

pub fn load_parquet(path: &PathBuf) -> Result<DataFrame, OairsError> {
    let mut file = std::fs::File::open(path)?;
    match ParquetReader::new(&mut file).finish() {
        Ok(df) => Ok(df),
        Err(e) => {
            return Err(OairsError::new(
                e.to_string(),
                ErrorType::FileError,
                Some(path.to_string_lossy().to_string()),
                None,
            ))
        }
    }
}

/// Convert a [`Response`](https://docs.rs/reqwest/0.11.14/reqwest/struct.Response.html) that 
/// is the result of a [`retrieve_file_content`](crate::client::Client::retrieve_file_content)
/// request into a [`DataFrame`](https://docs.rs/polars/0.27.2/polars/prelude/struct.DataFrame.html).
pub async fn df_from_response(response: reqwest::Response) -> Result<DataFrame, OairsError> {
    let s = response.text().await.unwrap();
    df_from_string(s, true)
}

fn df_from_string(s: String, has_headers: bool) -> Result<DataFrame, OairsError> {
    let csv = s.as_bytes();
    let cur = std::io::Cursor::new(csv);

    match CsvReader::new(cur).has_header(has_headers).finish() {
        Ok(df) => Ok(df),
        Err(e) => {
            return Err(OairsError::new(
                e.to_string(),
                ErrorType::PolarsError,
                None,
                None,
            ))
        }
    }
}

/// Save a [`DataFrame`](https://docs.rs/polars/0.27.2/polars/prelude/struct.DataFrame.html) as a
/// parquet file. The `.parquet` extension will be added if it is not present in the `path` argument.
pub fn write_parquet(df: &mut DataFrame, path: &str) -> Result<u64, OairsError> {
    let path = if !path.ends_with(".parquet") {
        format!("{}.parquet", path)
    } else {
        path.to_string()
    };

    let writer = std::fs::File::create(&path).map_err(|e| {
        OairsError::new(
            e.to_string(),
            ErrorType::FileError,
            Some(path.to_string()),
            None,
        )
    })?;

    ParquetWriter::new(writer).finish(df).map_err(|e| {
        OairsError::new(
            e.to_string(),
            ErrorType::PolarsError,
            Some(path.to_string()),
            None,
        )
    })
}

/// Save a [`DataFrame`](https://docs.rs/polars/0.27.2/polars/prelude/struct.DataFrame.html) as a
/// csv file. The `.csv` extension will be added if it is not present in the `path` argument.
pub fn write_csv(df: &mut DataFrame, path: &str) -> Result<(), OairsError> {
    let path = if !path.ends_with(".csv") {
        format!("{}.csv", path)
    } else {
        path.to_string()
    };

    let mut file = std::fs::File::create(&path)?;
    match CsvWriter::new(&mut file).finish(df) {
        Ok(_) => Ok(()),
        Err(e) => {
            return Err(OairsError::new(
                e.to_string(),
                ErrorType::PolarsError,
                Some(path.to_string()),
                None,
            ))
        }
    }
}
