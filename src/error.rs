//! Crate error type.
//!
//! Every variant names the artifact it failed on. A checkpoint directory
//! that's missing a file, carries a config that disagrees with its weights,
//! or ships a tokenizer we can't read should say which, not surface a bare
//! `No such file or directory`.

use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("reading {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("parsing {path}: {source}")]
    ConfigParse {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },

    /// A config that parsed but describes something incoherent.
    #[error("{path}: {message}")]
    ConfigInvalid { path: PathBuf, message: String },

    #[error("loading tokenizer {path}: {message}")]
    Tokenizer { path: PathBuf, message: String },

    #[error("tokenizing: {0}")]
    Encode(String),

    #[error(transparent)]
    Candle(#[from] candle_core::Error),
}

pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    pub(crate) fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Self::Io {
            path: path.into(),
            source,
        }
    }
}
