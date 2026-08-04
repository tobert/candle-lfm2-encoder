//! Shared `id2label` handling for the classification heads.
//!
//! Both [`crate::token_classification::Lfm2TokenClassifier`] and
//! [`crate::sequence_classification::Lfm2SequenceClassifier`] load a
//! `config.json` `id2label` map (`{"0": "O", "1": "B-x", …}`) and need it as
//! a dense, id-ordered `Vec<String>` before it can index a classifier head.
//! One helper, used by both, so a gap-detection fix only has to happen once.

use std::collections::HashMap;
use std::path::Path;

use crate::error::{Error, Result};

/// Turn `{"0": "O", "1": "B-x", …}` into a dense id-ordered vector, failing
/// loudly on a gap rather than leaving a silently-wrong label at that id.
pub(crate) fn order_labels(map: &HashMap<String, String>, path: &Path) -> Result<Vec<String>> {
    let mut out = vec![None; map.len()];
    for (k, v) in map {
        let id: usize = k.parse().map_err(|_| Error::ConfigInvalid {
            path: path.to_path_buf(),
            message: format!("id2label key {k:?} is not an integer"),
        })?;
        if id >= out.len() {
            return Err(Error::ConfigInvalid {
                path: path.to_path_buf(),
                message: format!("id2label id {id} is out of range for {} labels", map.len()),
            });
        }
        out[id] = Some(v.clone());
    }
    out.into_iter()
        .enumerate()
        .map(|(i, l)| {
            l.ok_or_else(|| Error::ConfigInvalid {
                path: path.to_path_buf(),
                message: format!("id2label has no entry for id {i}"),
            })
        })
        .collect()
}
