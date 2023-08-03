use thiserror::Error;

#[derive(Debug, Error)]
pub enum SIFTError {
    #[error("{0}")]
    Unsupported(String),
}

pub type Result<T, E = SIFTError> = std::result::Result<T, E>;