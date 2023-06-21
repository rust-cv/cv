use std::{error::Error, fmt::Display};


#[derive(Debug)]
pub enum SIFTError {
    Unsupported(String)
}

impl Display for SIFTError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:#?}", self)
    }
}

impl Error for SIFTError {}

pub type Result<T> = std::result::Result<T, SIFTError>;