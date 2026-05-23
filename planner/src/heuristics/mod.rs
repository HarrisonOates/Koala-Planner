mod classical;
mod structs;

use crate::domain_description::{ClassicalDomain, DomainTasks};
use crate::task_network::{Task, HTN};
pub use structs::TDG;

pub use classical::{
    h_add,
    h_ff,
    h_lmcut,
    h_lmcut_full,
    h_lmcut_incremental,
    h_max,
    LandmarkCuts,
};
