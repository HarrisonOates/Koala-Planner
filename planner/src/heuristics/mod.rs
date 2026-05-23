mod classical;
mod structs;

use crate::domain_description::{ClassicalDomain, DomainTasks};
use crate::task_network::{Applicability, CompoundTask, PrimitiveAction, Task, HTN};
pub use structs::TDG;

pub use classical::{
    h_add,
    h_ff,
    h_max,
    h_prob_max,
};
