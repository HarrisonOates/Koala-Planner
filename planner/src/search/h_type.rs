use crate::heuristics::{h_add, h_ff, h_lmcut, h_max};
use crate::search::fixed_method::heuristic_factory::ClassicalHeuristic;

pub enum HeuristicType {
    HFF,
    HAdd,
    HMax,
    HProb,
    HLMCut,
}

impl HeuristicType {
    pub fn as_classical_fn(&self) -> ClassicalHeuristic {
        match self {
            HeuristicType::HFF    => h_ff,
            HeuristicType::HAdd   => h_add,
            HeuristicType::HMax   => h_max,
            HeuristicType::HLMCut => h_lmcut,
            HeuristicType::HProb  => {
                panic!("--prob heuristic is not supported with --fixed mode")
            }
        }
    }
}
