use super::*;
use std::collections::HashSet;

/// h_prob_max: probability upper bound via relaxed (delete-free) fixed-point.
/// Returns the maximum probability in [0.0, 1.0] of achieving all goal facts.
/// Returns 0.0 if any goal fact is unreachable (dead-end signal).
pub fn h_prob_max(domain: &ClassicalDomain, state: &HashSet<u32>, goal: &HashSet<u32>) -> f64 {
    if goal.is_empty() {
        return 1.0;
    }
    let n = domain.facts.count() as usize;
    let mut p = vec![0.0f64; n];
    for &f in state {
        if (f as usize) < n {
            p[f as usize] = 1.0;
        }
    }
    loop {
        let mut changed = false;
        for action in domain.actions.iter() {
            let prec_prob: f64 = if action.pre_cond.is_empty() {
                1.0
            } else {
                action.pre_cond.iter()
                    .map(|&f| if (f as usize) < n { p[f as usize] } else { 0.0 })
                    .fold(f64::INFINITY, f64::min)
            };
            if prec_prob == 0.0 {
                continue;
            }
            let action_prob = action.probabilities.get(0).copied().unwrap_or(1.0);
            let ep = prec_prob * action_prob;
            if action.add_effects.is_empty() {
                continue;
            }
            for &e in action.add_effects[0].iter() {
                if (e as usize) < n && p[e as usize] < ep {
                    p[e as usize] = ep;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    let goal_prob = goal.iter()
        .map(|&g| if (g as usize) < n { p[g as usize] } else { 0.0 })
        .fold(f64::INFINITY, f64::min);
    if goal_prob.is_infinite() { 0.0 } else { goal_prob.min(1.0) }
}
