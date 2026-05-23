use std::time::Duration;

pub struct SearchStats {
    pub max_depth: u16,
    pub search_nodes: u32,
    pub explored_nodes: u32,
    pub seach_time: Duration,
    pub success_probability: Option<f64>,
    pub rho_threshold: f64,
}

impl std::fmt::Display for SearchStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        writeln!(f, "max depth: {}", self.max_depth)?;
        writeln!(f, "# of search nodes: {}", self.search_nodes)?;
        writeln!(f, "# of explored nodes: {}", self.explored_nodes)?;
        let time = self.seach_time.as_secs_f64();
        writeln!(f, "search duration: {}", time.trunc())?;
        if let Some(prob) = self.success_probability {
            writeln!(f, "success probability: {:.4}", prob)?;
        }
        if self.rho_threshold < 1.0 {
            writeln!(f, "rho threshold: {:.4}", self.rho_threshold)?;
        }
        Ok(())
    }
}
