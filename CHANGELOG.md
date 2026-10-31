# Revised portfolio edition - 2026-10-31

- Corrected buyer/seller next-offer bounds to use the previous offer plus/minus remaining budget.
- Added parameter validation and legal-action checking.
- Replaced h3 feasibility-zone gap with unresolved bid-ask pressure. This changes the heuristic definition and is not a probability model.
- Added h1 and h2 at depth 3 for comparison with h3 at the same depth.
- Regenerated all outputs and replaced the report. Historical benchmark numbers are superseded, not silently preserved.
- Added six regression/algorithm checks, CI, pinned dependencies, and a PDF generation script.
- Retained asymmetric_relative exit scoring and the original round-boundary convention, explicitly documented as limitations.

## Node-count provenance
The earlier report quoted 1,445.22 Minimax nodes, whereas its supplied code regenerated a different count before revisions. The historical origin was not established. No historical correctness claim is made. This edition uses the included code's recursive-visit counter and regenerated data consistently.
