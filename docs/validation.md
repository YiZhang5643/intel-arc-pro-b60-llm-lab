# Publication checks · 2026-10-25

- Six standard-library unittest cases passed against a mock HTTP server: exact-token streaming prefill, cache-contamination rejection, deterministic long-chat inputs and cache accounting, opt-in generated-code execution, preset command construction, and published evidence verification.
- SHA256 evidence checks passed for 39 files; central report numbers and 96K cache/retrieval assertions matched the measurements.
- The portable patch adapter regenerated the measured adapted Q4 patch exactly after newline normalization, using copies of the original source files and original external patch. The check did not modify the deployed source or running service.
- Python syntax and private-path/credential-pattern scans passed. Such scans complement an explicit publication allowlist, not a guarantee of detecting all secrets.
- This publication pass did not rerun GPU compilation or inference. Performance and numerical correctness figures refer to the historical evidence, not the mock tests.
