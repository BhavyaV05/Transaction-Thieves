## E18: backbone size sweep (validation only)

| candidate | family | params_b | active_b | quant | stack | size_gb | size_gb_source | n_alerts | brier | brier_ci_lo | brier_ci_hi | no_verdict_rate | valid_output_rate | usable_output_rate | valid_dsl_rate | json_parse_rate | median_s | p90_s | median_gpu_s | p90_gpu_s | calls_per_alert | tokens_in_per_alert | tokens_out_per_alert |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma4-12b | gemma | 12 | 12 | qat-w4a16 | vllm | 10.3 | config | 460 | 0.1741 | 0.1486 | 0.2018 | 0 | 1 | 1 | 1 | 1 | 5.5713 | 9.2707 | 5.5664 | 9.2642 | 1.7304 | 8816.713 | 238.7609 |
| qwen3-14b | qwen | 14 | 14 | awq | vllm | 10 | config | 460 | 0.189 | 0.1658 | 0.2145 | 0 | 1 | 0.9386 | 1 | 1 | 3.2449 | 9.9834 | 3.2416 | 9.9776 | 1.7348 | 8206.7761 | 189.1 |
| qwen3.8-27b-q3 | qwen | 27 | 27 | UD-Q3_K_XL | llama.cpp | 13.1 | config | 460 | 0.1844 | 0.1593 | 0.2109 | 0 | 1 | 1 | 1 | 1 | 5.4391 | 6.2514 | 5.4362 | 6.2487 | 1.15 | 5699.3826 | 141.6348 |
