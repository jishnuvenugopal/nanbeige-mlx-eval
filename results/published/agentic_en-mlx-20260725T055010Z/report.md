# Run report — agentic_en (mlx)
- **run_id:** `20260725T055010Z`
- **model:** `models/nanbeige-mlx-6bit`
- **env:** python 3.12.12 · mlx 0.32.0 · mlx-lm 0.31.3 · arm64 · git 0a0cf2c
- **quantization:** 6-bit, group_size=64

## Summary
- pass rate: **8/8** (100.0%) [0.68, 1.00]

| grade kind | pass / n |
|---|---|
| json_schema | 1 / 1 |
| tool_call | 7 / 7 |

## Latency / memory (real-model cases only)
| metric | value |
|---|---|
| decode throughput (aggregate) | 23.15 tok/s |
| decode throughput (median) | 14.04 tok/s |
| TTFT median | 2.636 s |
| TTFT with tools / bare prompt | 2.642 / 0.624 s |
| mean generated tokens | 164.6 |
| peak allocator memory | 3781.6 MB |

## Cases
| id | kind | pass | detail | stop | tok/s | tokens |
|---|---|---|---|---|---|---|
| en-tc-weather-tokyo | tool_call | ✅ | args_subset | stop | 12.58 | 60 |
| en-tc-weather-paris-select | tool_call | ✅ | args_subset | stop | 12.74 | 74 |
| en-tc-email-alice | tool_call | ✅ | args_subset | stop | 17.47 | 198 |
| en-tc-translate-french | tool_call | ✅ | args_subset | stop | 11.91 | 65 |
| en-tc-time-tokyo | tool_call | ✅ | args_subset | stop | 12.07 | 66 |
| en-tc-calendar-date | tool_call | ✅ | args_subset | stop | 15.34 | 132 |
| en-tc-flight-sf | tool_call | ✅ | args_subset | stop | 16.52 | 161 |
| en-json-profile | json_schema | ✅ | schema_valid | stop | 22.6 | 561 |
