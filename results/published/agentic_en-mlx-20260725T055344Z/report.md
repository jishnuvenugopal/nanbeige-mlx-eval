# Run report — agentic_en (mlx)
- **run_id:** `20260725T055344Z`
- **model:** `models/nanbeige-mlx-8bit`
- **env:** python 3.12.12 · mlx 0.32.0 · mlx-lm 0.31.3 · arm64 · git 0a0cf2c
- **quantization:** 8-bit, group_size=64

## Summary
- pass rate: **8/8** (100.0%) [0.68, 1.00]

| grade kind | pass / n |
|---|---|
| json_schema | 1 / 1 |
| tool_call | 7 / 7 |

## Latency / memory (real-model cases only)
| metric | value |
|---|---|
| decode throughput (aggregate) | 18.89 tok/s |
| decode throughput (median) | 12.59 tok/s |
| TTFT median | 2.546 s |
| TTFT with tools / bare prompt | 2.542 / 0.625 s |
| mean generated tokens | 208.9 |
| peak allocator memory | 4741.4 MB |

## Cases
| id | kind | pass | detail | stop | tok/s | tokens |
|---|---|---|---|---|---|---|
| en-tc-weather-tokyo | tool_call | ✅ | args_subset | stop | 11.38 | 60 |
| en-tc-weather-paris-select | tool_call | ✅ | args_subset | stop | 11.57 | 74 |
| en-tc-email-alice | tool_call | ✅ | args_subset | stop | 14.48 | 153 |
| en-tc-translate-french | tool_call | ✅ | args_subset | stop | 10.92 | 65 |
| en-tc-time-tokyo | tool_call | ✅ | args_subset | stop | 11.0 | 66 |
| en-tc-calendar-date | tool_call | ✅ | args_subset | stop | 13.62 | 132 |
| en-tc-flight-sf | tool_call | ✅ | args_subset | stop | 14.32 | 159 |
| en-json-profile | json_schema | ✅ | schema_valid | stop | 18.54 | 962 |
