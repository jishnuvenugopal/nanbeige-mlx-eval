# Run report — agentic_zh (mlx)
- **run_id:** `20260725T055137Z`
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
| decode throughput (aggregate) | 22.54 tok/s |
| decode throughput (median) | 13.46 tok/s |
| TTFT median | 2.664 s |
| TTFT with tools / bare prompt | 2.638 / 0.623 s |
| mean generated tokens | 188 |
| peak allocator memory | 3782.0 MB |

## Cases
| id | kind | pass | detail | stop | tok/s | tokens |
|---|---|---|---|---|---|---|
| zh-tc-weather-beijing | tool_call | ✅ | args_subset | stop | 11.71 | 57 |
| zh-tc-weather-paris-select | tool_call | ✅ | args_subset | stop | 12.21 | 70 |
| zh-tc-email-bob | tool_call | ✅ | args_subset | stop | 19.57 | 402 |
| zh-tc-translate-japanese | tool_call | ✅ | args_subset | stop | 12.15 | 68 |
| zh-tc-time-shanghai | tool_call | ✅ | args_subset | stop | 12.41 | 71 |
| zh-tc-calendar-date | tool_call | ✅ | args_subset | stop | 14.5 | 115 |
| zh-tc-flight-bj-sh | tool_call | ✅ | args_subset | stop | 15.93 | 127 |
| zh-json-profile | json_schema | ✅ | schema_valid | stop | 21.63 | 594 |
