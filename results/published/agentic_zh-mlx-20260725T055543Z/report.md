# Run report — agentic_zh (mlx)
- **run_id:** `20260725T055543Z`
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
| decode throughput (aggregate) | 18.1 tok/s |
| decode throughput (median) | 12.12 tok/s |
| TTFT median | 2.543 s |
| TTFT with tools / bare prompt | 2.557 / 0.622 s |
| mean generated tokens | 217.5 |
| peak allocator memory | 4739.8 MB |

## Cases
| id | kind | pass | detail | stop | tok/s | tokens |
|---|---|---|---|---|---|---|
| zh-tc-weather-beijing | tool_call | ✅ | args_subset | stop | 11.08 | 57 |
| zh-tc-weather-paris-select | tool_call | ✅ | args_subset | stop | 11.28 | 70 |
| zh-tc-email-bob | tool_call | ✅ | args_subset | stop | 15.24 | 199 |
| zh-tc-translate-japanese | tool_call | ✅ | args_subset | stop | 11.16 | 68 |
| zh-tc-time-shanghai | tool_call | ✅ | args_subset | stop | 11.29 | 71 |
| zh-tc-calendar-date | tool_call | ✅ | args_subset | stop | 12.95 | 150 |
| zh-tc-flight-bj-sh | tool_call | ✅ | args_subset | stop | 14.9 | 245 |
| zh-json-profile | json_schema | ✅ | schema_valid | stop | 17.66 | 880 |
