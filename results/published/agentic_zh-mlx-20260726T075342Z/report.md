# Run report — agentic_zh (mlx)
- **run_id:** `20260726T075342Z`
- **model:** `models/nanbeige-mlx-4bit`
- **env:** python 3.12.12 · mlx 0.32.0 · mlx-lm 0.31.3 · arm64 · git c800313
- **quantization:** 4-bit, group_size=64

## Summary
- pass rate: **27/30** (90.0%) [0.74, 0.97]
- ⚠️ **1 case(s) truncated** (hit token cap, no eos): zh-tc-email-wangwu

| grade kind | pass / n |
|---|---|
| exact_match | 1 / 1 |
| json_schema | 2 / 2 |
| tool_call | 24 / 27 |

## Latency / memory (real-model cases only)
| metric | value |
|---|---|
| decode throughput (aggregate) | 35.83 tok/s |
| decode throughput (median) | 20.0 tok/s |
| TTFT median | 2.291 s |
| TTFT with tools / bare prompt | 2.277 / 0.568 s |
| mean generated tokens | 167.6 |
| peak allocator memory | 2851.8 MB |

## Cases
| id | kind | pass | detail | stop | tok/s | tokens |
|---|---|---|---|---|---|---|
| zh-tc-weather-beijing | tool_call | ✅ | args_subset | stop | 16.33 | 58 |
| zh-tc-weather-paris-select | tool_call | ✅ | args_subset | stop | 16.57 | 71 |
| zh-tc-email-bob | tool_call | ✅ | args_subset | stop | 26.01 | 228 |
| zh-tc-translate-japanese | tool_call | ✅ | args_subset | stop | 16.1 | 68 |
| zh-tc-time-shanghai | tool_call | ✅ | args_subset | stop | 16.72 | 73 |
| zh-tc-calendar-date | tool_call | ✅ | args_subset | stop | 20.03 | 113 |
| zh-tc-flight-bj-sh | tool_call | ✅ | args_subset | stop | 22.27 | 134 |
| zh-json-profile | json_schema | ✅ | schema_valid | stop | 35.42 | 775 |
| zh-tc-weather-shanghai | tool_call | ✅ | args_subset | stop | 14.76 | 58 |
| zh-tc-weather-guangzhou | tool_call | ✅ | args_subset | stop | 17.05 | 69 |
| zh-tc-email-lisi | tool_call | ✅ | args_subset | stop | 21.17 | 129 |
| zh-tc-email-wangwu | tool_call | ❌ | no_tool_call_found | length | 30.53 | 512 |
| zh-tc-time-tokyo | tool_call | ✅ | args_subset | stop | 17.17 | 68 |
| zh-tc-time-newyork | tool_call | ✅ | args_subset | stop | 15.38 | 68 |
| zh-tc-calendar-meeting | tool_call | ✅ | args_subset | stop | 23.23 | 166 |
| zh-tc-calendar-deadline | tool_call | ✅ | args_subset | stop | 21.95 | 120 |
| zh-tc-translate-english | tool_call | ❌ | missing:['target_language'] | stop | 18.31 | 72 |
| zh-tc-translate-korean | tool_call | ✅ | args_subset | stop | 18.15 | 84 |
| zh-tc-flight-bj-gz | tool_call | ❌ | no_tool_call_found | stop | 28.41 | 320 |
| zh-tc-flight-sh-gz | tool_call | ✅ | args_subset | stop | 18.32 | 92 |
| zh-tc-search-docs | tool_call | ✅ | args_subset | stop | 14.01 | 60 |
| zh-tc-search-recipe | tool_call | ✅ | args_subset | stop | 16.93 | 67 |
| zh-tc-choice-routing | tool_call | ✅ | args_subset | stop | 20.73 | 122 |
| zh-tc-multiarg-calendar | tool_call | ✅ | args_subset | stop | 19.97 | 110 |
| zh-tc-multiarg-email | tool_call | ✅ | args_subset | stop | 28.03 | 298 |
| zh-tc-distractor-time | tool_call | ✅ | args_subset | stop | 22.31 | 141 |
| zh-tc-distractor-search | tool_call | ✅ | args_subset | stop | 16.65 | 68 |
| zh-tc-flight-date | tool_call | ✅ | args_subset | stop | 22.66 | 142 |
| zh-json-config | json_schema | ✅ | schema_valid | stop | 34.91 | 403 |
| zh-exact-capital | exact_match | ✅ | exact_match | stop | 35.26 | 340 |
