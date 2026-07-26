# Run report — agentic_en (mlx)
- **run_id:** `20260726T074814Z`
- **model:** `models/nanbeige-mlx-8bit`
- **env:** python 3.12.12 · mlx 0.32.0 · mlx-lm 0.31.3 · arm64 · git c800313
- **quantization:** 8-bit, group_size=64

## Summary
- pass rate: **26/30** (86.7%) [0.70, 0.95]
- ⚠️ **1 case(s) truncated** (hit token cap, no eos): en-tc-flight-london-paris

| grade kind | pass / n |
|---|---|
| exact_match | 1 / 1 |
| json_schema | 2 / 2 |
| tool_call | 23 / 27 |

## Latency / memory (real-model cases only)
| metric | value |
|---|---|
| decode throughput (aggregate) | 20.67 tok/s |
| decode throughput (median) | 13.96 tok/s |
| TTFT median | 2.312 s |
| TTFT with tools / bare prompt | 2.379 / 0.636 s |
| mean generated tokens | 166.3 |
| peak allocator memory | 4741.4 MB |

## Cases
| id | kind | pass | detail | stop | tok/s | tokens |
|---|---|---|---|---|---|---|
| en-tc-weather-tokyo | tool_call | ✅ | args_subset | stop | 10.07 | 60 |
| en-tc-weather-paris-select | tool_call | ✅ | args_subset | stop | 10.86 | 74 |
| en-tc-email-alice | tool_call | ✅ | args_subset | stop | 13.09 | 153 |
| en-tc-translate-french | tool_call | ✅ | args_subset | stop | 9.86 | 65 |
| en-tc-time-tokyo | tool_call | ✅ | args_subset | stop | 9.98 | 66 |
| en-tc-calendar-date | tool_call | ✅ | args_subset | stop | 12.49 | 132 |
| en-tc-flight-sf | tool_call | ✅ | args_subset | stop | 13.0 | 159 |
| en-json-profile | json_schema | ✅ | schema_valid | stop | 19.51 | 962 |
| en-tc-weather-london | tool_call | ✅ | args_subset | stop | 11.8 | 59 |
| en-tc-weather-singapore | tool_call | ✅ | args_subset | stop | 13.63 | 75 |
| en-tc-email-bob-cc | tool_call | ✅ | args_subset | stop | 15.75 | 141 |
| en-tc-email-carol | tool_call | ❌ | no_tool_call_found | stop | 18.88 | 322 |
| en-tc-time-ny | tool_call | ✅ | args_subset | stop | 13.78 | 71 |
| en-tc-time-london | tool_call | ❌ | missing:['timezone'] | stop | 12.23 | 63 |
| en-tc-calendar-meeting | tool_call | ✅ | args_subset | stop | 15.96 | 145 |
| en-tc-calendar-deadline | tool_call | ✅ | args_subset | stop | 15.97 | 122 |
| en-tc-translate-spanish | tool_call | ✅ | args_subset | stop | 13.3 | 65 |
| en-tc-translate-german | tool_call | ✅ | args_subset | stop | 14.13 | 91 |
| en-tc-flight-london-paris | tool_call | ❌ | missing:['destination'] | length | 19.73 | 512 |
| en-tc-flight-la | tool_call | ✅ | args_subset | stop | 15.36 | 99 |
| en-tc-search-docs | tool_call | ✅ | args_subset | stop | 12.65 | 63 |
| en-tc-search-recipe | tool_call | ❌ | missing:['query'] | stop | 13.28 | 64 |
| en-tc-choice-routing | tool_call | ✅ | args_subset | stop | 17.51 | 198 |
| en-tc-multiarg-calendar | tool_call | ✅ | args_subset | stop | 15.51 | 111 |
| en-tc-multiarg-email | tool_call | ✅ | args_subset | stop | 17.32 | 171 |
| en-tc-distractor-time | tool_call | ✅ | args_subset | stop | 18.19 | 244 |
| en-tc-distractor-search | tool_call | ✅ | args_subset | stop | 12.98 | 73 |
| en-tc-flight-date | tool_call | ✅ | args_subset | stop | 14.49 | 139 |
| en-json-config | json_schema | ✅ | schema_valid | stop | 20.81 | 281 |
| en-exact-capital | exact_match | ✅ | exact_match | stop | 20.62 | 208 |
