# Run report — agentic_en (mlx)
- **run_id:** `20260726T071154Z`
- **model:** `models/nanbeige-mlx-4bit`
- **env:** python 3.12.12 · mlx 0.32.0 · mlx-lm 0.31.3 · arm64 · git c800313
- **quantization:** 4-bit, group_size=64

## Summary
- pass rate: **27/30** (90.0%) [0.74, 0.97]

| grade kind | pass / n |
|---|---|
| exact_match | 1 / 1 |
| json_schema | 2 / 2 |
| tool_call | 24 / 27 |

## Latency / memory (real-model cases only)
| metric | value |
|---|---|
| decode throughput (aggregate) | 35.11 tok/s |
| decode throughput (median) | 19.73 tok/s |
| TTFT median | 2.304 s |
| TTFT with tools / bare prompt | 2.38 / 0.552 s |
| mean generated tokens | 116.4 |
| peak allocator memory | 2850.9 MB |

## Cases
| id | kind | pass | detail | stop | tok/s | tokens |
|---|---|---|---|---|---|---|
| en-tc-weather-tokyo | tool_call | ✅ | args_subset | stop | 14.25 | 58 |
| en-tc-weather-paris-select | tool_call | ✅ | args_subset | stop | 14.64 | 71 |
| en-tc-email-alice | tool_call | ✅ | args_subset | stop | 20.68 | 163 |
| en-tc-translate-french | tool_call | ✅ | args_subset | stop | 15.5 | 65 |
| en-tc-time-tokyo | tool_call | ✅ | args_subset | stop | 14.91 | 66 |
| en-tc-calendar-date | tool_call | ✅ | args_subset | stop | 22.79 | 152 |
| en-tc-flight-sf | tool_call | ✅ | args_subset | stop | 22.47 | 147 |
| en-json-profile | json_schema | ✅ | schema_valid | stop | 33.66 | 188 |
| en-tc-weather-london | tool_call | ✅ | args_subset | stop | 14.06 | 56 |
| en-tc-weather-singapore | tool_call | ✅ | args_subset | stop | 16.02 | 67 |
| en-tc-email-bob-cc | tool_call | ✅ | args_subset | stop | 20.1 | 121 |
| en-tc-email-carol | tool_call | ❌ | no_tool_call_found | stop | 26.25 | 255 |
| en-tc-time-ny | tool_call | ✅ | args_subset | stop | 16.02 | 71 |
| en-tc-time-london | tool_call | ❌ | missing:['timezone'] | stop | 13.49 | 62 |
| en-tc-calendar-meeting | tool_call | ✅ | args_subset | stop | 19.86 | 152 |
| en-tc-calendar-deadline | tool_call | ✅ | args_subset | stop | 18.41 | 109 |
| en-tc-translate-spanish | tool_call | ✅ | args_subset | stop | 15.17 | 65 |
| en-tc-translate-german | tool_call | ✅ | args_subset | stop | 15.88 | 84 |
| en-tc-flight-london-paris | tool_call | ✅ | args_subset | stop | 19.61 | 132 |
| en-tc-flight-la | tool_call | ✅ | args_subset | stop | 21.0 | 99 |
| en-tc-search-docs | tool_call | ✅ | args_subset | stop | 16.41 | 63 |
| en-tc-search-recipe | tool_call | ❌ | missing:['query'] | stop | 17.29 | 63 |
| en-tc-choice-routing | tool_call | ✅ | args_subset | stop | 24.58 | 171 |
| en-tc-multiarg-calendar | tool_call | ✅ | args_subset | stop | 20.41 | 97 |
| en-tc-multiarg-email | tool_call | ✅ | args_subset | stop | 25.36 | 173 |
| en-tc-distractor-time | tool_call | ✅ | args_subset | stop | 25.78 | 200 |
| en-tc-distractor-search | tool_call | ✅ | args_subset | stop | 16.68 | 70 |
| en-tc-flight-date | tool_call | ✅ | args_subset | stop | 23.38 | 135 |
| en-json-config | json_schema | ✅ | schema_valid | stop | 34.79 | 218 |
| en-exact-capital | exact_match | ✅ | exact_match | stop | 32.7 | 120 |
