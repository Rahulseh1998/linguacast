# MODELS

Every model the LinguaCast Python sidecar can download at runtime is listed
here, with its license and the canonical upstream URL. This file is the
**source of truth** for model licensing — `scripts/check_model_licenses.py`
parses the table below in CI and fails the build if any entry has a license
outside the permissive allowlist.

`check_model_licenses.py` was originally written by the OPE-4 track
(workspaces/7ab06d1c-e658-4d1d-bcd7-e86497acb404) and is re-used here
per the intra-company lift approved in the OPE-19 CTO ack (2026-05-19).

## Allowlist

Same allowlist as `cargo-deny.toml` (Rust deps) and the CI `pip-licenses`
audit (Python deps), so the same floor applies end-to-end:

`Apache-2.0`, `MIT`, `MIT-0`, `BSD-2-Clause`, `BSD-3-Clause`, `ISC`,
`Unlicense`, `Unicode-3.0`, `CC0-1.0`, `MPL-2.0` (case-by-case CTO sign-off,
per [OPE-17](../OPE/issues/OPE-17)).

Anything else — GPL/LGPL/AGPL, CC-BY-NC, research-only, custom
"non-commercial", "OpenRAIL", undeclared — **fails CI**.

## Models

The table is machine-parsed. Do not reflow columns, rename headers, or
introduce nested tables. Add rows in alphabetical order by `Model`.

| Role | Model | HF Repo / Source | License | License URL |
|---|---|---|---|---|
| ASR | Whisper large-v3 (CTranslate2) | Systran/faster-whisper-large-v3 | MIT | https://github.com/SYSTRAN/faster-whisper/blob/master/LICENSE |
| ASR | Whisper large-v3 (original) | openai/whisper-large-v3 | MIT | https://github.com/openai/whisper/blob/main/LICENSE |
| ASR | Whisper medium (CTranslate2 fallback) | Systran/faster-whisper-medium | MIT | https://github.com/SYSTRAN/faster-whisper/blob/master/LICENSE |
| MT | M2M-100 418M (default) | facebook/m2m100_418M | MIT | https://huggingface.co/facebook/m2m100_418M/blob/main/README.md |
| MT | MADLAD-400 3B-MT (opt-in, ≥16 GB) | google/madlad400-3b-mt | Apache-2.0 | https://huggingface.co/google/madlad400-3b-mt/blob/main/LICENSE |
| TTS | Qwen3-TTS 12Hz 0.6B Base | Qwen/Qwen3-TTS-12Hz-0.6B-Base | Apache-2.0 | https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base/blob/main/LICENSE |
| TTS | Qwen3-TTS 12Hz 1.7B Base | Qwen/Qwen3-TTS-12Hz-1.7B-Base | Apache-2.0 | https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base/blob/main/LICENSE |

## Adding a new model

1. Confirm the upstream license file. If it isn't on the allowlist, **stop
   and escalate to the CTO** — do not add the model and silently swap to a
   "close enough" alternative.
2. Add a row to the table above, keeping it alphabetised by `Model`.
3. Update the `NOTICE` file and the README "License" section if the model
   is wired into a default code path.
4. Run `python scripts/check_model_licenses.py` locally before pushing.
