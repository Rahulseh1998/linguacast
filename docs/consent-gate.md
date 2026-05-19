# Voice-clone consent gate

Tracking: [OPE-12](https://github.com/openpipe-ai/linguacast/issues/12).
This is a launch-blocker. v0 does not ship without it on by default and
without bypass.

LinguaCast clones the speaker's voice into the target language. Treating
that as table stakes for the launch hook would be naive: in the year
preceding the v0 cut, voice-clone abuse showed up in election-cycle
robocalls, non-consensual synthetic media, and crypto scams. The gate
below is what "responsible by default" looks like for a single-binary
local-first tool.

## The gate

Before any TTS render, the speaker (or an authorised consenter) must
attest in writing:

> "I am the speaker in this audio, or I have written consent from the
> speaker."

The attestation is:

- **Keyed on the reference audio** — specifically the SHA-256 of the
  16 kHz mono WAV the TTS step actually consumes. Renaming the source
  video does not bypass the gate; replacing the audio bytes re-prompts.
- **Stored locally** under `~/.linguacast/consents/<sha256>.json` (or
  `$LINGUACAST_HOME/consents/`). Each record carries the audio hash,
  the attestation, the signer (`$USER@$HOSTNAME`), the OS-derived
  machine fingerprint, a UTC timestamp, the binary version, and the
  signing mode.
- **Reused silently** on subsequent runs against the same audio bytes.
  No re-prompt unless the bytes change.

## CLI surface

```
linguacast input.mp4 --langs es
  → interactive prompt for I AGREE on first run; silent re-use after.

linguacast input.mp4 --langs es --i-have-speaker-consent consent.txt
  → required for non-TTY runs (CI, batch). `consent.txt` must contain
    the attestation line verbatim on its own line. Optional `signer:`
    line gets recorded.

linguacast input.mp4 --langs es --speaker-name "Some Person"
  → metadata-only self-declaration; also runs against the refusal list.

linguacast input.mp4 --langs es --refusal-list path/to/list.json
  → override the embedded refusal list (ops can tighten upstream).

linguacast input.mp4 --langs es --consent-store-dir /var/lc/consents
  → override the consent record store (default ~/.linguacast/consents/).
```

If the gate cannot resolve consent it exits non-zero with a helpful
error pointing at the next concrete step.

## Refusal list

A small list of high-profile public figures whose voices have been most
frequently abused in published deepfake reports lives at
`crates/linguacast/data/refusal-list.json` and is embedded into the
binary at build time. The list is checked against:

1. The `--speaker-name` value (case-insensitive, against `name` +
   `aliases`).
2. The reference-audio SHA-256 (against the entry's `audio_sha256`
   list).

A match aborts the run regardless of consent attestation. The error
message points at this document. The list is deliberately short and
biased toward names that are press-confirmable; we'd rather under-block
than expose ourselves to subjective "celebrity" judgement calls.

**Update path.** Open a PR against
`crates/linguacast/data/refusal-list.json` with:

- The full name and a non-empty `aliases` array of variants.
- A `reason` that points at a published source (a news article, a
  research paper, or the subject's own public complaint).
- Optionally, `audio_sha256` entries for known abusive samples.

CTO review is required for additions — the bar is "would a press reader
find this entry obviously well-grounded?" not "could this voice be
abused?"

## Output provenance

Every output MP4 carries the consent metadata in its container
(`udta` atoms), preserved with `-movflags +use_metadata_tags`:

| Atom | Value |
| --- | --- |
| `comment` | `linguacast:consent_hash=<sha>;consent_ts=<iso>;signer=<id>;lang=<target>;version=<v>` |
| `linguacast_consent_hash` | hex SHA-256 of the reference audio |
| `linguacast_consent_timestamp` | UTC ISO-8601 |
| `linguacast_consent_signer` | `$USER@$HOSTNAME` from the signing run |
| `linguacast_version` | binary version |

Verify with `ffprobe -show_format <output.mp4>` — the tags appear in the
`TAG:` section of the output.

## Consent file format (non-TTY mode)

The file passed via `--i-have-speaker-consent` is plain UTF-8 text. It
must contain the attestation line verbatim on its own line. An optional
`signer:` line records the signing identity; we hash the file body and
store both the path and the body's SHA-256 in the consent record.

```
I am the speaker in this audio, or I have written consent from the speaker.
signer: Jane Speaker
```

Comments / extra lines are fine. The file is read at gate time only —
we do not require cryptographic signing for v0 (we are not a notary).
The audit value comes from (1) the per-host record, (2) the hash chain
into the output MP4, and (3) the body SHA-256 stored in the record.

## What this gate does *not* do

- **It does not verify the audio actually contains the named speaker.**
  Voice-bio matching is a v0.2 enhancement, not a launch dependency.
- **It does not protect against a determined adversary on the same
  machine.** Local store + plaintext attestation. The threat model is
  "make casual misuse explicit," not "stop a motivated forger."
- **It does not handle real-time / streaming consent.** v0 is batch
  only; the streaming consent flow lives with the v0.2 streaming work.

## Implementation pointers

- `crates/linguacast/src/consent.rs` — module entrypoint, schema,
  refusal list, store I/O, interactive + file flows.
- `crates/linguacast/src/cli.rs` — flags (`--i-have-speaker-consent`,
  `--speaker-name`, `--refusal-list`, `--consent-store-dir`).
- `crates/linguacast/src/pipeline.rs` — `run_consent_gate` is called
  once per dub, before any TTS render, with the extracted 16k mono WAV
  as the hash target.
- `crates/linguacast/src/ffmpeg.rs` — `mux_replace_audio` accepts the
  metadata k/v pairs and writes them with `+use_metadata_tags`.
- `crates/linguacast/data/refusal-list.json` — the embedded refusal
  list.
