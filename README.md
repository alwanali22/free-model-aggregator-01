# Free Model Aggregator

One Node 22 server (stdlib only) that serves OpenAI Chat Completions and
Anthropic Messages over the free models on the brief's provider directory.

Routing: every free model meeting the request's requirements is a candidate,
ranked by the quality table's rating_milli (descending), provider id, then
model id. A provider with two consecutive refusals is demoted to the bottom
until it answers again. JSON-mode requests never go to an anthropic-wire
provider. Failover is sequential, one call per candidate: a 402/429/5xx or a
timeout moves to the next candidate, a 400 marks that model bad for the
trial, and the first 2xx wins and is translated back into the client's wire.

`main.ts` is the entrypoint; `lib/providers.ts` probes and catalogs the
directory; `lib/wire.ts` translates between the OpenAI, Anthropic and
Gemini wires. Deterministic: fixed timestamps and response ids, no clock
or unseeded randomness anywhere.

Run: `node main.ts` with `MAKEFUN_ENV_URL` and `MAKEFUN_PROMPT_FILE` set
(the harness provides both). `node main.ts --list` prints the catalog.
