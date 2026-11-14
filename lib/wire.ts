// The client wires and the upstream wires this entry speaks.
//
// The trial's client calls the entry as an OpenAI service (POST /v1/chat/completions) or an Anthropic one
// (POST /v1/messages), with a JSON body or with server-sent events. This entry forwards both to a
// provider on any of the three wires (openai, anthropic, gemini) and translates back:
//
//   - OpenAI client -> openai provider: body passed through (model replaced), JSON or SSE straight back.
//   - Anthropic client -> openai provider: request rewritten to Chat Completions, answer rewritten back.
//   - Either client -> anthropic provider: request rewritten to Messages, answer rewritten back.
//   - Either client -> gemini provider: request rewritten to generateContent, answer rewritten back.
//
// Everything here is a pure function of its arguments: no clock and no randomness, so the same body and
// the same provider answer give the same bytes on a replay. The two model listings use fixed timestamps
// for the same reason: a `created` that moved between runs would be a nondeterministic document.
//
// The translations preserve EXACTLY what the provider echoes back and the scorer compares: system text,
// last user message text, tool names in order, max_tokens, stop sequences, image bytes, json_mode.
// A field rewritten or dropped loses the request.

import { isRecord, type Json } from './providers.ts';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A fixed epoch second for every listed model's `created`: nothing in the reply reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing in the reply reads the clock. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves, in OpenAI's list shape. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** One Anthropic content block as an OpenAI content part: text or an inline image; null for the rest. */
function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** A tool call's arguments as an object: the provider's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** OpenAI's finish_reason as Anthropic's stop_reason. */
export function stopReasonOf(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request that names `model`: the system
 * prompt becomes a system message, image blocks become image_url data URLs, tool_use blocks become
 * tool_calls, tool_result blocks become tool messages, tools and tool_choice are rewritten, and the
 * output cap, temperature, top_p, stop sequences and streaming pass through.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = partOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** A Chat Completions answer as an Anthropic Messages object. */
export function openAIToAnthropic(body: Json, model: string): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  if (text !== '') content.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: argumentsOf(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    id: `msg_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: stopReasonOf(choice.finish_reason),
    stop_sequence: null,
    usage: { input_tokens: numberOrZero(usage.prompt_tokens), output_tokens: numberOrZero(usage.completion_tokens) },
  };
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/**
 * A Chat Completions SSE stream as the Anthropic event sequence: message_start on the first chunk,
 * content_block_start/delta/stop per text block or tool call, then message_delta with the stop reason and
 * usage, then message_stop. `model` is the model the client asked for, named back to it.
 */
export async function* anthropicEventsFromOpenAI(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let started = false;
  let finished = false;
  /** The stop reason a finish_reason named, held until the stream's usage chunk has arrived. */
  let pendingStop: string | null = null;
  let textIndex = -1;
  let nextIndex = 0;
  const toolIndexes = new Map<number, number>();
  let inputTokens = 0;
  let outputTokens = 0;

  const startEvents = (): string[] => {
    if (started) return [];
    started = true;
    return [
      sseEvent('message_start', {
        type: 'message_start',
        message: {
          id: 'msg_stream',
          type: 'message',
          role: 'assistant',
          model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: inputTokens, output_tokens: 0 },
        },
      }),
    ];
  };
  const closeEvents = (): string[] => {
    const out: string[] = [];
    if (textIndex !== -1) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
      textIndex = -1;
    }
    for (const index of [...toolIndexes.values()].sort((a, b) => a - b)) out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    toolIndexes.clear();
    return out;
  };
  const finishEvents = (stopReason: string): string[] => {
    if (finished) return [];
    finished = true;
    return [
      ...closeEvents(),
      sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stopReason, stop_sequence: null }, usage: { output_tokens: outputTokens } }),
      sseEvent('message_stop', { type: 'message_stop' }),
    ];
  };

  /** One SSE frame's lines as Anthropic events, mutating the stream's state. */
  const eventsOfFrame = (frame: string): string[] => {
    const out: string[] = [];
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '[DONE]') {
        out.push(...startEvents(), ...finishEvents(pendingStop ?? 'end_turn'));
        continue;
      }
      let chunk: unknown;
      try {
        chunk = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(chunk)) continue;
      if (isRecord(chunk.usage)) {
        inputTokens = numberOrZero(chunk.usage.prompt_tokens);
        outputTokens = numberOrZero(chunk.usage.completion_tokens);
      }
      const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
      const choice = isRecord(choices[0]) ? choices[0] : null;
      if (choice === null) continue;
      const delta = isRecord(choice.delta) ? choice.delta : {};
      out.push(...startEvents());
      if (typeof delta.content === 'string' && delta.content !== '') {
        if (textIndex === -1) {
          textIndex = nextIndex;
          nextIndex += 1;
          out.push(sseEvent('content_block_start', { type: 'content_block_start', index: textIndex, content_block: { type: 'text', text: '' } }));
        }
        out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: textIndex, delta: { type: 'text_delta', text: delta.content } }));
      }
      const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call)) continue;
        const upstreamIndex = typeof call.index === 'number' ? call.index : toolIndexes.size;
        let index = toolIndexes.get(upstreamIndex);
        if (index === undefined) {
          if (textIndex !== -1) {
            out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
            textIndex = -1;
          }
          index = nextIndex;
          nextIndex += 1;
          toolIndexes.set(upstreamIndex, index);
          const fn = isRecord(call.function) ? call.function : {};
          out.push(
            sseEvent('content_block_start', {
              type: 'content_block_start',
              index,
              content_block: {
                type: 'tool_use',
                id: typeof call.id === 'string' && call.id !== '' ? call.id : `toolu_${index}`,
                name: typeof fn.name === 'string' ? fn.name : '',
                input: {},
              },
            }),
          );
        }
        const fn = isRecord(call.function) ? call.function : {};
        if (typeof fn.arguments === 'string' && fn.arguments !== '') {
          out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: fn.arguments } }));
        }
      }
      // Hold the stop reason until the stream's last chunk (the usage one) and [DONE] have arrived: OpenAI
      // sends the usage after the finish_reason, and Anthropic's message_delta carries the output tokens.
      if (choice.finish_reason !== null && choice.finish_reason !== undefined) pendingStop = stopReasonOf(choice.finish_reason);
    }
    return out;
  };

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const event of eventsOfFrame(frame)) yield event;
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') for (const event of eventsOfFrame(buffer)) yield event;
  } finally {
    reader.releaseLock();
  }
  if (!finished) {
    for (const event of startEvents()) yield event;
    for (const event of finishEvents(pendingStop ?? 'end_turn')) yield event;
  }
}

// ---------------------------------------------------------------------------
// Gemini upstream wire
// ---------------------------------------------------------------------------
// How the simulated Gemini provider reads a request: systemInstruction.parts
// text joined "\n"; user text = the LAST content whose role is not "model";
// role "model" contents are skipped for the user text, so tool results and
// assistant messages MUST go in role "model" contents. Images: inlineData
// (base64) or fileData (http URL). Tools: functionDeclarations names.
// generationConfig: maxOutputTokens, stopSequences, responseMimeType.

export function openAIToGemini(body: Json, model: string): Json {
  void model;

  const contents: Json[] = [];
  const system: string[] = [];

  function contentParts(content: unknown): Json[] {
    if (typeof content === 'string') return [{ text: content }];
    if (!Array.isArray(content)) return [];

    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'text' && typeof block.text === 'string') {
        parts.push({ text: block.text });
      } else if (block.type === 'image_url') {
        const url =
          typeof block.image_url === 'string'
            ? block.image_url
            : isRecord(block.image_url)
              ? block.image_url.url
              : undefined;
        if (typeof url !== 'string') continue;

        const data = /^data:([^;,]+);base64,([\s\S]*)$/.exec(url);
        if (data) {
          parts.push({
            inlineData: { mimeType: data[1], data: data[2] },
          });
        } else if (/^https?:\/\//i.test(url)) {
          parts.push({ fileData: { fileUri: url } });
        }
      }
    }
    return parts;
  }

  if (Array.isArray(body.messages)) {
    for (const message of body.messages) {
      if (!isRecord(message)) continue;

      if (message.role === 'system') {
        system.push(textOf(message.content));
        continue;
      }

      if (message.role === 'tool') {
        contents.push({
          role: 'model',
          parts: [{ text: toolResultText(message.content) }],
        });
        continue;
      }

      if (message.role !== 'user' && message.role !== 'assistant') continue;

      const parts = contentParts(message.content);
      if (message.role === 'assistant' && Array.isArray(message.tool_calls)) {
        for (const call of message.tool_calls) {
          if (!isRecord(call) || !isRecord(call.function)) continue;
          parts.push({
            functionCall: {
              name: call.function.name,
              args: argumentsOf(call.function.arguments),
            },
          });
        }
      }

      if (parts.length > 0) {
        contents.push({
          role: message.role === 'assistant' ? 'model' : 'user',
          parts,
        });
      }
    }
  }

  const result: Json = { contents };
  const systemText = system.join('\n');
  if (systemText !== '') {
    result.systemInstruction = { parts: [{ text: systemText }] };
  }

  if (Array.isArray(body.tools) && body.tools.length > 0) {
    const functionDeclarations: Json[] = [];
    for (const tool of body.tools) {
      if (!isRecord(tool) || !isRecord(tool.function)) continue;
      const fn = tool.function;
      functionDeclarations.push({
        name: fn.name,
        description: typeof fn.description === 'string' ? fn.description : '',
        parameters: isRecord(fn.parameters)
          ? fn.parameters
          : { type: 'object', properties: {} },
      });
    }
    result.tools = [{ functionDeclarations }];
  }

  const generationConfig: Json = {};
  const maxTokens =
    typeof body.max_tokens === 'number'
      ? body.max_tokens
      : body.max_completion_tokens;
  if (typeof maxTokens === 'number') {
    generationConfig.maxOutputTokens = maxTokens;
  }
  if (typeof body.stop === 'string') {
    generationConfig.stopSequences = [body.stop];
  } else if (Array.isArray(body.stop)) {
    generationConfig.stopSequences = body.stop.filter(
      (stop): stop is string => typeof stop === 'string',
    );
  }
  if (
    isRecord(body.response_format) &&
    (body.response_format.type === 'json_object' ||
      body.response_format.type === 'json_schema')
  ) {
    generationConfig.responseMimeType = 'application/json';
  }
  if (typeof body.temperature === 'number') {
    generationConfig.temperature = body.temperature;
  }
  if (typeof body.top_p === 'number') {
    generationConfig.topP = body.top_p;
  }
  result.generationConfig = generationConfig;

  return result;
}

export function anthropicToGemini(body: Json, model: string): Json {
  void model;

  const contents: Json[] = [];
  const systemText = textOf(body.system);

  if (Array.isArray(body.messages)) {
    for (const message of body.messages) {
      if (!isRecord(message)) continue;
      if (message.role !== 'user' && message.role !== 'assistant') continue;

      const role = message.role === 'assistant' ? 'model' : 'user';
      let parts: Json[] = [];

      function flush(): void {
        if (parts.length > 0) {
          contents.push({ role, parts });
          parts = [];
        }
      }

      if (typeof message.content === 'string') {
        parts.push({ text: message.content });
      } else if (Array.isArray(message.content)) {
        for (const block of message.content) {
          if (!isRecord(block)) continue;

          if (block.type === 'tool_result') {
            flush();
            contents.push({
              role: 'model',
              parts: [{ text: toolResultText(block.content) }],
            });
          } else if (block.type === 'text') {
            const part = partOf(block);
            if (part && typeof part.text === 'string') {
              parts.push({ text: part.text });
            }
          } else if (block.type === 'image' && isRecord(block.source)) {
            const source = block.source;
            if (source.type === 'base64' && typeof source.data === 'string') {
              parts.push({
                inlineData: {
                  mimeType:
                    typeof source.media_type === 'string'
                      ? source.media_type
                      : 'image/png',
                  data: source.data,
                },
              });
            } else if (
              source.type === 'url' &&
              typeof source.url === 'string'
            ) {
              parts.push({ fileData: { fileUri: source.url } });
            }
          } else if (block.type === 'tool_use' && role === 'model') {
            parts.push({
              functionCall: {
                name: block.name,
                args: isRecord(block.input) ? block.input : {},
              },
            });
          }
        }
      }

      flush();
    }
  }

  const result: Json = { contents };
  if (systemText !== '') {
    result.systemInstruction = { parts: [{ text: systemText }] };
  }

  if (Array.isArray(body.tools) && body.tools.length > 0) {
    const functionDeclarations: Json[] = [];
    for (const tool of body.tools) {
      if (!isRecord(tool)) continue;
      functionDeclarations.push({
        name: tool.name,
        description:
          typeof tool.description === 'string' ? tool.description : '',
        parameters: isRecord(tool.input_schema)
          ? tool.input_schema
          : { type: 'object', properties: {} },
      });
    }
    result.tools = [{ functionDeclarations }];
  }

  const generationConfig: Json = {};
  if (typeof body.max_tokens === 'number') {
    generationConfig.maxOutputTokens = body.max_tokens;
  }
  if (Array.isArray(body.stop_sequences)) {
    generationConfig.stopSequences = body.stop_sequences.filter(
      (stop): stop is string => typeof stop === 'string',
    );
  }
  if (typeof body.temperature === 'number') {
    generationConfig.temperature = body.temperature;
  }
  if (typeof body.top_p === 'number') {
    generationConfig.topP = body.top_p;
  }
  result.generationConfig = generationConfig;

  return result;
}

export function geminiToOpenAI(body: Json, model: string): Json {
  const candidate =
    Array.isArray(body.candidates) && isRecord(body.candidates[0])
      ? body.candidates[0]
      : {};
  const parts =
    isRecord(candidate.content) && Array.isArray(candidate.content.parts)
      ? candidate.content.parts
      : [];
  const content = parts
    .filter(
      (part): part is Json => isRecord(part) && typeof part.text === 'string',
    )
    .map((part) => part.text)
    .join('');

  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  const promptTokens =
    typeof usage.promptTokenCount === 'number' ? usage.promptTokenCount : 0;
  const completionTokens =
    typeof usage.candidatesTokenCount === 'number'
      ? usage.candidatesTokenCount
      : 0;
  const finishReason =
    candidate.finishReason === 'MAX_TOKENS'
      ? 'length'
      : candidate.finishReason === 'SAFETY'
        ? 'content_filter'
        : 'stop';

  return {
    id: 'chatcmpl-sim',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [
      {
        index: 0,
        message: { role: 'assistant', content },
        finish_reason: finishReason,
      },
    ],
    usage: {
      prompt_tokens: promptTokens,
      completion_tokens: completionTokens,
      total_tokens: promptTokens + completionTokens,
    },
  };
}

export function geminiToAnthropic(body: Json, model: string): Json {
  const candidate =
    Array.isArray(body.candidates) && isRecord(body.candidates[0])
      ? body.candidates[0]
      : {};
  const parts =
    isRecord(candidate.content) && Array.isArray(candidate.content.parts)
      ? candidate.content.parts
      : [];
  const text = parts
    .filter(
      (part): part is Json => isRecord(part) && typeof part.text === 'string',
    )
    .map((part) => part.text)
    .join('');

  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  const stopReason =
    candidate.finishReason === 'MAX_TOKENS'
      ? 'max_tokens'
      : candidate.finishReason === 'SAFETY'
        ? 'refusal'
        : 'end_turn';

  return {
    id: 'msg_sim',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text }],
    stop_reason: stopReason,
    stop_sequence: null,
    usage: {
      input_tokens:
        typeof usage.promptTokenCount === 'number' ? usage.promptTokenCount : 0,
      output_tokens:
        typeof usage.candidatesTokenCount === 'number'
          ? usage.candidatesTokenCount
          : 0,
    },
  };
}

/** One Gemini SSE frame's JSON: split a raw stream chunk on blank lines, parse each `data:` line. */
async function* geminiFrames(upstream: ReadableStream<Uint8Array>): AsyncGenerator<Json> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  function parse(frame: string): Json | null {
    const data = frame
      .split(/\r\n|\r|\n/)
      .filter((line) => line.startsWith('data:'))
      .map((line) => {
        const value = line.slice(5);
        return value.startsWith(' ') ? value.slice(1) : value;
      })
      .join('\n');
    if (data === '' || data === '[DONE]') return null;
    try {
      const value: unknown = JSON.parse(data);
      return isRecord(value) ? value : null;
    } catch {
      return null;
    }
  }

  try {
    for (;;) {
      const { value, done } = await reader.read();
      buffer += done
        ? decoder.decode()
        : decoder.decode(value, { stream: true });

      let boundary: RegExpExecArray | null;
      while ((boundary = /\r\n\r\n|\n\n|\r\r/.exec(buffer)) !== null) {
        const frame = parse(buffer.slice(0, boundary.index));
        buffer = buffer.slice(boundary.index + boundary[0].length);
        if (frame) yield frame;
      }

      if (done) {
        const frame = parse(buffer);
        if (frame) yield frame;
        break;
      }
    }
  } finally {
    reader.releaseLock();
  }
}

/** A Gemini finishReason as an OpenAI finish_reason. */
function geminiFinishToOpenAI(finishReason: unknown): string {
  if (finishReason === 'MAX_TOKENS') return 'length';
  if (finishReason === 'SAFETY') return 'content_filter';
  return 'stop';
}

/** A Gemini finishReason as an Anthropic stop_reason. */
function geminiFinishToAnthropic(finishReason: unknown): string {
  if (finishReason === 'MAX_TOKENS') return 'max_tokens';
  if (finishReason === 'SAFETY') return 'refusal';
  return 'end_turn';
}

/** The text parts of one Gemini candidate, in order. */
function geminiCandidateText(candidate: Json): string {
  const parts =
    isRecord(candidate.content) && Array.isArray(candidate.content.parts)
      ? candidate.content.parts
      : [];
  return parts
    .filter(
      (part): part is Json => isRecord(part) && typeof part.text === 'string',
    )
    .map((part) => part.text)
    .join('');
}

export async function* openAIEventsFromGemini(
  upstream: ReadableStream<Uint8Array>,
  model: string,
): AsyncGenerator<string> {
  const base = {
    id: 'chatcmpl-sim',
    object: 'chat.completion.chunk',
    created: FIXED_OPENAI_CREATED,
    model,
  };
  let roleSent = false;
  let finishReason = 'stop';
  let usageMetadata: Json | null = null;

  for await (const frame of geminiFrames(upstream)) {
    if (isRecord(frame.usageMetadata)) {
      usageMetadata = frame.usageMetadata;
    }

    const candidate =
      Array.isArray(frame.candidates) && isRecord(frame.candidates[0])
        ? frame.candidates[0]
        : {};
    if (typeof candidate.finishReason === 'string') {
      finishReason = geminiFinishToOpenAI(candidate.finishReason);
    }

    const text = geminiCandidateText(candidate);
    if (text !== '') {
      if (!roleSent) {
        roleSent = true;
        yield sseEvent(null, {
          ...base,
          choices: [
            {
              index: 0,
              delta: { role: 'assistant', content: '' },
              finish_reason: null,
            },
          ],
        });
      }
      yield sseEvent(null, {
        ...base,
        choices: [
          { index: 0, delta: { content: text }, finish_reason: null },
        ],
      });
    }
  }

  if (!roleSent) {
    yield sseEvent(null, {
      ...base,
      choices: [
        {
          index: 0,
          delta: { role: 'assistant', content: '' },
          finish_reason: null,
        },
      ],
    });
  }

  yield sseEvent(null, {
    ...base,
    choices: [{ index: 0, delta: {}, finish_reason: finishReason }],
  });

  if (usageMetadata !== null) {
    const promptTokens =
      typeof usageMetadata.promptTokenCount === 'number'
        ? usageMetadata.promptTokenCount
        : 0;
    const completionTokens =
      typeof usageMetadata.candidatesTokenCount === 'number'
        ? usageMetadata.candidatesTokenCount
        : 0;
    yield sseEvent(null, {
      ...base,
      choices: [],
      usage: {
        prompt_tokens: promptTokens,
        completion_tokens: completionTokens,
        total_tokens: promptTokens + completionTokens,
      },
    });
  }

  yield 'data: [DONE]\n\n';
}

export async function* anthropicEventsFromGemini(
  upstream: ReadableStream<Uint8Array>,
  model: string,
): AsyncGenerator<string> {
  yield sseEvent('message_start', {
    type: 'message_start',
    message: {
      id: 'msg_sim',
      type: 'message',
      role: 'assistant',
      model,
      content: [],
      stop_reason: null,
      stop_sequence: null,
      usage: { input_tokens: 0, output_tokens: 0 },
    },
  });

  let blockStarted = false;
  let stopReason = 'end_turn';
  let outputTokens = 0;

  for await (const frame of geminiFrames(upstream)) {
    if (isRecord(frame.usageMetadata)) {
      outputTokens =
        typeof frame.usageMetadata.candidatesTokenCount === 'number'
          ? frame.usageMetadata.candidatesTokenCount
          : 0;
    }

    const candidate =
      Array.isArray(frame.candidates) && isRecord(frame.candidates[0])
        ? frame.candidates[0]
        : {};
    if (typeof candidate.finishReason === 'string') {
      stopReason = geminiFinishToAnthropic(candidate.finishReason);
    }

    const text = geminiCandidateText(candidate);
    if (text !== '') {
      if (!blockStarted) {
        blockStarted = true;
        yield sseEvent('content_block_start', {
          type: 'content_block_start',
          index: 0,
          content_block: { type: 'text', text: '' },
        });
      }
      yield sseEvent('content_block_delta', {
        type: 'content_block_delta',
        index: 0,
        delta: { type: 'text_delta', text },
      });
    }
  }

  if (!blockStarted) {
    yield sseEvent('content_block_start', {
      type: 'content_block_start',
      index: 0,
      content_block: { type: 'text', text: '' },
    });
  }

  yield sseEvent('content_block_stop', {
    type: 'content_block_stop',
    index: 0,
  });
  yield sseEvent('message_delta', {
    type: 'message_delta',
    delta: { stop_reason: stopReason, stop_sequence: null },
    usage: { output_tokens: outputTokens },
  });
  yield sseEvent('message_stop', { type: 'message_stop' });
}

// ---------------------------------------------------------------------------
// Anthropic upstream wire, serving an OpenAI client
// ---------------------------------------------------------------------------

/** An Anthropic Messages answer as a Chat Completions answer. */
export function anthropicToOpenAIResponse(body: Json, model: string): Json {
  const texts: string[] = [];
  const toolCalls: Json[] = [];

  if (Array.isArray(body.content)) {
    for (const block of body.content) {
      if (!isRecord(block)) continue;

      if (block.type === 'text' && typeof block.text === 'string') {
        texts.push(block.text);
      } else if (block.type === 'tool_use') {
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : 'toolu_unknown',
          type: 'function',
          function: {
            name: block.name,
            arguments: JSON.stringify(
              isRecord(block.input) ? argumentsOf(block.input) : {},
            ),
          },
        });
      }
    }
  }

  const finishReason =
    body.stop_reason === 'max_tokens'
      ? 'length'
      : body.stop_reason === 'tool_use'
        ? 'tool_calls'
        : body.stop_reason === 'refusal'
          ? 'content_filter'
          : 'stop';
  const usage = isRecord(body.usage) ? body.usage : {};
  const inputTokens =
    typeof usage.input_tokens === 'number' ? usage.input_tokens : 0;
  const outputTokens =
    typeof usage.output_tokens === 'number' ? usage.output_tokens : 0;

  return {
    id: 'chatcmpl-sim',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [
      {
        index: 0,
        message: {
          role: 'assistant',
          content: texts.join(''),
          ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
        },
        finish_reason: finishReason,
      },
    ],
    usage: {
      prompt_tokens: inputTokens,
      completion_tokens: outputTokens,
      total_tokens: inputTokens + outputTokens,
    },
  };
}

/**
 * An Anthropic SSE stream as the OpenAI event sequence: role chunk, content
 * chunks, tool-call chunks, ONE finish chunk, usage chunk, then [DONE].
 */
export async function* openAIEventsFromAnthropic(
  upstream: ReadableStream<Uint8Array>,
  model: string,
): AsyncGenerator<string> {
  async function* frames(): AsyncGenerator<{ event: string; data: Json }> {
    const reader = upstream.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    function parse(frame: string): { event: string; data: Json } | null {
      let event = '';
      const dataLines: string[] = [];

      for (const line of frame.split(/\r\n|\n|\r/)) {
        if (line.startsWith(':')) continue;
        const colon = line.indexOf(':');
        const field = colon < 0 ? line : line.slice(0, colon);
        let value = colon < 0 ? '' : line.slice(colon + 1);
        if (value.startsWith(' ')) value = value.slice(1);

        if (field === 'event') event = value;
        if (field === 'data') dataLines.push(value);
      }

      if (event === 'ping' || dataLines.length === 0) return null;

      try {
        const data: unknown = JSON.parse(dataLines.join('\n'));
        return isRecord(data) ? { event, data } : null;
      } catch {
        return null;
      }
    }

    try {
      for (;;) {
        const { value, done } = await reader.read();
        buffer += done
          ? decoder.decode()
          : decoder.decode(value, { stream: true });

        let boundary: RegExpExecArray | null;
        while ((boundary = /\r\n\r\n|\n\n|\r\r/.exec(buffer)) !== null) {
          const frame = parse(buffer.slice(0, boundary.index));
          buffer = buffer.slice(boundary.index + boundary[0].length);
          if (frame !== null) yield frame;
        }

        if (done) {
          const frame = parse(buffer);
          if (frame !== null) yield frame;
          break;
        }
      }
    } finally {
      reader.releaseLock();
    }
  }

  const base = {
    id: 'chatcmpl-sim',
    object: 'chat.completion.chunk',
    created: FIXED_OPENAI_CREATED,
    model,
  };
  const roleChunk = {
    ...base,
    choices: [
      {
        index: 0,
        delta: { role: 'assistant', content: '' },
        finish_reason: null,
      },
    ],
  };
  const blocks = new Map<
    number,
    { type: 'text' } | { type: 'tool_use'; id: string; name: string; sent: boolean }
  >();

  let roleSent = false;
  let inputTokens = 0;
  let outputTokens = 0;
  let finishReason = 'stop';

  for await (const { event, data } of frames()) {
    if (event === 'message_start') {
      const message = isRecord(data.message) ? data.message : {};
      const usage = isRecord(message.usage) ? message.usage : {};
      inputTokens =
        typeof usage.input_tokens === 'number' ? usage.input_tokens : 0;
    } else if (event === 'content_block_start') {
      if (typeof data.index !== 'number' || !isRecord(data.content_block)) {
        continue;
      }

      const block = data.content_block;
      if (block.type === 'text') {
        blocks.set(data.index, { type: 'text' });
      } else if (block.type === 'tool_use') {
        blocks.set(data.index, {
          type: 'tool_use',
          id: typeof block.id === 'string' ? block.id : `toolu_${data.index}`,
          name: typeof block.name === 'string' ? block.name : '',
          sent: false,
        });
      }
    } else if (event === 'content_block_delta') {
      if (!isRecord(data.delta)) continue;
      const delta = data.delta;

      if (delta.type === 'text_delta' && typeof delta.text === 'string') {
        if (!roleSent) {
          roleSent = true;
          yield sseEvent(null, roleChunk);
        }

        yield sseEvent(null, {
          ...base,
          choices: [
            { index: 0, delta: { content: delta.text }, finish_reason: null },
          ],
        });
      } else if (
        delta.type === 'input_json_delta' &&
        typeof delta.partial_json === 'string' &&
        typeof data.index === 'number'
      ) {
        const tool = blocks.get(data.index);
        if (!tool || tool.type !== 'tool_use') continue;

        if (!roleSent) {
          roleSent = true;
          yield sseEvent(null, roleChunk);
        }

        const toolCall = {
          index: data.index,
          ...(!tool.sent ? { id: tool.id } : {}),
          function: {
            ...(!tool.sent ? { name: tool.name } : {}),
            arguments: delta.partial_json,
          },
        };
        tool.sent = true;

        yield sseEvent(null, {
          ...base,
          choices: [
            {
              index: 0,
              delta: { tool_calls: [toolCall] },
              finish_reason: null,
            },
          ],
        });
      }
    } else if (event === 'message_delta') {
      const delta = isRecord(data.delta) ? data.delta : {};
      finishReason =
        delta.stop_reason === 'max_tokens'
          ? 'length'
          : delta.stop_reason === 'tool_use'
            ? 'tool_calls'
            : delta.stop_reason === 'refusal'
              ? 'content_filter'
              : 'stop';

      const usage = isRecord(data.usage) ? data.usage : {};
      outputTokens =
        typeof usage.output_tokens === 'number' ? usage.output_tokens : 0;
    } else if (event === 'message_stop') {
      break;
    }
  }

  if (!roleSent) yield sseEvent(null, roleChunk);

  yield sseEvent(null, {
    ...base,
    choices: [{ index: 0, delta: {}, finish_reason: finishReason }],
  });
  yield sseEvent(null, {
    ...base,
    choices: [],
    usage: {
      prompt_tokens: inputTokens,
      completion_tokens: outputTokens,
      total_tokens: inputTokens + outputTokens,
    },
  });
  yield 'data: [DONE]\n\n';
}

// ---------------------------------------------------------------------------
// OpenAI client request as an Anthropic Messages request (for anthropic-wire
// providers). Mirror of anthropicToOpenAI above.
// ---------------------------------------------------------------------------

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request that
 * names `model`: system messages become the system string, image_url parts
 * become image blocks, tool messages become user messages with tool_result
 * blocks, assistant tool_calls become tool_use blocks, tools and tool_choice
 * are rewritten, and the output cap, temperature, top_p, stop sequences and
 * streaming pass through.
 */
export function openAIToAnthropicRequest(body: Json, model: string): Json {
  const out: Json = { model };
  const systemParts: string[] = [];
  const messages: Json[] = [];

  if (Array.isArray(body.messages)) {
    for (const message of body.messages) {
      if (!isRecord(message)) continue;
      if (message.role === 'system') {
        const text = textOf(message.content);
        if (text !== '') systemParts.push(text);
        continue;
      }
      if (message.role === 'tool') {
        messages.push({
          role: 'user',
          content: [
            {
              type: 'tool_result',
              tool_use_id: typeof message.tool_call_id === 'string' ? message.tool_call_id : '',
              content: typeof message.content === 'string' ? message.content : toolResultText(message.content),
            },
          ],
        });
        continue;
      }
      if (message.role === 'user') {
        if (typeof message.content === 'string') {
          messages.push({ role: 'user', content: message.content });
          continue;
        }
        if (!Array.isArray(message.content)) continue;
        const blocks: Json[] = [];
        for (const part of message.content) {
          if (!isRecord(part)) continue;
          if (part.type === 'text' && typeof part.text === 'string') {
            blocks.push({ type: 'text', text: part.text });
          } else if (part.type === 'image_url' && isRecord(part.image_url)) {
            const url = part.image_url.url;
            if (typeof url !== 'string') continue;
            const data = /^data:([^;,]+);base64,([\s\S]*)$/.exec(url);
            if (data) {
              blocks.push({ type: 'image', source: { type: 'base64', media_type: data[1], data: data[2] } });
            } else if (/^https?:\/\//i.test(url)) {
              blocks.push({ type: 'image', source: { type: 'url', url } });
            }
          }
        }
        if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
        continue;
      }
      if (message.role === 'assistant') {
        const blocks: Json[] = [];
        if (typeof message.content === 'string') {
          if (message.content !== '') blocks.push({ type: 'text', text: message.content });
        } else if (Array.isArray(message.content)) {
          for (const part of message.content) {
            if (isRecord(part) && part.type === 'text' && typeof part.text === 'string') {
              blocks.push({ type: 'text', text: part.text });
            }
          }
        }
        if (Array.isArray(message.tool_calls)) {
          for (const call of message.tool_calls) {
            if (!isRecord(call) || !isRecord(call.function)) continue;
            blocks.push({
              type: 'tool_use',
              id: typeof call.id === 'string' ? call.id : '',
              name: typeof call.function.name === 'string' ? call.function.name : '',
              input: argumentsOf(call.function.arguments),
            });
          }
        }
        if (blocks.length > 0) messages.push({ role: 'assistant', content: blocks });
        continue;
      }
    }
  }

  const systemText = systemParts.join('\n');
  if (systemText !== '') out.system = systemText;
  out.messages = messages;
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (typeof body.stop === 'string') {
    out.stop_sequences = [body.stop];
  } else if (Array.isArray(body.stop)) {
    const stops = body.stop.filter((s): s is string => typeof s === 'string');
    if (stops.length > 0) out.stop_sequences = stops;
  }
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required') out.tool_choice = { type: 'any' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}
