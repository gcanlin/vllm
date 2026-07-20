# vLLM Speech-to-Text Entrypoint 实现与功能走读

本文基于当前仓库代码，解释 vLLM 的 speech-to-text entrypoint 如何实现：

- OpenAI-compatible HTTP API:
  - `POST /v1/audio/transcriptions`
  - `POST /v1/audio/translations`
- SSE 流式转写：
  - `stream=true`
- WebSocket 实时语音识别：
  - `WS /v1/realtime`
- 模型侧如何接入：
  - `SupportsTranscription`
  - `SupportsRealtime`

重点代码目录：

```text
vllm/entrypoints/speech_to_text/
  base/
    serving.py
    protocol.py
    utils.py
  transcription/
    api_router.py
    protocol.py
    serving.py
  translation/
    api_router.py
    protocol.py
    serving.py
  realtime/
    api_router.py
    connection.py
    protocol.py
    serving.py
  factories.py

vllm/config/speech_to_text.py
vllm/model_executor/models/interfaces.py
vllm/model_executor/models/whisper.py
vllm/model_executor/models/qwen3_asr.py
vllm/model_executor/models/qwen3_asr_realtime.py
vllm/model_executor/models/voxtral_realtime.py
```

## 总体架构

speech-to-text 在 vLLM 里不是一个独立 engine，而是 OpenAI server 上的一组 API router 和 serving wrapper。请求进入 FastAPI 后，服务层把音频转成 vLLM engine 能消费的 `EngineInput` 或 `StreamingInput`，再调用已有的 `engine_client.generate()`。

HTTP transcription/translation 的主链路：

```text
FastAPI route
  -> read_upload_with_limit()
  -> OpenAIServingTranscription / OpenAIServingTranslation
  -> OpenAISpeechToText._create_speech_to_text()
  -> _preprocess_speech_to_text()
       - 校验 language / to_language
       - decode audio + resample
       - 长音频 chunking
       - 构造 SpeechToTextParams
       - model_cls.get_generation_prompt()
       - renderer.render_cmpl_async()
  -> engine_client.generate()
  -> JSON / verbose_json / SSE stream response
```

Realtime 的主链路：

```text
WebSocket /v1/realtime
  -> RealtimeConnection.handle_connection()
  -> input_audio_buffer.append: base64 PCM16 -> np.float32 -> audio_queue
  -> input_audio_buffer.commit: start_generation()
  -> OpenAIServingRealtime.transcribe_realtime()
       - model_cls.buffer_realtime_audio()
       - render PromptType -> StreamingInput
  -> engine_client.generate(prompt=async_generator)
  -> transcription.delta / transcription.done
```

## 任务发现与路由注册

vLLM 通过模型实现的接口决定支持哪些 task。`vllm/tasks.py` 定义了 generation task：

```python
# vllm/tasks.py
GenerationTask = Literal["generate", "transcription", "realtime"]
```

V1 GPU runner 根据模型能力暴露 task：

```python
# vllm/v1/worker/gpu_model_runner.py
def get_supported_generation_tasks(self) -> list[GenerationTask]:
    model = self.get_model()
    supported_tasks = list[GenerationTask]()

    if is_text_generation_model(model):
        supported_tasks.append("generate")

    if supports_transcription(model):
        if model.supports_transcription_only:
            return ["transcription"]

        supported_tasks.append("transcription")

    if supports_realtime(model):
        supported_tasks.append("realtime")

    return supported_tasks
```

OpenAI server 启动时，根据 `supported_tasks` 注册 speech-to-text router：

```python
# vllm/entrypoints/speech_to_text/factories.py
def register_speech_to_text_api_routers(app: FastAPI,
                                        supported_tasks: tuple[SupportedTask, ...]):
    if "realtime" in supported_tasks:
        from .realtime.api_router import router as realtime_router
        app.include_router(realtime_router)

    if "transcription" in supported_tasks:
        from .transcription.api_router import router as transcription_router
        app.include_router(transcription_router)

        from .translation.api_router import router as translation_router
        app.include_router(translation_router)
```

同一个文件还负责初始化 serving object：

```python
# vllm/entrypoints/speech_to_text/factories.py
def init_speech_to_text_state(engine_client, state, args, request_logger,
                              supported_tasks):
    if "transcription" in supported_tasks:
        state.openai_serving_transcription = OpenAIServingTranscription(
            engine_client,
            state.openai_serving_models,
            request_logger=request_logger,
            enable_force_include_usage=args.enable_force_include_usage,
        )

        state.openai_serving_translation = OpenAIServingTranslation(
            engine_client,
            state.openai_serving_models,
            request_logger=request_logger,
            enable_force_include_usage=args.enable_force_include_usage,
        )

    if "realtime" in supported_tasks:
        state.openai_serving_realtime = OpenAIServingRealtime(
            engine_client,
            state.openai_serving_models,
            request_logger=request_logger,
        )
```

## HTTP Transcriptions API

`/v1/audio/transcriptions` 的 route 很薄，只负责：

- 从 multipart form 解析 `TranscriptionRequest`
- 安全读取上传文件
- 调用 serving
- 根据结果返回 JSON 或 SSE

```python
# vllm/entrypoints/speech_to_text/transcription/api_router.py
@router.post("/v1/audio/transcriptions")
@with_cancellation
@load_aware_call
async def create_transcriptions(
    raw_request: Request, request: Annotated[TranscriptionRequest, Form()]
):
    handler = transcription(raw_request)
    if handler is None:
        raise NotImplementedError("The model does not support Transcriptions API")

    audio_data = await read_upload_with_limit(request.file)

    generator = await handler.create_transcription(audio_data, request, raw_request)

    if isinstance(generator, ErrorResponse):
        return JSONResponse(
            content=generator.model_dump(), status_code=generator.error.code
        )

    elif isinstance(generator, TranscriptionResponseVariant):
        return JSONResponse(content=generator.model_dump())

    return StreamingResponse(content=generator, media_type="text/event-stream")
```

上传文件不是一次性无限读入，而是按 64 KiB chunk 读取，并在超过限制时提前失败：

```python
# vllm/entrypoints/speech_to_text/base/utils.py
async def read_upload_with_limit(file: UploadFile,
                                 max_size_mb: float | None = None) -> bytes:
    if max_size_mb is None:
        max_size_mb = envs.VLLM_MAX_AUDIO_CLIP_FILESIZE_MB

    max_bytes = int(max_size_mb * MiB_bytes)

    if file.size is not None and file.size > max_bytes:
        raise VLLMValidationError(
            "Maximum file size exceeded",
            parameter="audio_filesize_mb",
            value=file.size / MiB_bytes,
        )

    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(_READ_CHUNK_SIZE)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise VLLMValidationError(
                "Maximum file size exceeded",
                parameter="audio_filesize_mb",
                value=total / MiB_bytes,
            )
        chunks.append(chunk)

    return b"".join(chunks)
```

## TranscriptionRequest 支持的参数

`TranscriptionRequest` 是 OpenAI-compatible form schema，同时扩展了一些 vLLM 参数。

关键字段：

```python
# vllm/entrypoints/speech_to_text/transcription/protocol.py
class TranscriptionRequest(OpenAIBaseModel):
    file: UploadFile
    model: str | None = None
    language: str | None = None
    hotwords: str | None = None
    prompt: str = Field(default="")
    response_format: AudioResponseFormat = Field(default="json")
    timestamp_granularities: list[Literal["word", "segment"]] = Field(
        alias="timestamp_granularities[]", default=[]
    )

    stream: bool | None = False
    stream_include_usage: bool | None = False
    stream_continuous_usage_stats: bool | None = False

    vllm_xargs: dict[str, str | int | float | bool] | None = Field(default=None)
    to_language: str | None = None

    use_beam_search: bool = False
    n: int = 1
    length_penalty: float = 1.0
    include_stop_str_in_output: bool = False
    temperature: float = Field(default=0.0)
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    seed: int | None = Field(None, ge=_LONG_INFO.min, le=_LONG_INFO.max)
    frequency_penalty: float | None = 0.0
    repetition_penalty: float | None = None
    presence_penalty: float | None = 0.0
    max_completion_tokens: int | None = None
```

请求对象会把 API 参数封装成模型侧统一消费的 `SpeechToTextParams`：

```python
# vllm/entrypoints/speech_to_text/transcription/protocol.py
def build_stt_params(self, audio: np.ndarray, stt_config: SpeechToTextConfig,
                     model_config: ModelConfig, task_type: str) -> SpeechToTextParams:
    return SpeechToTextParams(
        audio=audio,
        stt_config=stt_config,
        model_config=model_config,
        language=self.language,
        task_type=task_type,
        request_prompt=self.prompt,
        to_language=self.to_language,
        hotwords=self.hotwords,
    )
```

采样参数转换：

```python
# vllm/entrypoints/speech_to_text/transcription/protocol.py
def to_sampling_params(self, default_max_tokens: int,
                       default_sampling_params: dict | None = None) -> SamplingParams:
    ...
    return SamplingParams.from_optional(
        temperature=temperature,
        max_tokens=max_tokens,
        seed=self.seed,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        frequency_penalty=self.frequency_penalty,
        repetition_penalty=repetition_penalty,
        presence_penalty=self.presence_penalty,
        output_kind=RequestOutputKind.DELTA
        if self.stream
        else RequestOutputKind.FINAL_ONLY,
        extra_args=self.vllm_xargs,
        skip_clone=True,
    )
```

## 核心服务：OpenAISpeechToText

`OpenAIServingTranscription` 和 `OpenAIServingTranslation` 都继承同一个 base class，只是 `task_type` 和 response type 不同。

初始化时，服务会从模型类读取 ASR 配置：

```python
# vllm/entrypoints/speech_to_text/base/serving.py
class OpenAISpeechToText(OpenAIServing):
    def __init__(..., task_type: Literal["transcribe", "translate"] = "transcribe",
                 enable_force_include_usage: bool = False):
        ...
        self.default_sampling_params = self.model_config.get_diff_sampling_param()
        self.task_type: Final = task_type

        self.asr_config = self.model_cls.get_speech_to_text_config(
            self.model_config, task_type
        )

        self.max_audio_filesize_mb = envs.VLLM_MAX_AUDIO_CLIP_FILESIZE_MB
        self.max_audio_decode_duration_s = envs.VLLM_MAX_AUDIO_DECODE_DURATION_S
```

模型类来自 registry：

```python
# vllm/entrypoints/speech_to_text/base/serving.py
@cached_property
def model_cls(self) -> type[SupportsTranscription]:
    from vllm.model_executor.model_loader import get_model_cls

    model_cls = get_model_cls(self.model_config)
    return cast(type[SupportsTranscription], model_cls)
```

### 音频 decode、resample、chunking

音频 decode 在单独 thread pool 执行，避免阻塞 event loop：

```python
# vllm/entrypoints/speech_to_text/base/serving.py
self._preprocess_executor = ThreadPoolExecutor(
    max_workers=num_audio_preprocess_workers,
    thread_name_prefix="stt-preprocess",
)
self._decode_and_chunk_speech_async = make_async_with_semaphore(
    self._decode_and_chunk_speech, executor=self._preprocess_executor
)
```

具体 decode 逻辑：

```python
# vllm/entrypoints/speech_to_text/base/serving.py
def _decode_and_chunk_speech(self, audio_data: bytes) -> tuple[list[np.ndarray], float]:
    with io.BytesIO(audio_data) as buf:
        y, sr = load_audio(
            buf,
            sr=self.asr_config.sample_rate,
            max_duration_s=self.max_audio_decode_duration_s,
        )

    duration = get_audio_duration(y=y, sr=sr)
    do_split_audio = self.asr_config.allow_audio_chunking and (
        self.asr_config.max_audio_clip_s is not None
        and duration > self.asr_config.max_audio_clip_s
    )

    if not do_split_audio:
        chunks = [y]
    else:
        chunks = split_audio(
            audio_data=y,
            sample_rate=int(sr),
            max_clip_duration_s=self.asr_config.max_audio_clip_s,
            overlap_duration_s=self.asr_config.overlap_chunk_second,
            min_energy_window_size=self.asr_config.min_energy_split_window_size,
        )

    return chunks, duration
```

chunking 的参数来自 `SpeechToTextConfig`：

```python
# vllm/config/speech_to_text.py
@config
class SpeechToTextConfig:
    sample_rate: float = 16_000
    max_audio_clip_s: int | None = 30
    overlap_chunk_second: int = 1
    min_energy_split_window_size: int | None = 1600

    @property
    def allow_audio_chunking(self) -> bool:
        return (
            self.min_energy_split_window_size is not None
            and self.max_audio_clip_s is not None
        )
```

### 预处理为 EngineInput

`_preprocess_speech_to_text()` 做了模型无关的前处理，然后把模型侧 prompt 交给 renderer：

```python
# vllm/entrypoints/speech_to_text/base/serving.py
async def _preprocess_speech_to_text(
    self,
    request: SpeechToTextRequest,
    audio_data: bytes,
    request_id: str,
) -> tuple[list[EngineInput], float]:
    request.language = self.model_cls.validate_language(request.language)
    request.to_language = (
        self.model_cls.validate_language(request.to_language)
        if request.to_language
        else None
    )

    chunks, duration = await self._decode_and_chunk_speech_async(audio_data)

    if request.language is None and getattr(
        self.model_cls, "supports_explicit_language_detection", False
    ):
        request.language = await self._detect_language(
            chunks[0], f"{request_id}-lang_detect"
        )

    parsed_prompts: list[DictPrompt] = []
    for chunk in chunks:
        stt_params = request.build_stt_params(
            audio=chunk,
            stt_config=self.asr_config,
            model_config=self.model_config,
            task_type=self.task_type,
        )
        prompt = self.model_cls.get_generation_prompt(stt_params)

        if request.response_format == "verbose_json":
            parsed_prompt = parse_enc_dec_prompt(prompt)
            parsed_prompt = self._preprocess_verbose_prompt(parsed_prompt)
        else:
            parsed_prompt = parse_model_prompt(self.model_config, prompt)

        parsed_prompts.append(parsed_prompt)

    engine_inputs = await self.renderer.render_cmpl_async(parsed_prompts)
    return engine_inputs, duration
```

这里最重要的是：entrypoint 不知道每个 ASR 模型的 prompt 格式。它只构造 `SpeechToTextParams`，然后调用：

```python
prompt = self.model_cls.get_generation_prompt(stt_params)
```

具体如何把音频放进 prompt，是模型类负责。

### 语言检测

Whisper 这类模型可以实现显式语言检测。服务侧发现 `language=None` 且模型声明支持后，会先发一次单 token generation：

```python
# vllm/entrypoints/speech_to_text/base/serving.py
async def _detect_language(self, audio_chunk: np.ndarray, request_id: str) -> str:
    prompt = self.model_cls.get_language_detection_prompt(
        audio_chunk,
        self.asr_config,
    )
    allowed_token_ids = self.model_cls.get_language_token_ids(self.tokenizer)
    sampling_params = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        allowed_token_ids=allowed_token_ids,
    )

    result_generator = self.engine_client.generate(
        prompt,
        sampling_params,
        request_id,
    )
    ...
    lang = self.model_cls.parse_language_detection_output(
        token_ids,
        self.tokenizer,
    )
    return lang
```

### 调用 engine.generate

预处理完成后，服务为每个 audio chunk 创建一个 engine request：

```python
# vllm/entrypoints/speech_to_text/base/serving.py
for request_id_item, engine_input in zip(engine_request_ids, engine_inputs):
    if isinstance(sampling_params, BeamSearchParams):
        generator = self.beam_search(
            prompt=engine_input,
            params=sampling_params,
            request_id=request_id_item,
            lora_request=lora_request,
            trace_headers=trace_headers,
        )
    else:
        generator = self.engine_client.generate(
            engine_input,
            sampling_params,
            request_id_item,
            lora_request=lora_request,
            trace_headers=trace_headers,
        )

    list_result_generator.append(generator)
```

beam search 当前不支持 streaming：

```python
if request.stream and request.use_beam_search:
    return self.create_error_response(
        "Streaming is not currently supported with beam search"
    )
```

### 非流式 response

非流式时，服务收集所有 chunk 输出，并按语言决定 chunk 之间是否插空格：

```python
# vllm/entrypoints/speech_to_text/base/serving.py
separator = asr_inter_chunk_separator(
    request.language, self.model_cls.no_space_languages
)
...
raw_text = op.outputs[0].text
chunk_text_parts[idx].append(
    self.model_cls.post_process_output(raw_text)
)
...
text_parts = [text for text_part in chunk_text_parts for text in text_part]
text = separator.join(text_parts)
```

中文、日文默认不插空格：

```python
# vllm/entrypoints/speech_to_text/base/serving.py
def asr_inter_chunk_separator(language: str | None,
                              no_space_languages: Set[str]) -> str:
    return "" if language and language.lower() in no_space_languages else " "
```

transcription 的普通 JSON response 会带 duration usage：

```python
usage = {
    "type": "duration",
    "seconds": int(math.ceil(duration_s)),
}
final_response = TranscriptionResponse(text=text, usage=usage)
```

translation response 当前不带 usage。

### verbose_json 与 segment timestamp

`verbose_json` 只允许支持 `supports_segment_timestamp` 的模型。服务侧用 timestamp token 切 segment：

```python
# vllm/entrypoints/speech_to_text/base/serving.py
if (
    request.response_format == "verbose_json"
    and not self.model_cls.supports_segment_timestamp
):
    return self.create_error_response(
        f"Currently do not support verbose_json for {request.model}"
    )
```

segment 构造逻辑：

```python
# vllm/entrypoints/speech_to_text/base/serving.py
def _get_verbose_segments(self, tokens: tuple,
                          log_probs: FlatLogprobs | list[dict[int, Logprob]],
                          request: SpeechToTextRequest,
                          segment_class: type[SpeechToTextSegment],
                          start_time: float = 0) -> list[SpeechToTextSegment]:
    BASE_OFFSET = 0.02
    init_token = self.tokenizer.encode("<|0.00|>", add_special_tokens=False)[0]
    ...
    if token >= init_token and tokens_with_start[idx - 1] >= init_token:
        sliced_timestamp_tokens = tokens_with_start[last_timestamp_start:idx]
        start_timestamp = sliced_timestamp_tokens[0] - init_token
        end_timestamp = sliced_timestamp_tokens[-1] - init_token
        text = self.tokenizer.decode(sliced_timestamp_tokens[1:-1])
        ...
        segment_class(
            id=len(segments),
            seek=start_time,
            start=start_time + BASE_OFFSET * start_timestamp,
            end=start_time + BASE_OFFSET * end_timestamp,
            temperature=request.temperature,
            text=text,
            compression_ratio=len(text_bytes) / len(zlib.compress(text_bytes)),
            tokens=sliced_timestamp_tokens[1:-1],
            avg_logprob=avg_logprob / (idx - last_timestamp_start),
        )
```

## SSE 流式转写

当 `stream=true` 时，sampling params 使用 `RequestOutputKind.DELTA`，服务把每个 engine delta 包成 SSE：

```python
# vllm/entrypoints/speech_to_text/base/serving.py
delta_message = DeltaMessage(content=output.text)
completion_tokens += len(output.token_ids)

if output.finish_reason is None:
    choice_data = response_stream_choice_class(delta=delta_message)
else:
    choice_data = response_stream_choice_class(
        delta=delta_message,
        finish_reason=output.finish_reason,
        stop_reason=output.stop_reason,
    )

chunk = stream_response_class(
    id=request_id,
    object=chunk_object_type,
    created=created_time,
    choices=[choice_data],
    model=model_name,
)

data = chunk.model_dump_json(exclude_unset=True)
yield f"data: {data}\n\n"
```

如果请求了 usage，最后再发一个空 choices 的 usage chunk：

```python
if include_usage:
    final_usage = UsageInfo(
        prompt_tokens=num_prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=num_prompt_tokens + completion_tokens,
    )

    final_usage_chunk = stream_response_class(
        id=request_id,
        object=chunk_object_type,
        created=created_time,
        choices=[],
        model=model_name,
        usage=final_usage,
    )
    yield f"data: {final_usage_data}\n\n"

yield "data: [DONE]\n\n"
```

客户端示例：

```python
# examples/speech_to_text/openai/openai_transcription_client.py
transcription = await client.audio.transcriptions.create(
    file=f,
    model=model,
    language="en",
    prompt=prompt or "",
    response_format="json",
    temperature=0.0,
    extra_body=dict(seed=420, top_p=0.6, hotwords=hotwords),
    stream=True,
)
async for chunk in transcription:
    if chunk.choices:
        content = chunk.choices[0].get("delta", {}).get("content")
        print(content, end="", flush=True)
```

## Translation API

translation 与 transcription 共用 `OpenAISpeechToText`，区别是：

- `task_type="translate"`
- response object 是 `translation.chunk` 或 translation JSON
- 模型在 `get_generation_prompt()` 里根据 `task_type` 构造翻译 prompt

```python
# vllm/entrypoints/speech_to_text/translation/serving.py
class OpenAIServingTranslation(OpenAISpeechToText):
    def __init__(...):
        super().__init__(
            engine_client=engine_client,
            models=models,
            request_logger=request_logger,
            return_tokens_as_token_ids=return_tokens_as_token_ids,
            task_type="translate",
            enable_force_include_usage=enable_force_include_usage,
        )
```

## 模型接口：SupportsTranscription

一个模型要支持 `/v1/audio/transcriptions` 和 `/v1/audio/translations`，需要实现 `SupportsTranscription`。

核心协议：

```python
# vllm/model_executor/models/interfaces.py
@runtime_checkable
class SupportsTranscription(Protocol):
    supported_languages: ClassVar[Mapping[str, str]]
    supports_transcription: ClassVar[Literal[True]] = True
    supports_transcription_only: ClassVar[bool] = False
    supports_segment_timestamp: ClassVar[bool] = False
    supports_explicit_language_detection: ClassVar[bool] = False
    no_space_languages: ClassVar[set[str]] = {"ja", "zh"}

    @classmethod
    def get_generation_prompt(cls, stt_params: SpeechToTextParams) -> PromptType:
        ...

    @classmethod
    def get_speech_to_text_config(
        cls, model_config: ModelConfig, task_type: Literal["transcribe", "translate"]
    ) -> SpeechToTextConfig:
        ...

    @classmethod
    def get_num_audio_tokens(
        cls, audio_duration_s: float, stt_config: SpeechToTextConfig,
        model_config: ModelConfig
    ) -> int | None:
        return None

    @classmethod
    def post_process_output(cls, text: str) -> str:
        return text
```

### Whisper 示例

Whisper 是 encoder-decoder ASR 模型，只支持 transcription，不支持普通 text generation：

```python
# vllm/model_executor/models/whisper.py
class WhisperForConditionalGeneration(..., SupportsTranscription,
                                      SupportsMultiModal, SupportsLoRA):
    supports_transcription_only = True
    supports_segment_timestamp = True
    supports_explicit_language_detection = True
    supported_languages = ISO639_1_SUPPORTED_LANGS
```

Whisper prompt 是 `ExplicitEncoderDecoderPrompt`：音频放 encoder，文本 prompt 放 decoder。

```python
# vllm/model_executor/models/whisper.py
@classmethod
def get_generation_prompt(cls, stt_params: SpeechToTextParams) -> PromptType:
    audio = stt_params.audio
    stt_config = stt_params.stt_config
    language = stt_params.language
    task_type = stt_params.task_type
    request_prompt = stt_params.request_prompt

    if language is None:
        raise ValueError("Language must be specified when creating the Whisper prompt")

    decoder_text = (
        f"<|prev|>{request_prompt}" if request_prompt else ""
    ) + f"<|startoftranscript|><|{language}|><|{task_type}|><|notimestamps|>"

    return ExplicitEncoderDecoderPrompt(
        encoder_prompt=TextPrompt(
            prompt="",
            multi_modal_data={"audio": (audio, stt_config.sample_rate)},
        ),
        decoder_prompt=TextPrompt(prompt=decoder_text),
    )
```

Whisper 的语言检测也在模型类里实现：

```python
# vllm/model_executor/models/whisper.py
@classmethod
def get_language_detection_prompt(cls, audio: np.ndarray,
                                  stt_config: SpeechToTextConfig) -> PromptType:
    return ExplicitEncoderDecoderPrompt(
        encoder_prompt=TextPrompt(
            prompt="",
            multi_modal_data={"audio": (audio, stt_config.sample_rate)},
        ),
        decoder_prompt=TextPrompt(prompt="<|startoftranscript|>"),
    )

@classmethod
def parse_language_detection_output(cls, token_ids: list[int],
                                    tokenizer: object) -> str | None:
    decoded = tokenizer.decode([token_ids[0]], skip_special_tokens=False)
    assert decoded.startswith("<|") and decoded.endswith("|>")
    lang_code = decoded[2:-2]
    assert lang_code in cls.supported_languages
    return lang_code
```

### Qwen3-ASR 示例

Qwen3-ASR 是 decoder-only multimodal 模型，音频通过 multimodal placeholder 合进 prompt：

```python
# vllm/model_executor/models/qwen3_asr.py
class Qwen3ASRForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsPP,
    SupportsMRoPE,
    SupportsTranscription,
    SupportsLoRA,
):
    supported_languages = ISO639_1_SUPPORTED_LANGS

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("audio"):
            return "<|audio_start|><|audio_pad|><|audio_end|>"
        raise ValueError("Only audio modality is supported")
```

音频 tower 生成 embeddings，然后 `embed_input_ids()` 把 audio embeddings scatter 到 text embeddings：

```python
# vllm/model_executor/models/qwen3_asr.py
def _process_audio_input(self, audio_input: Qwen2_5OmniAudioFeatureInputs):
    input_features = audio_input["input_features"]
    audio_feature_lengths = audio_input["audio_feature_lengths"]

    audio_output_lengths = _get_feat_extract_output_lengths(audio_feature_lengths)

    audio_features = self.audio_tower(
        input_features.to(self.audio_tower.dtype),
        feature_lens=audio_feature_lengths,
        aftercnn_lens=audio_output_lengths,
    )
    return audio_features.split(audio_output_lengths.tolist())

def embed_input_ids(self, input_ids, multimodal_embeddings=None,
                    *, is_multimodal=None):
    inputs_embeds = self._embed_text_input_ids(
        input_ids,
        self.language_model.embed_input_ids,
        is_multimodal=is_multimodal,
    )

    if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
        return inputs_embeds

    return _merge_multimodal_embeddings(
        inputs_embeds=inputs_embeds,
        multimodal_embeddings=multimodal_embeddings,
        is_multimodal=is_multimodal,
    )
```

Qwen3-ASR 的 HTTP prompt：

```python
# vllm/model_executor/models/qwen3_asr.py
@classmethod
def get_generation_prompt(cls, stt_params: SpeechToTextParams) -> PromptType:
    audio = stt_params.audio
    model_config = stt_params.model_config
    language = stt_params.language
    task_type = stt_params.task_type
    request_prompt = stt_params.request_prompt
    to_language = stt_params.to_language

    tokenizer = cached_tokenizer_from_config(model_config)
    audio_placeholder = cls.get_placeholder_str("audio", 0)

    context = _sanitize_transcription_user_text(request_prompt)
    system_turn = f"<|im_start|>system\n{context}<|im_end|>\n" if context else ""

    prompt = (
        f"{system_turn}"
        f"<|im_start|>user\n{audio_placeholder}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    lang_code = to_language if task_type == "translate" else language
    if lang_code is not None:
        full_lang_name = cls.supported_languages.get(lang_code, lang_code)
        prompt += f"language {full_lang_name}{_ASR_TEXT_TAG}"

    prompt_token_ids = tokenizer.encode(prompt)

    return TokensPrompt(
        prompt_token_ids=prompt_token_ids,
        multi_modal_data={"audio": audio},
    )
```

模型原始输出可能带结构化前缀，因此模型类负责后处理：

```python
# vllm/model_executor/models/qwen3_asr.py
@classmethod
def post_process_output(cls, text: str) -> str:
    if not text:
        return ""

    if _ASR_TEXT_TAG not in text:
        return text

    _, text_part = text.rsplit(_ASR_TEXT_TAG, 1)
    return text_part
```

## Realtime API

Realtime API 是 WebSocket，不是 SSE。入口：

```python
# vllm/entrypoints/speech_to_text/realtime/api_router.py
@router.websocket("/v1/realtime")
async def realtime_endpoint(websocket: WebSocket):
    app = websocket.app
    serving = app.state.openai_serving_realtime

    connection = RealtimeConnection(websocket, serving)
    await connection.handle_connection()
```

协议对象：

```python
# vllm/entrypoints/speech_to_text/realtime/protocol.py
class InputAudioBufferAppend(OpenAIBaseModel):
    type: Literal["input_audio_buffer.append"] = "input_audio_buffer.append"
    audio: str  # base64-encoded PCM16 @ 16kHz

class InputAudioBufferCommit(OpenAIBaseModel):
    type: Literal["input_audio_buffer.commit"] = "input_audio_buffer.commit"
    final: bool = False

class SessionUpdate(OpenAIBaseModel):
    type: Literal["session.update"] = "session.update"
    model: str | None = None

class SessionCreated(OpenAIBaseModel):
    type: Literal["session.created"] = "session.created"
    id: str = Field(default_factory=lambda: f"sess-{random_uuid()}")
    created: int = Field(default_factory=lambda: int(time.time()))

class TranscriptionDelta(OpenAIBaseModel):
    type: Literal["transcription.delta"] = "transcription.delta"
    delta: str

class TranscriptionDone(OpenAIBaseModel):
    type: Literal["transcription.done"] = "transcription.done"
    text: str
    usage: UsageInfo | None = None
```

### WebSocket connection lifecycle

连接建立后，服务立即发送 `session.created`：

```python
# vllm/entrypoints/speech_to_text/realtime/connection.py
async def handle_connection(self):
    await self.websocket.accept()
    self._is_connected = True

    await self.send(SessionCreated())

    try:
        while True:
            message = await self.websocket.receive_text()
            try:
                event = json.loads(message)
                await self.handle_event(event)
            except json.JSONDecodeError:
                await self.send_error("Invalid JSON", "invalid_json")
            except Exception as e:
                await self.send_error(sanitize_message(str(e)), "processing_error")
    finally:
        await self.cleanup()
```

`session.update` 只做模型校验：

```python
# vllm/entrypoints/speech_to_text/realtime/connection.py
if event_type == "session.update":
    model = event.get("model")
    if model is None:
        await self.send_error("Missing required field: model", "invalid_event")
        return
    err = self._check_model(model)
    if err is not None:
        await self.send_error(err.error.message, "model_not_found")
        return
    self._is_model_validated = True
```

`input_audio_buffer.append` 把 base64 PCM16 转为 float32 waveform，并写入 queue：

```python
# vllm/entrypoints/speech_to_text/realtime/connection.py
elif event_type == "input_audio_buffer.append":
    append_event = InputAudioBufferAppend(**event)
    audio_bytes = base64.b64decode(append_event.audio)
    audio_array = (
        np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
        / 32768.0
    )

    if len(audio_array) == 0:
        raise VLLMValidationError("Can't process empty audio.")

    self.audio_queue.put_nowait(audio_array)
```

`input_audio_buffer.commit` 负责启动 generation。`final=true` 表示音频结束，向 queue 写入 sentinel `None`：

```python
# vllm/entrypoints/speech_to_text/realtime/connection.py
elif event_type == "input_audio_buffer.commit":
    if not self._is_model_validated:
        await self.send_error(err_msg, "model_not_validated")
        return

    commit_event = InputAudioBufferCommit(**event)
    if commit_event.final:
        self.audio_queue.put_nowait(None)
    else:
        await self.start_generation()
```

### Realtime generation

`start_generation()` 创建两个 stream：

- `audio_stream`: 从 WebSocket audio queue 读 numpy audio
- `input_stream`: 把模型上一次输出 token 回灌给模型侧 buffer，用于 autoregressive streaming

```python
# vllm/entrypoints/speech_to_text/realtime/connection.py
async def start_generation(self):
    if self.generation_task is not None and not self.generation_task.done():
        logger.warning("Generation already in progress, ignoring commit")
        return

    audio_stream = self.audio_stream_generator()
    input_stream = asyncio.Queue[list[int]]()

    streaming_input_gen = self.serving.transcribe_realtime(
        audio_stream, input_stream
    )

    self.generation_task = asyncio.create_task(
        self._run_generation(streaming_input_gen, input_stream)
    )
```

`_run_generation()` 把 async generator 直接传给 engine：

```python
# vllm/entrypoints/speech_to_text/realtime/connection.py
sampling_params = SamplingParams.from_optional(
    temperature=0.0,
    max_tokens=self.serving.model_cls.realtime_max_tokens,
    output_kind=RequestOutputKind.DELTA,
    skip_clone=True,
)

result_gen = self.serving.engine_client.generate(
    prompt=streaming_input_gen,
    sampling_params=sampling_params,
    request_id=request_id,
)

async for output in result_gen:
    delta = output.outputs[0].text
    full_text += delta

    input_stream.put_nowait(list(output.outputs[0].token_ids))
    await self.send(TranscriptionDelta(delta=delta))
```

结束时发 `transcription.done`：

```python
usage = UsageInfo(
    prompt_tokens=prompt_token_ids_len,
    completion_tokens=completion_tokens_len,
    total_tokens=prompt_token_ids_len + completion_tokens_len,
)

await self.send(TranscriptionDone(text=full_text, usage=usage))
```

### OpenAIServingRealtime

Realtime serving 的职责是把模型侧 `PromptType` stream render 成 engine 可消费的 `StreamingInput`：

```python
# vllm/entrypoints/speech_to_text/realtime/serving.py
async def transcribe_realtime(
    self,
    audio_stream: AsyncGenerator[np.ndarray, None],
    input_stream: asyncio.Queue[list[int]],
) -> AsyncGenerator[StreamingInput, None]:
    model_config = self.model_config
    renderer = self.renderer

    stream_input_iter = cast(
        AsyncGenerator[PromptType, None],
        self.model_cls.buffer_realtime_audio(
            audio_stream, input_stream, model_config
        ),
    )

    async for prompt in stream_input_iter:
        parsed_prompt = parse_model_prompt(model_config, prompt)
        (engine_input,) = await renderer.render_cmpl_async([parsed_prompt])

        yield StreamingInput(prompt=engine_input)
```

### Realtime 模型接口

模型通过 `SupportsRealtime` 声明实时能力：

```python
# vllm/model_executor/models/interfaces.py
@runtime_checkable
class SupportsRealtime(Protocol):
    supports_realtime: ClassVar[Literal[True]] = True

    realtime_max_tokens: ClassVar[int] = 1

    @classmethod
    async def buffer_realtime_audio(
        cls,
        audio_stream: AsyncGenerator[np.ndarray, None],
        input_stream: asyncio.Queue[list[int]],
        model_config: ModelConfig,
    ) -> AsyncGenerator[PromptType, None]:
        ...
```

#### Qwen3-ASR realtime

Qwen3-ASR realtime 使用固定 5 秒 segment buffering：

```python
# vllm/model_executor/models/qwen3_asr_realtime.py
class Qwen3ASRRealtimeBuffer:
    def __init__(self, sampling_rate: int, segment_duration_s: float = 5.0):
        self._sampling_rate = sampling_rate
        self._segment_size = int(segment_duration_s * sampling_rate)
        self._buffer_size = _PRE_ALLOCATE_BUFFER_SIZE_IN_S * sampling_rate
        self._buffer: np.ndarray = np.empty(self._buffer_size, dtype=np.float32)
        self._filled_len = 0

    def write_audio(self, audio: np.ndarray) -> None:
        ...

    def read_audio(self) -> np.ndarray | None:
        if self._filled_len < self._segment_size:
            return None

        segment = self._buffer[: self._segment_size].copy()
        ...
        return segment

    def flush(self) -> np.ndarray | None:
        if self._filled_len == 0:
            return None
        audio = self._buffer[: self._filled_len].copy()
        self._filled_len = 0
        return audio
```

模型类继承普通 Qwen3-ASR，同时实现 `SupportsRealtime`：

```python
# vllm/model_executor/models/qwen3_asr_realtime.py
class Qwen3ASRRealtimeGeneration(Qwen3ASRForConditionalGeneration,
                                 SupportsRealtime):
    realtime_max_tokens = 64

    @classmethod
    async def buffer_realtime_audio(
        cls,
        audio_stream: AsyncGenerator[np.ndarray, None],
        input_stream: asyncio.Queue[list[int]],
        model_config: ModelConfig,
    ) -> AsyncGenerator[PromptType, None]:
        processor = cached_processor_from_config(model_config)
        feature_extractor = processor.feature_extractor
        sampling_rate = feature_extractor.sampling_rate
        tokenizer = cached_tokenizer_from_config(model_config)

        segment_duration_s = 5.0
        buffer = Qwen3ASRRealtimeBuffer(
            sampling_rate=sampling_rate,
            segment_duration_s=segment_duration_s,
        )

        audio_placeholder = cls.get_placeholder_str("audio", 0)
        prompt_template = (
            f"<|im_start|>user\n{audio_placeholder}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        prompt_token_ids = tokenizer.encode(prompt_template)

        async for audio_chunk in audio_stream:
            buffer.write_audio(audio_chunk)

            while (segment := buffer.read_audio()) is not None:
                yield TokensPrompt(
                    prompt_token_ids=prompt_token_ids,
                    multi_modal_data={"audio": segment},
                )

        remaining = buffer.flush()
        if remaining is not None and len(remaining) > 0:
            yield TokensPrompt(
                prompt_token_ids=prompt_token_ids,
                multi_modal_data={"audio": remaining},
            )
```

注意：Qwen3-ASR realtime 注册了独立 multimodal processor。原因是 realtime prompt update 逻辑和普通 transcription 不同：

```python
# vllm/model_executor/models/qwen3_asr_realtime.py
# NOTE: A separate model class is required here because the multimodal
# processor registry binds one processor per model class. The realtime
# endpoint needs a different processor (Qwen3ASRRealtimeMultiModalProcessor)
# than the base transcription endpoint, so we register it on this subclass.
@MULTIMODAL_REGISTRY.register_processor(
    Qwen3ASRRealtimeMultiModalProcessor,
    info=Qwen3ASRProcessingInfo,
    dummy_inputs=Qwen3ASRDummyInputsBuilder,
)
class Qwen3ASRRealtimeGeneration(...):
    ...
```

#### Voxtral realtime

Voxtral realtime 是更严格的 streaming 模型：它维护 look-back/look-ahead frame，且把上一次输出 token 回灌到下一次输入。

关键 buffer：

```python
# vllm/model_executor/models/voxtral_realtime.py
class VoxtralRealtimeBuffer:
    def __init__(self, config: AudioConfig, prompt_tokens: list[int]) -> None:
        self._config = config
        self._look_ahead_in_samples = self._ms_to_samples(
            self._config.streaming_look_ahead_ms
        )
        self._look_back_in_samples = self._ms_to_samples(
            self._config.streaming_look_back_ms
        )
        self._audio_queue: asyncio.Queue[np.ndarray | None] = asyncio.Queue()
        self._leftover: np.ndarray | None = None
        self._token_queue: asyncio.Queue[int] = asyncio.Queue()

        self._initial_end = len(prompt_tokens) * self._config.raw_audio_length_per_tok
        for token in prompt_tokens:
            self._token_queue.put_nowait(token)
```

它输出的是一帧音频加一段 token prompt：

```python
# vllm/model_executor/models/voxtral_realtime.py
async def get_input_stream(self) -> AsyncGenerator[StreamingInput]:
    for frame_size, num_tokens in self._generate_frame_size_and_num_tokens():
        next_tokens = [await self._token_queue.get() for _ in range(num_tokens)]
        ...
        yield StreamingInput(
            TokensPrompt(
                prompt_token_ids=next_tokens,
                multi_modal_data={"audio": (frame, None)},
            )
        )
```

模型侧同时启动两个后台任务：

- `feed_audio`: WebSocket audio stream -> Voxtral buffer
- `feed_tokens`: engine output token -> Voxtral buffer

```python
# vllm/model_executor/models/voxtral_realtime.py
@classmethod
async def buffer_realtime_audio(...):
    tokenizer = cached_tokenizer_from_config(model_config)
    audio_encoder = tokenizer.instruct.audio_encoder
    config = audio_encoder.audio_config

    prompt_tokens = (
        tokenizer.instruct.start() + audio_encoder.encode_streaming_tokens()
    )

    left_pad, right_pad = audio_encoder.get_padding_audio()
    buffer = VoxtralRealtimeBuffer(config, prompt_tokens)

    async def feed_audio():
        yielded_first_chunk = False
        async for audio_chunk in audio_stream:
            if not yielded_first_chunk:
                yielded_first_chunk = True
                await buffer.append_audio(left_pad.audio_array)
            await buffer.append_audio(audio_chunk)
        await buffer.append_audio(right_pad.audio_array)
        await buffer.append_audio(None)

    async def feed_tokens():
        while True:
            all_outputs = await asyncio.wait_for(
                input_stream.get(),
                timeout=VLLM_ENGINE_ITERATION_TIMEOUT_S,
            )
            await buffer.append_tokens(all_outputs[-1:])

    audio_task = asyncio.create_task(feed_audio())
    token_task = asyncio.create_task(feed_tokens())

    try:
        async for streaming_input in buffer.get_input_stream():
            yield streaming_input.prompt
    finally:
        audio_task.cancel()
        token_task.cancel()
```

## 客户端协议示例

Realtime 客户端发送 PCM16 16kHz mono，base64 编码：

```python
# examples/speech_to_text/realtime/openai_realtime_client.py
def audio_to_pcm16_base64(audio_path: str) -> str:
    audio, _ = load_audio(audio_path, sr=16000, mono=True)
    pcm16 = (audio * 32767).astype(np.int16)
    return base64.b64encode(pcm16.tobytes()).decode("utf-8")
```

完整事件序列：

```python
# examples/speech_to_text/realtime/openai_realtime_client.py
async with websockets.connect(uri) as ws:
    response = json.loads(await ws.recv())
    assert response["type"] == "session.created"

    await ws.send(json.dumps({"type": "session.update", "model": model}))
    await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

    for i in range(0, len(audio_bytes), chunk_size):
        chunk = audio_bytes[i : i + chunk_size]
        await ws.send(json.dumps({
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(chunk).decode("utf-8"),
        }))

    await ws.send(json.dumps({
        "type": "input_audio_buffer.commit",
        "final": True,
    }))

    while True:
        response = json.loads(await ws.recv())
        if response["type"] == "transcription.delta":
            print(response["delta"], end="", flush=True)
        elif response["type"] == "transcription.done":
            print(response["text"])
            break
```

## 功能清单

当前 speech-to-text entrypoint 具备这些功能：

| 功能 | HTTP transcription | HTTP translation | Realtime |
|---|---:|---:|---:|
| OpenAI-compatible route | yes | yes | WebSocket-style |
| multipart audio upload | yes | yes | no |
| PCM16 chunk append | no | no | yes |
| audio decode/resample | yes | yes | client/realtime model dependent |
| upload size limit | yes | yes | append chunk size check |
| max decode duration | yes | yes | no direct decode path |
| long audio chunking | yes | yes | model-specific streaming buffer |
| language validation | yes | yes | model validation only |
| explicit language detection | model-dependent | model-dependent | no shared path |
| prompt/hotwords plumbing | yes | yes | model-specific |
| beam search | yes, non-stream only | yes, non-stream only | no |
| SSE streaming | yes | yes | no |
| WebSocket delta events | no | no | yes |
| usage stats | transcription duration / streaming tokens | streaming tokens | tokens |
| verbose_json segment timestamps | model-dependent | model-dependent | no |

## 加一个 ASR 模型需要做什么

最小 HTTP ASR 接入：

```python
class MyASRModel(nn.Module, SupportsMultiModal, SupportsTranscription):
    supported_languages = ISO639_1_SUPPORTED_LANGS

    @classmethod
    def get_speech_to_text_config(cls, model_config, task_type):
        processor = cached_processor_from_config(model_config)
        return SpeechToTextConfig(
            max_audio_clip_s=processor.feature_extractor.chunk_length,
            sample_rate=processor.feature_extractor.sampling_rate,
        )

    @classmethod
    def get_generation_prompt(cls, stt_params: SpeechToTextParams) -> PromptType:
        return TokensPrompt(
            prompt_token_ids=...,
            multi_modal_data={"audio": stt_params.audio},
        )

    @classmethod
    def post_process_output(cls, text: str) -> str:
        return text
```

如果模型只服务 ASR，不应该暴露普通 generation：

```python
supports_transcription_only = True
```

如果支持 `verbose_json` segment timestamps：

```python
supports_segment_timestamp = True
```

如果需要显式语言检测：

```python
supports_explicit_language_detection = True

@classmethod
def get_language_detection_prompt(cls, audio, stt_config) -> PromptType:
    ...

@classmethod
def get_language_token_ids(cls, tokenizer) -> list[int] | None:
    ...

@classmethod
def parse_language_detection_output(cls, token_ids, tokenizer) -> str:
    ...
```

最小 realtime 接入：

```python
class MyRealtimeASRModel(MyASRModel, SupportsRealtime):
    realtime_max_tokens = 64

    @classmethod
    async def buffer_realtime_audio(
        cls,
        audio_stream: AsyncGenerator[np.ndarray, None],
        input_stream: asyncio.Queue[list[int]],
        model_config: ModelConfig,
    ) -> AsyncGenerator[PromptType, None]:
        tokenizer = cached_tokenizer_from_config(model_config)
        prompt_token_ids = tokenizer.encode("...")

        async for audio_chunk in audio_stream:
            yield TokensPrompt(
                prompt_token_ids=prompt_token_ids,
                multi_modal_data={"audio": audio_chunk},
            )
```

实际 realtime 模型通常还需要：

- 自定义 buffer，把短 audio chunk 聚合成模型可处理的 frame/segment。
- 根据模型需要添加 left/right padding 或 look-back/look-ahead。
- 如果模型需要 autoregressive streaming token，把 `input_stream` 中的上一次输出 token 回灌给下一帧。
- 必要时注册独立 multimodal processor，因为 `MULTIMODAL_REGISTRY` 是按模型类绑定 processor 的。

## 当前限制和注意点

- HTTP `stream=true` 不支持 beam search。
- `verbose_json` 只支持模型声明 `supports_segment_timestamp=True` 的情况。
- `verbose_json` 的 segment 解析假设模型输出类似 Whisper 的 timestamp tokens。
- `srt` / `vtt` 虽在 `AudioResponseFormat` 类型中，但服务层当前只接受 `text`、`json`、`verbose_json`。
- SSE 流式输出目前直接转发 delta；如果模型输出带结构化前缀，可能需要模型/服务层额外 buffering 才能在 streaming 中干净剥离。
- Realtime API 当前协议很轻量：`session.update` 主要校验模型，采样参数固定在服务侧，尚未像 HTTP transcription 一样暴露完整 sampling 参数。
- Realtime 输入要求 PCM16、16kHz、mono、base64；普通音频文件 decode/resample 是示例客户端或调用方负责。
