# Contributing

To work on MLX-VLM in editable mode run:

```bash
pip install -e .
```

Check that the model weights are available in the `safetensors` format, convert if necessary and add the model file to `mlx_vlm/models`.

Tests can be run from the `mlx_vlm/` directory:

```bash
python -m unittest discover tests/
```

Please format code using `pre-commit` before submitting a pull request.

## Native chat capabilities

Models whose default chat encoder is implemented in Python can export
`NATIVE_CHAT_CAPABILITIES` from their model module. This is a set of strings:
`tools` means the encoder supports tool definitions, assistant tool calls and
tool-result messages; `reasoning` means it supports the `enable_thinking` option.
Declare only capabilities implemented by the native encoder.

Clients can resolve the module with `mlx_vlm.utils.get_model_and_args(config)`
and read `getattr(module, "NATIVE_CHAT_CAPABILITIES", frozenset())` without
loading model weights or downloading a tokenizer. The loader handles model-type
aliases, so clients do not need model-family or repository-name allowlists.

These declarations describe the default encoder only. A checkpoint-provided or
request-level custom chat template must be assessed separately. Absence of a
declaration does not mean a checkpoint's Jinja template lacks those capabilities.
