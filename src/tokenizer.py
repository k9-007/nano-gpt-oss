import tiktoken


class HarmonyTokenizer:
    """Harmony tokenizer extending O200K BPE with special chat/tool tokens."""

    SPECIAL_TOKENS = {
        "<|begin_of_text|>": 200000,
        "<|end_of_text|>": 200001,
        "<|start_header|>": 200002,
        "<|end_header|>": 200003,
        "<|eot|>": 200004,
        "<|tool_call|>": 200005,
        "<|tool_result|>": 200006,
        "<|python|>": 200007,
        "<|fim_prefix|>": 200008,
        "<|fim_middle|>": 200009,
        "<|fim_suffix|>": 200010,
    }

    def __init__(self):
        base = tiktoken.get_encoding("o200k_base")
        self.encoding = tiktoken.Encoding(
            name="harmony",
            pat_str=base._pat_str,
            mergeable_ranks=base._mergeable_ranks,
            special_tokens=self.SPECIAL_TOKENS,
        )

    def encode(self, text: str) -> list[int]:
        return self.encoding.encode(text, allowed_special="all")

    def decode(self, ids: list[int]) -> str:
        return self.encoding.decode(ids)

    @property
    def vocab_size(self) -> int:
        return 201088

    @property
    def bos_id(self) -> int:
        return self.SPECIAL_TOKENS["<|begin_of_text|>"]

    @property
    def eos_id(self) -> int:
        return self.SPECIAL_TOKENS["<|end_of_text|>"]
