import torch
import clip
import torch.nn as nn
from .VitaCLIP_text_encoder_utils import SimpleTokenizer as _Tokenizer


def _find_subsequence(sequence, subsequence):
    """Return the indices of a contiguous subsequence; empty list if not found."""
    max_start = len(sequence) - len(subsequence) + 1
    for start in range(max_start):
        if sequence[start:start + len(subsequence)] == subsequence:
            return list(range(start, start + len(subsequence)))
    return []

def text_prompt(data):
    text_aug = 'This is a video about {}'
    classes = torch.cat([clip.tokenize(text_aug.format(c)) for i, c in data.classes])
    return classes, classes.size(0)


class AttributePromptLearner(nn.Module):
    def __init__(
        self,
        classnames,
        clip_model,
        num_ctx_tokens,
        attributes,
        attribute_ctx_tokens,
        template="a video about {}.",
        class_specific=False,
    ):
        super().__init__()

        if not isinstance(classnames, (list, tuple)) or len(classnames) == 0:
            raise ValueError("classnames must be a non-empty sequence")
        if not isinstance(attributes, (list, tuple)) or len(attributes) == 0:
            raise ValueError("attributes must be a non-empty sequence")

        self._tokenizer = _Tokenizer()
        self.classnames = [name.replace("_", " ") for name in classnames]
        self.n_cls = len(self.classnames)
        self.attributes = [str(attr).strip() for attr in attributes]
        self.class_specific = class_specific

        token_embedding = clip_model.token_embedding
        dtype = token_embedding.weight.dtype
        ctx_dim = token_embedding.weight.shape[1]

        if isinstance(attribute_ctx_tokens, int):
            attribute_ctx_tokens = [attribute_ctx_tokens] * len(self.attributes)
        elif len(attribute_ctx_tokens) != len(self.attributes):
            raise ValueError("attribute_ctx_tokens must match number of attributes")

        if num_ctx_tokens < 0:
            raise ValueError("num_ctx_tokens must be non-negative")
        if any(n < 0 for n in attribute_ctx_tokens):
            raise ValueError("attribute_ctx_tokens must be non-negative")

        self.n_ctx_main = num_ctx_tokens
        self.attribute_ctx_counts = attribute_ctx_tokens

        if class_specific:
            ctx_shape = (self.n_cls, self.n_ctx_main, ctx_dim)
            self.ctx_main = nn.Parameter(torch.empty(ctx_shape, dtype=dtype))
        else:
            self.ctx_main = nn.Parameter(torch.empty(self.n_ctx_main, ctx_dim, dtype=dtype))
        nn.init.normal_(self.ctx_main, std=0.02)

        self.attribute_ctx = nn.ParameterList()
        for n_ctx in self.attribute_ctx_counts:
            if class_specific:
                param = nn.Parameter(torch.empty(self.n_cls, n_ctx, ctx_dim, dtype=dtype))
            else:
                param = nn.Parameter(torch.empty(n_ctx, ctx_dim, dtype=dtype))
            if n_ctx > 0:
                nn.init.normal_(param, std=0.01)
            self.attribute_ctx.append(param)

        prompts = []
        for name in self.classnames:
            pieces = []
            for attr, n_ctx in zip(self.attributes, self.attribute_ctx_counts):
                if n_ctx > 0:
                    pieces.append(" ".join(["X"] * n_ctx))
                pieces.append(attr)
            if self.n_ctx_main > 0:
                pieces.append(" ".join(["X"] * self.n_ctx_main))
            pieces.append(template.format(name))
            prompt = " ".join(piece for piece in pieces if piece).strip()
            prompts.append(prompt)

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])

        self.attribute_tokens = []
        offset = 1
        for idx, (attr, n_ctx) in enumerate(zip(self.attributes, self.attribute_ctx_counts)):
            offset += n_ctx
            attr_token_len = len(self._tokenizer.encode(attr))
            token_slice = embedding[:, offset : offset + attr_token_len, :]
            self.register_buffer(f"token_attribute_{idx}", token_slice)
            self.attribute_tokens.append(f"token_attribute_{idx}")
            offset += attr_token_len

        suffix_start = offset + self.n_ctx_main
        self.register_buffer("token_suffix", embedding[:, suffix_start:, :])
        self.register_buffer("tokenized_prompts", tokenized_prompts)

    def _expand_param(self, param, indices):
        if param.shape[0] == 0:
            return torch.zeros(indices.size(0), 0, param.shape[-1], device=param.device, dtype=param.dtype)

        if param.dim() == 2:
            param = param.unsqueeze(0).expand(indices.size(0), -1, -1)
        else:
            param = param[indices]
        return param

    def forward(self, class_ids=None):
        if class_ids is None:
            indices = torch.arange(self.n_cls, device=self.token_prefix.device)
        else:
            indices = class_ids.to(self.token_prefix.device, dtype=torch.long)

        pieces = [self.token_prefix[indices]]

        for name, ctx_param in zip(self.attribute_tokens, self.attribute_ctx):
            idx = int(name.rsplit("_", 1)[-1])
            ctx_blocks = self._expand_param(ctx_param, indices)
            if ctx_blocks.numel() > 0:
                pieces.append(ctx_blocks)
            attr_tokens = getattr(self, name)[indices]
            pieces.append(attr_tokens)

        if self.n_ctx_main > 0:
            pieces.append(self._expand_param(self.ctx_main, indices))

        pieces.append(self.token_suffix[indices])
        return torch.cat(pieces, dim=1)

    def get_tokenized_prompts(self, class_ids=None):
        if class_ids is None:
            indices = torch.arange(self.n_cls, device=self.tokenized_prompts.device)
        else:
            indices = class_ids.to(self.tokenized_prompts.device, dtype=torch.long)
        return self.tokenized_prompts[indices]


class TemplateAttributePromptLearner(nn.Module):
    """Prompt learner that keeps template tokens trainable while freezing class and attribute tokens."""

    def __init__(self, classnames, prompt_strings, clip_model, template):
        super().__init__()

        if not prompt_strings or len(prompt_strings) != len(classnames):
            raise ValueError("prompt_strings must provide one prompt per class")

        if template.count("{}") != 1:
            raise ValueError("TemplateAttributePromptLearner currently supports templates with exactly one '{}' placeholder")

        self._tokenizer = _Tokenizer()
        self.classnames = [name.replace("_", " ") for name in classnames]
        self.prompts = list(prompt_strings)
        self.n_cls = len(self.classnames)

        token_embedding = clip_model.token_embedding
        dtype = token_embedding.weight.dtype

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in self.prompts])
        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("tokenized_prompts", tokenized_prompts)
        self.register_buffer("base_embedding", embedding)

        template_prompts = [template.format(name) for name in self.classnames]
        template_tokenized = torch.cat([clip.tokenize(p) for p in template_prompts])

        class_token_lists = [self._tokenizer.encode(name) for name in self.classnames]
        eos_token = 49407  # CLIP end-of-text id

        template_tokens_ref = template_tokenized[0].tolist()
        template_class_pos = _find_subsequence(template_tokens_ref, class_token_lists[0])
        if not template_class_pos:
            raise ValueError("Failed to locate class tokens inside template prompt")

        prefix_start = 1
        prefix_end = template_class_pos[0]
        prefix_len = max(0, prefix_end - prefix_start)

        suffix_start_template = template_class_pos[-1] + 1
        try:
            eos_index = template_tokens_ref.index(eos_token)
        except ValueError as exc:
            raise ValueError("Template prompt is missing EOS token") from exc
        suffix_len = max(0, eos_index - suffix_start_template)

        # Verify consistency across classes for template pieces
        for idx in range(1, self.n_cls):
            tokens_template = template_tokenized[idx].tolist()
            class_pos = _find_subsequence(tokens_template, class_token_lists[idx])
            if not class_pos:
                raise ValueError(f"Failed to locate class tokens in template for class index {idx}")
            if class_pos[0] != prefix_end:
                raise ValueError("Inconsistent template prefix length across classes")
            other_suffix_start = class_pos[-1] + 1
            if suffix_len != max(0, tokens_template.index(eos_token) - other_suffix_start):
                raise ValueError("Inconsistent template suffix length across classes")

        self.prefix_slice = slice(prefix_start, prefix_end)
        self.suffix_len = suffix_len

        if prefix_len > 0:
            prefix_init = embedding[0, self.prefix_slice, :].clone()
            self.template_prefix = nn.Parameter(prefix_init)
        else:
            self.register_parameter("template_prefix", None)

        if suffix_len > 0:
            suffix_tokens = template_tokens_ref[suffix_start_template:suffix_start_template + suffix_len]
            suffix_init = embedding[0, suffix_start_template:suffix_start_template + suffix_len, :].clone()
            self.template_suffix = nn.Parameter(suffix_init)
            self.register_buffer("template_suffix_tokens", torch.tensor(suffix_tokens, dtype=torch.long))
        else:
            self.register_parameter("template_suffix", None)
            self.register_buffer("template_suffix_tokens", torch.empty(0, dtype=torch.long))

        suffix_starts = []
        for cls_idx in range(self.n_cls):
            tokens = tokenized_prompts[cls_idx].tolist()
            class_positions = _find_subsequence(tokens, class_token_lists[cls_idx])
            if not class_positions:
                raise ValueError(f"Failed to locate class tokens in full prompt for class index {cls_idx}")
            if prefix_len > 0:
                prompt_prefix = tokens[self.prefix_slice.start:self.prefix_slice.stop]
                template_prefix = template_tokens_ref[self.prefix_slice.start:self.prefix_slice.stop]
                if prompt_prefix != template_prefix:
                    raise ValueError("Prompt prefix tokens differ from template prefix tokens")
            suffix_start = class_positions[-1] + 1
            if self.suffix_len > 0:
                prompt_suffix = tokens[suffix_start:suffix_start + self.suffix_len]
                if prompt_suffix != template_tokens_ref[suffix_start_template:suffix_start_template + self.suffix_len]:
                    raise ValueError("Prompt suffix tokens differ from template suffix tokens")
            suffix_starts.append(suffix_start)

        self.register_buffer("suffix_starts", torch.tensor(suffix_starts, dtype=torch.long))

    def forward(self, class_ids=None):
        if class_ids is None:
            indices = torch.arange(self.n_cls, device=self.base_embedding.device)
        else:
            indices = class_ids.to(self.base_embedding.device, dtype=torch.long)

        prompts = self.base_embedding[indices].clone()

        if self.template_prefix is not None:
            prefix = self.template_prefix.unsqueeze(0).expand(prompts.size(0), -1, -1)
            prompts[:, self.prefix_slice, :] = prefix

        if self.template_suffix is not None and self.suffix_len > 0:
            suffix = self.template_suffix
            for batch_idx, cls_idx in enumerate(indices.tolist()):
                start = int(self.suffix_starts[cls_idx].item())
                end = start + self.suffix_len
                prompts[batch_idx, start:end, :] = suffix

        return prompts

    def get_tokenized_prompts(self, class_ids=None):
        if class_ids is None:
            indices = torch.arange(self.n_cls, device=self.tokenized_prompts.device)
        else:
            indices = class_ids.to(self.tokenized_prompts.device, dtype=torch.long)
        return self.tokenized_prompts[indices]

class TextPromptLearner(nn.Module):
    def __init__(self, classnames, token_embedding, num_prompts, CSC=False, ctx_pos='end'):
        super().__init__()

        _tokenizer = _Tokenizer()
        n_cls = len(classnames)
        n_ctx = num_prompts
        ctx_dim = 768


        if CSC:
            print("Initializing class-specific contexts")
            ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim)
        else:
            print("Initializing a generic context")
            ctx_vectors = torch.empty(n_ctx, ctx_dim)
        nn.init.normal_(ctx_vectors, std=0.02)
        prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        # 注释了一些保存模型的东西，会减慢速度，但是不太好修改，无奈之举
        # print(tokenized_prompts.shape)
        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = ctx_pos

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


class TextPromptLearnerOnly(nn.Module):
    def __init__(self, classname, token_embedding, num_prompts, CSC=False, ctx_pos='end'):
        super().__init__()

        _tokenizer = _Tokenizer()
        n_cls = 1
        n_ctx = num_prompts
        ctx_dim = 768


        if CSC:
            print("Initializing class-specific contexts")
            ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim)
        else:
            print("Initializing a generic context")
            ctx_vectors = torch.empty(n_ctx, ctx_dim)
        nn.init.normal_(ctx_vectors, std=0.02)
        prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        # 注释了一些保存模型的东西，会减慢速度，但是不太好修改，无奈之举
        # print(tokenized_prompts.shape)
        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = ctx_pos

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


class VerbObjectPromptLearner(nn.Module):
    def __init__(
        self,
        classnames,
        clip_model,
        template="A video of people {}ing {}.",
        verb_init="doing",
        object_init="something",
        class_specific=True,
    ):
        super().__init__()

        self._tokenizer = _Tokenizer()
        self.class_specific = class_specific

        n_cls = len(classnames)
        token_embedding = clip_model.token_embedding
        device = token_embedding.weight.device
        dtype = token_embedding.weight.dtype

        base_prompt = template.format(verb_init, object_init)
        tokenized_single = clip.tokenize(base_prompt)
        tokenized = tokenized_single.repeat(n_cls, 1)

        with torch.no_grad():
            embedding = token_embedding(tokenized).detach()

        if "{}ing" in template:
            if verb_init.endswith("ing"):
                verb_search = verb_init
            else:
                verb_search = verb_init + "ing"
        else:
            verb_search = verb_init

        verb_tokens = self._tokenizer.encode(verb_search)
        object_tokens = self._tokenizer.encode(object_init)

        token_list = tokenized_single[0].tolist()
        verb_indices = _find_subsequence(token_list, verb_tokens)
        object_indices = _find_subsequence(token_list, object_tokens)

        if not verb_indices or not object_indices:
            raise ValueError(
                f"Failed to locate verb/object placeholders in template '{base_prompt}'."
            )

        self.register_buffer("tokenized_prompts", tokenized)
        self.register_buffer("template_embedding", embedding)
        self.register_buffer("verb_indices", torch.tensor(verb_indices, dtype=torch.long))
        self.register_buffer("object_indices", torch.tensor(object_indices, dtype=torch.long))

        if class_specific:
            verb_embed_init = embedding[:, verb_indices, :].clone()
            object_embed_init = embedding[:, object_indices, :].clone()
        else:
            verb_embed_init = embedding[0, verb_indices, :].clone()
            object_embed_init = embedding[0, object_indices, :].clone()

        self.verb_embeddings = nn.Parameter(verb_embed_init.to(dtype=dtype, device=device))
        self.object_embeddings = nn.Parameter(object_embed_init.to(dtype=dtype, device=device))

    def forward(self, class_ids):
        if isinstance(class_ids, torch.Tensor):
            indices = class_ids.to(self.tokenized_prompts.device, dtype=torch.long)
        else:
            indices = torch.as_tensor(class_ids, device=self.tokenized_prompts.device, dtype=torch.long)

        prompts = self.template_embedding[indices].clone()

        if self.class_specific:
            prompts[:, self.verb_indices, :] = self.verb_embeddings[indices]
            prompts[:, self.object_indices, :] = self.object_embeddings[indices]
        else:
            prompts[:, self.verb_indices, :] = self.verb_embeddings.unsqueeze(0)
            prompts[:, self.object_indices, :] = self.object_embeddings.unsqueeze(0)

        tokenized = self.tokenized_prompts[indices]
        return prompts, tokenized

    def get_all_prompts(self):
        prompts = self.template_embedding.clone()
        if self.class_specific:
            prompts[:, self.verb_indices, :] = self.verb_embeddings
            prompts[:, self.object_indices, :] = self.object_embeddings
        else:
            prompts[:, self.verb_indices, :] = self.verb_embeddings.unsqueeze(0)
            prompts[:, self.object_indices, :] = self.object_embeddings.unsqueeze(0)
        return prompts, self.tokenized_prompts

