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

