import math
import torch

from backend.text_processing.classic_engine import CLIPEmbeddingForTextualInversion, PromptChunk, PromptChunkFix
from backend.text_processing.textual_inversion import EmbeddingDatabase
from backend import memory_management

from scripts import emphasis, parsing 

last_extra_generation_params = {}

class ClassicTextProcessingEngineTopKEmphasis:
    def __init__(
            self, text_encoder, tokenizer, chunk_length=75,
            embedding_dir=None, embedding_key='clip_l', embedding_expected_shape=768, emphasis_name="Original",
            text_projection=False, minimal_clip_skip=1, clip_skip=1, return_pooled=False, final_layer_norm=True
    ):
        super().__init__()

        self.embeddings = EmbeddingDatabase(tokenizer, embedding_expected_shape)

        if isinstance(embedding_dir, str):
            self.embeddings.add_embedding_dir(embedding_dir)
            self.embeddings.load_textual_inversion_embeddings()

        self.embedding_key = embedding_key

        self.text_encoder = text_encoder
        self.tokenizer = tokenizer

        self.emphasis = emphasis.TopKEmphasis()
        self.text_projection = text_projection
        self.minimal_clip_skip = minimal_clip_skip
        self.clip_skip = clip_skip
        self.return_pooled = return_pooled
        self.final_layer_norm = final_layer_norm

        self.chunk_length = chunk_length

        self.id_start = self.tokenizer.bos_token_id
        self.id_end = self.tokenizer.eos_token_id
        self.id_pad = self.tokenizer.pad_token_id

        model_embeddings = text_encoder.transformer.text_model.embeddings
        model_embeddings.token_embedding = CLIPEmbeddingForTextualInversion(model_embeddings.token_embedding, self.embeddings, textual_inversion_key=embedding_key)

        vocab = self.tokenizer.get_vocab()

        self.comma_token = vocab.get(',</w>', None)

        self.token_mults = {}

        tokens_with_parens = [(k, v) for k, v in vocab.items() if '(' in k or ')' in k or '[' in k or ']' in k]
        for text, ident in tokens_with_parens:
            mult = 1.0
            for c in text:
                if c == '[':
                    mult /= 1.1
                if c == ']':
                    mult *= 1.1
                if c == '(':
                    mult *= 1.1
                if c == ')':
                    mult /= 1.1

            if mult != 1.0:
                self.token_mults[ident] = mult

    def empty_chunk(self):
        chunk = PromptChunk()
        chunk.tokens = [self.id_start] + [self.id_end] * (self.chunk_length + 1)
        chunk.multipliers = [parsing.EmphasisPair()] * (self.chunk_length + 2)
        return chunk

    def get_target_prompt_token_count(self, token_count):
        return math.ceil(max(token_count, 1) / self.chunk_length) * self.chunk_length

    def tokenize(self, texts):
        tokenized = self.tokenizer(texts, truncation=False, add_special_tokens=False)["input_ids"]

        return tokenized

    def encode_with_transformers(self, tokens):
        target_device = memory_management.text_encoder_device()

        self.text_encoder.transformer.text_model.embeddings.position_ids = self.text_encoder.transformer.text_model.embeddings.position_ids.to(device=target_device)
        self.text_encoder.transformer.text_model.embeddings.position_embedding = self.text_encoder.transformer.text_model.embeddings.position_embedding.to(dtype=torch.float32)
        self.text_encoder.transformer.text_model.embeddings.token_embedding = self.text_encoder.transformer.text_model.embeddings.token_embedding.to(dtype=torch.float32)

        tokens = tokens.to(target_device)

        outputs = self.text_encoder.transformer(tokens, output_hidden_states=True)

        layer_id = - max(self.clip_skip, self.minimal_clip_skip)
        z = outputs.hidden_states[layer_id]

        if self.final_layer_norm:
            z = self.text_encoder.transformer.text_model.final_layer_norm(z)

        if self.return_pooled:
            pooled_output = outputs.pooler_output

            if self.text_projection:
                pooled_output = self.text_encoder.transformer.text_projection(pooled_output)

            z.pooled = pooled_output
        return z

    def tokenize_line(self, line):
        parsed = parsing.parse_prompt_attention(line)

        if len(parsed) == 0:
            parsed = [parsing.EmphasisPair()]
        tokenized = self.tokenize([pair.text for pair in parsed])

        chunks = []
        chunk = PromptChunk()
        token_count = 0
        last_comma = -1

        def next_chunk(is_last=False):
            nonlocal token_count
            nonlocal last_comma
            nonlocal chunk

            if is_last:
                token_count += len(chunk.tokens)
            else:
                token_count += self.chunk_length

            to_add = self.chunk_length - len(chunk.tokens)
            if to_add > 0:
                chunk.tokens += [self.id_end] * to_add
                chunk.multipliers += [parsing.EmphasisPair()] * to_add

            chunk.tokens = [self.id_start] + chunk.tokens + [self.id_end]
            chunk.multipliers = [parsing.EmphasisPair()] + chunk.multipliers + [parsing.EmphasisPair()]

            last_comma = -1
            chunks.append(chunk)
            chunk = PromptChunk()

        for tokens, weight in zip(tokenized, parsed):
            if weight.text == 'BREAK':
                next_chunk()
                continue

            position = 0
            while position < len(tokens):
                token = tokens[position]

                comma_padding_backtrack = 20

                if token == self.comma_token:
                    last_comma = len(chunk.tokens)

                elif comma_padding_backtrack != 0 and len(chunk.tokens) == self.chunk_length and last_comma != -1 and len(chunk.tokens) - last_comma <= comma_padding_backtrack:
                    break_location = last_comma + 1

                    reloc_tokens = chunk.tokens[break_location:]
                    reloc_mults = chunk.multipliers[break_location:]

                    chunk.tokens = chunk.tokens[:break_location]
                    chunk.multipliers = chunk.multipliers[:break_location]

                    next_chunk()
                    chunk.tokens = reloc_tokens
                    chunk.multipliers = reloc_mults

                if len(chunk.tokens) == self.chunk_length:
                    next_chunk()

                embedding, embedding_length_in_tokens = self.embeddings.find_embedding_at_position(tokens, position)
                if embedding is None:
                    chunk.tokens.append(token)
                    chunk.multipliers.append(weight)
                    position += 1
                    continue

                emb_len = int(embedding.vectors)
                if len(chunk.tokens) + emb_len > self.chunk_length:
                    next_chunk()

                chunk.fixes.append(PromptChunkFix(len(chunk.tokens), embedding))

                chunk.tokens += [0] * emb_len
                chunk.multipliers += [weight] * emb_len
                position += embedding_length_in_tokens

        if chunk.tokens or not chunks:
            next_chunk(is_last=True)

        return chunks, token_count

    def process_texts(self, texts):
        token_count = 0

        cache = {}
        batch_chunks = []
        for line in texts:
            if line in cache:
                chunks = cache[line]
            else:
                chunks, current_token_count = self.tokenize_line(line)
                token_count = max(current_token_count, token_count)

                cache[line] = chunks

            batch_chunks.append(chunks)

        return batch_chunks, token_count

    def __call__(self, texts):
        batch_chunks, token_count = self.process_texts(texts)

        used_embeddings = {}
        chunk_count = max([len(x) for x in batch_chunks])

        zs = []
        for i in range(chunk_count):
            batch_chunk = [chunks[i] if i < len(chunks) else self.empty_chunk() for chunks in batch_chunks]

            tokens = [x.tokens for x in batch_chunk]
            multipliers = [x.multipliers for x in batch_chunk]
            self.embeddings.fixes = [x.fixes for x in batch_chunk]

            for fixes in self.embeddings.fixes:
                for _position, embedding in fixes:
                    used_embeddings[embedding.name] = embedding

            z = self.process_tokens(tokens, multipliers)
            zs.append(z)

        global last_extra_generation_params

        if used_embeddings:
            names = []

            for name, embedding in used_embeddings.items():
                print(f'[Textual Inversion] Used Embedding [{name}] in CLIP of [{self.embedding_key}]')
                names.append(name.replace(":", "").replace(",", ""))

            if "TI" in last_extra_generation_params:
                last_extra_generation_params["TI"] += ", " + ", ".join(names)
            else:
                last_extra_generation_params["TI"] = ", ".join(names)

        if any(x for x in texts if "(" in x or "[" in x) and self.emphasis.name != "Original":
            last_extra_generation_params["Emphasis"] = self.emphasis.name

        if self.return_pooled:
            return torch.hstack(zs), zs[0].pooled
        else:
            return torch.hstack(zs)

    def process_tokens(self, remade_batch_tokens, batch_multipliers):
        tokens = torch.asarray(remade_batch_tokens)

        if self.id_end != self.id_pad:
            for batch_pos in range(len(remade_batch_tokens)):
                index = remade_batch_tokens[batch_pos].index(self.id_end)
                tokens[batch_pos, index + 1:tokens.shape[1]] = self.id_pad

        z = self.encode_with_transformers(tokens)

        pooled = getattr(z, 'pooled', None)

        self.emphasis.tokens = remade_batch_tokens
        self.emphasis.multipliers = batch_multipliers
        self.emphasis.z = z
        self.emphasis.after_transformers()
        z = self.emphasis.z

        if pooled is not None:
            z.pooled = pooled

        return z