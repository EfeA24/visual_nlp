"""Turn character logits into text. Scores are not accuracy percentages."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from visual_nlp.reader.model import ReaderModel
from visual_nlp.reader.tokenizer import CharacterTokenizer


@dataclass
class DecodeResult:
    texts: list[str]
    token_ids: list[list[int]]
    scores: list[list[float]]
    completed: list[bool]
    truncated: list[bool] = field(default_factory=list)


def greedy_decode(
    model: ReaderModel,
    images: torch.Tensor,
    tokenizer: CharacterTokenizer,
    max_length: int,
    valid_hw: torch.Tensor | None = None,
    use_cache: bool = True,
) -> DecodeResult:
    model.eval()
    with torch.no_grad():
        memory, memory_mask = model.encode(images, valid_hw=valid_hw)
        batch = images.size(0)
        device = images.device
        generated = torch.full((batch, 1), tokenizer.bos_id, dtype=torch.long, device=device)
        token_scores = [[] for _ in range(batch)]
        finished = torch.zeros(batch, dtype=torch.bool, device=device)
        cache = None
        for _ in range(max_length):
            if use_cache:
                step_ids = generated if cache is None else generated[:, -1:]
                start_pos = 0 if cache is None else generated.size(1) - 1
                logits, cache = model.character_decoder(
                    step_ids,
                    memory,
                    tgt_padding_mask=step_ids.eq(tokenizer.pad_id),
                    memory_padding_mask=memory_mask,
                    start_pos=start_pos,
                    cache=cache,
                    use_cache=True,
                )
            else:
                logits = model.character_decoder(
                    generated,
                    memory,
                    tgt_padding_mask=generated.eq(tokenizer.pad_id),
                    memory_padding_mask=memory_mask,
                )
            step_logits = logits[:, -1]
            probs = F.softmax(step_logits, dim=-1)
            next_tokens = step_logits.argmax(dim=-1)
            next_scores = probs.gather(1, next_tokens.unsqueeze(1)).squeeze(1)
            for idx in range(batch):
                if not finished[idx]:
                    token_scores[idx].append(float(next_scores[idx].item()))
            generated = torch.cat([generated, next_tokens.unsqueeze(1)], dim=1)
            finished = finished | next_tokens.eq(tokenizer.eos_id)
            if bool(finished.all()):
                break
        token_lists = generated[:, 1:].tolist()
        texts = [tokenizer.decode(ids) for ids in token_lists]
        completed = []
        truncated = []
        cleaned: list[list[int]] = []
        for ids, done in zip(token_lists, finished.tolist()):
            if tokenizer.eos_id in ids:
                ids = ids[: ids.index(tokenizer.eos_id)]
                completed.append(True)
                truncated.append(False)
            else:
                completed.append(False)
                truncated.append(True)
            cleaned.append(ids)
        return DecodeResult(
            texts=texts,
            token_ids=cleaned,
            scores=token_scores,
            completed=completed,
            truncated=truncated,
        )


def beam_search_decode(
    model: ReaderModel,
    images: torch.Tensor,
    tokenizer: CharacterTokenizer,
    max_length: int,
    beam_size: int = 5,
    valid_hw: torch.Tensor | None = None,
) -> DecodeResult:
    if beam_size <= 1:
        return greedy_decode(model, images, tokenizer, max_length, valid_hw=valid_hw)
    model.eval()
    results = DecodeResult(texts=[], token_ids=[], scores=[], completed=[], truncated=[])
    with torch.no_grad():
        memory, memory_mask = model.encode(images, valid_hw=valid_hw)
        for batch_idx in range(images.size(0)):
            sample_mem = memory[batch_idx : batch_idx + 1]
            sample_mask = memory_mask[batch_idx : batch_idx + 1]
            beams = [(0.0, [tokenizer.bos_id], [], False)]
            for _ in range(max_length):
                candidates = []
                active = [beam for beam in beams if not beam[3]]
                if not active:
                    break
                for score, tokens, token_scores, _done in active:
                    ids = torch.tensor([tokens], device=images.device)
                    logits = model.character_decoder(
                        ids,
                        sample_mem,
                        tgt_padding_mask=ids.eq(tokenizer.pad_id),
                        memory_padding_mask=sample_mask,
                    )
                    log_probs = F.log_softmax(logits[:, -1], dim=-1).squeeze(0)
                    top_log, top_idx = torch.topk(log_probs, beam_size)
                    for log_p, token in zip(top_log.tolist(), top_idx.tolist()):
                        done = token == tokenizer.eos_id
                        candidates.append(
                            (score + log_p, tokens + [token], token_scores + [float(log_p)], done)
                        )
                beams = sorted(candidates + [beam for beam in beams if beam[3]], key=lambda item: item[0], reverse=True)[
                    :beam_size
                ]
                if all(beam[3] for beam in beams):
                    break
            best = max(beams, key=lambda item: item[0] / max(len(item[1]) - 1, 1))
            ids = best[1][1:]
            completed = tokenizer.eos_id in ids or best[3]
            if tokenizer.eos_id in ids:
                ids = ids[: ids.index(tokenizer.eos_id)]
            results.texts.append(tokenizer.decode(best[1]))
            results.token_ids.append(ids)
            results.scores.append(best[2])
            results.completed.append(completed)
            results.truncated.append(not completed)
    return results


def decode_images(
    model: ReaderModel,
    images: torch.Tensor,
    tokenizer: CharacterTokenizer,
    strategy: str = "greedy",
    max_length: int = 128,
    beam_size: int = 5,
    valid_hw: torch.Tensor | None = None,
) -> DecodeResult:
    if strategy == "beam":
        return beam_search_decode(model, images, tokenizer, max_length, beam_size, valid_hw)
    return greedy_decode(model, images, tokenizer, max_length, valid_hw)
