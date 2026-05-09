"""Machine Translation evaluation utilities."""

from __future__ import annotations

import logging
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any
import torch

LOGGER = logging.getLogger(__name__)
_LABEL_KEYS = ("labels", "target_ids", "targets", "y")
_REFERENCE_KEYS = ("references", "refs")
_NON_MODEL_INPUT_KEYS = set(_LABEL_KEYS) | set(_REFERENCE_KEYS)

_METRIC_IMPORT_ERROR = (
    "Missing evaluation dependencies. Install with: "
    "pip install sacrebleu rouge-score wandb python-dotenv"
)


def normalize_text(text: str) -> str:
    """Normalize text safely for multilingual MT metrics."""
    if text is None:
        return ""
    try:
        normalized = unicodedata.normalize("NFKC", str(text))
    except Exception:
        normalized = str(text)
    return " ".join(normalized.strip().split())


def _to_list(data: Any) -> list[Any]:
    if data is None:
        return []
    if isinstance(data, str):
        return [data]
    if isinstance(data, Sequence):
        return list(data)
    return [data]


def _prepare_inputs(
    predictions: Any, references: Any
) -> tuple[list[str], list[list[str]], list[str]]:
    pred_list = _to_list(predictions)
    ref_list = _to_list(references)

    if not pred_list or not ref_list:
        return [], [[]], []

    n = min(len(pred_list), len(ref_list))
    if len(pred_list) != len(ref_list):
        LOGGER.warning(
            "Predictions/references length mismatch (%s vs %s). Truncating to %s.",
            len(pred_list),
            len(ref_list),
            n,
        )

    cleaned_predictions: list[str] = []
    sample_refs: list[list[str]] = []

    for pred_item, ref_item in zip(pred_list[:n], ref_list[:n]):
        pred_text = normalize_text(str(pred_item) if pred_item is not None else "")
        cleaned_predictions.append(pred_text)

        if isinstance(ref_item, str) or ref_item is None:
            refs = [normalize_text(str(ref_item) if ref_item is not None else "")]
        elif isinstance(ref_item, Sequence):
            refs = [normalize_text(str(r) if r is not None else "") for r in ref_item]
            refs = refs or [""]
        else:
            refs = [normalize_text(str(ref_item))]
        sample_refs.append(refs)

    max_refs = max(len(x) for x in sample_refs) if sample_refs else 1
    refs_for_sacrebleu: list[list[str]] = [[] for _ in range(max_refs)]
    primary_refs: list[str] = []

    for refs in sample_refs:
        primary = refs[0] if refs else ""
        primary_refs.append(primary)
        for idx in range(max_refs):
            refs_for_sacrebleu[idx].append(refs[idx] if idx < len(refs) else primary)

    return cleaned_predictions, refs_for_sacrebleu, primary_refs


def compute_bleu(predictions: Any, references: Any) -> float:
    """Compute corpus BLEU using sacrebleu."""
    try:
        import sacrebleu
    except ImportError as exc:
        raise ImportError(_METRIC_IMPORT_ERROR) from exc

    preds, refs_for_bleu, _ = _prepare_inputs(predictions, references)
    if not preds:
        return 0.0
    try:
        return float(sacrebleu.corpus_bleu(preds, refs_for_bleu).score)
    except Exception:
        LOGGER.exception("BLEU computation failed; returning 0.0.")
        return 0.0


def compute_chrf(predictions: Any, references: Any) -> float:
    """Compute corpus chrF using sacrebleu."""
    try:
        import sacrebleu
    except ImportError as exc:
        raise ImportError(_METRIC_IMPORT_ERROR) from exc

    preds, refs_for_chrf, _ = _prepare_inputs(predictions, references)
    if not preds:
        return 0.0
    try:
        return float(sacrebleu.corpus_chrf(preds, refs_for_chrf).score)
    except Exception:
        LOGGER.exception("chrF computation failed; returning 0.0.")
        return 0.0


def compute_rouge(predictions: Any, references: Any) -> dict[str, float]:
    """Compute average ROUGE-1/2/L F1 scores (scaled 0-100)."""
    try:
        from rouge_score import rouge_scorer
    except ImportError as exc:
        raise ImportError(_METRIC_IMPORT_ERROR) from exc

    preds, _, primary_refs = _prepare_inputs(predictions, references)
    if not preds:
        return {"eval/rouge1": 0.0, "eval/rouge2": 0.0, "eval/rougeL": 0.0}

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    count = 0

    for pred, ref in zip(preds, primary_refs):
        try:
            score = scorer.score(ref, pred)
            totals["rouge1"] += float(score["rouge1"].fmeasure)
            totals["rouge2"] += float(score["rouge2"].fmeasure)
            totals["rougeL"] += float(score["rougeL"].fmeasure)
            count += 1
        except Exception:
            LOGGER.warning("Skipping malformed sample during ROUGE computation.")

    if count == 0:
        return {"eval/rouge1": 0.0, "eval/rouge2": 0.0, "eval/rougeL": 0.0}

    return {
        "eval/rouge1": 100.0 * totals["rouge1"] / count,
        "eval/rouge2": 100.0 * totals["rouge2"] / count,
        "eval/rougeL": 100.0 * totals["rougeL"] / count,
    }


def compute_mt_metrics(predictions: Any, references: Any) -> dict[str, float]:
    """Compute BLEU, chrF, and ROUGE metrics with stable flat keys."""
    bleu = compute_bleu(predictions, references)
    chrf = compute_chrf(predictions, references)
    rouge = compute_rouge(predictions, references)
    return {"eval/bleu": bleu, "eval/chrf": chrf, **rouge}


def _move_to_device(batch: Any, device: torch.device) -> Any:
    if torch.is_tensor(batch):
        return batch.to(device)
    if isinstance(batch, Mapping):
        return {k: _move_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, tuple):
        return tuple(_move_to_device(x, device) for x in batch)
    if isinstance(batch, list):
        return [_move_to_device(x, device) for x in batch]
    return batch


def _decode_token_batch(tokenizer: Any, data: Any, ignore_index: int = 0) -> list[str]:
    if isinstance(data, list) and data and isinstance(data[0], str):
        return [normalize_text(x) for x in data]

    if torch.is_tensor(data):
        sequences = data.detach().cpu().tolist()
    elif isinstance(data, Sequence):
        sequences = list(data)
    else:
        return [normalize_text(str(data))]

    if tokenizer is not None and hasattr(tokenizer, "batch_decode"):
        cleaned: list[list[int]] = []
        pad_id = getattr(tokenizer, "pad_token_id", 0)
        for seq in sequences:
            if not isinstance(seq, Sequence):
                seq = [seq]
            cleaned.append([int(tok) if int(tok) != ignore_index else int(pad_id) for tok in seq])
        return [normalize_text(x) for x in tokenizer.batch_decode(cleaned, skip_special_tokens=True)]

    output: list[str] = []
    for seq in sequences:
        if isinstance(seq, Sequence):
            tokens = [str(int(x)) for x in seq if int(x) != ignore_index]
            output.append(" ".join(tokens))
        else:
            output.append(str(seq))
    return [normalize_text(x) for x in output]


def _extract_mapping_targets(batch: Mapping[str, Any]) -> tuple[torch.Tensor | None, Any]:
    labels = None
    for key in _LABEL_KEYS:
        value = batch.get(key)
        if torch.is_tensor(value):
            labels = value
            break

    references = None
    for key in _REFERENCE_KEYS:
        if key in batch:
            references = batch[key]
            break
    return labels, references


def _labels_for_references(labels: Any) -> Any:
    if torch.is_tensor(labels) and labels.ndim >= 2 and labels.size(1) > 1:
        return labels[:, 1:]
    return labels


def evaluate_model(
    model: torch.nn.Module,
    eval_dataloader: Any,
    tokenizer: Any | None = None,
    device: torch.device | str | None = None,
    generation_kwargs: dict[str, Any] | None = None,
    ignore_index: int = 0,
) -> dict[str, float]:
    """Run model evaluation and return MT metrics."""
    model_device = next(model.parameters()).device
    active_device = torch.device(device) if device is not None else model_device
    generation_kwargs = generation_kwargs or {}

    predictions: list[str] = []
    references: list[Any] = []
    model.eval()

    with torch.no_grad():
        for batch in eval_dataloader:
            # Di chuyển batch lên device
            batch_on_device = _move_to_device(batch, active_device)
            pred_tokens = None

            if isinstance(batch_on_device, Mapping):
                labels, references_data = _extract_mapping_targets(batch_on_device)
                model_inputs = {k: v for k, v in batch_on_device.items() if k not in _NON_MODEL_INPUT_KEYS}
                
                if hasattr(model, "generate"):
                    try:
                        pred_tokens = model.generate(**model_inputs, **generation_kwargs)
                    except Exception:
                        outputs = model(**model_inputs)
                        logits = outputs.logits if hasattr(outputs, "logits") else outputs["logits"]
                        pred_tokens = logits.argmax(dim=-1)
                else:
                    outputs = model(**model_inputs)
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs["logits"]
                    pred_tokens = logits.argmax(dim=-1)

                predictions.extend(_decode_token_batch(tokenizer, pred_tokens, ignore_index))
                
                if labels is not None:
                    references.extend(_decode_token_batch(tokenizer, labels, ignore_index))
                elif references_data is not None:
                    references.extend(_to_list(references_data))
                else:
                    references.extend([""] * len(pred_tokens))

            elif isinstance(batch_on_device, (tuple, list)) and len(batch_on_device) >= 2:
                inputs, labels = batch_on_device[0], batch_on_device[1]

                if hasattr(model, "greedy_decode") and torch.is_tensor(inputs):
                    bos_token_id = generation_kwargs.get("bos_token_id", 1)
                    max_length = generation_kwargs.get("max_length", labels.size(1))
                    outputs = model.greedy_decode(src=inputs, bos_token_id=bos_token_id, max_length=max_length)
                    pred_tokens = outputs[0].argmax(dim=-1) if isinstance(outputs, (tuple, list)) else outputs
                else:
                    try:
                        outputs = model(*batch_on_device)
                    except TypeError:
                        outputs = model(inputs, labels)
                    logits = outputs[0] if isinstance(outputs, (tuple, list)) else getattr(outputs, "logits", outputs)
                    pred_tokens = logits.argmax(dim=-1)

                predictions.extend(_decode_token_batch(tokenizer, pred_tokens, ignore_index))
                references.extend(_decode_token_batch(tokenizer, _labels_for_references(labels), ignore_index))
            
            else:
                raise ValueError("Unsupported batch format.")

            # DỌN DẸP CUỐI VÒNG LẶP (An toàn & Sạch sẽ)
            if 'outputs' in locals(): del outputs
            del batch_on_device, pred_tokens

    # DỌN CACHE MỘT LẦN DUY NHẤT SAU KHI KẾT THÚC TOÀN BỘ EVAL
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return compute_mt_metrics(predictions, references)

