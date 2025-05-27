from lm_eval.api.registry import register_metric
from sentence_transformers import SentenceTransformer, util
from rouge_score import rouge_scorer


@register_metric(
    metric='cosine_sim',
    higher_is_better=True,
    aggregation='mean'
)
def cosine_sim(preds, targets):
    model = SentenceTransformer("nasa-impact/nasa-smd-ibm-st-v2")
    query = model.encode(preds)
    targets = model.encode(targets)
    return util.cos_sim(query, targets).item()


@register_metric(
    metric='rougeL',
    higher_is_better=True,
    aggregation='mean'
)
def rougeL(preds, targets):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(preds, targets)
    return scores['rougeL'].fmeasure

@register_metric(
    metric='rouge1',
    higher_is_better=True,
    aggregation='mean'
)
def rouge1(preds, targets):
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    scores = scorer.score(preds, targets)
    return scores['rouge1'].fmeasure

