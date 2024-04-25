from statistics import mean
from typing import Dict, List
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.gleu_score import sentence_gleu
from nltk.translate.meteor_score import meteor_score


class BLUE:
    def __init__(self, ngrams: int = 4) -> None:
        self.smoothing = SmoothingFunction().method3

        self.n = ngrams
        weights = [1 / ngrams if i <= ngrams else 0 for i in range(1, 5)]
        self.weights = tuple(weights)

    def __call__(self, references, hypothesis) -> float:
        score = sentence_bleu(references,
                              hypothesis,
                              weights=self.weights,
                              smoothing_function=self.smoothing)
        return score

    def __repr__(self) -> str:
        return f"bleu{self.n}"


class GLEU:
    def __init__(self) -> None:
        pass

    def __call__(self, *args, **kwargs):
        return sentence_gleu(*args, **kwargs)

    def __repr__(self):
        return "gleu"