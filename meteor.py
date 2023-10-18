from nltk.translate import meteor_score
import numpy as np

def mscore(predictions, references):
    scores = [
                meteor_score.single_meteor_score(
                    ref.split(), pred.split()
                )
                for ref, pred in zip(references, predictions)
            ]
    
    return np.mean(scores)*100.
    