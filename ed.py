from nltk.metrics.distance import edit_distance
import numpy as np

def edscore(predictions, references):
    scores = [
                edit_distance(
                    ref, pred
                )
                for ref, pred in zip(references, predictions)
            ]
    
    return np.mean(scores)