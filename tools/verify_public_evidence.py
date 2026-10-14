"""Verify saved public evidence without images, checkpoints, or retraining."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ('angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise')

def verify():
    selection = json.loads((ROOT/'outputs/selected_candidate.json').read_text())
    selected = selection['selected']
    ranking = selection['ranking']
    assert ranking == sorted(ranking, key=lambda r: (r['validation_accuracy'], r['validation_macro_f1']), reverse=True)
    assert selected == ranking[0]
    frame = pd.read_csv(ROOT/'outputs/analysis'/selected['candidate_id']/'validation_predictions.csv')
    audit = json.loads((ROOT/'outputs/eda/audit_summary.json').read_text())
    assert audit['split']['exact_hash_overlap'] == 0
    assert len(frame) == audit['split']['validation_rows'] == 5741
    assert frame.id.is_unique
    probabilities = frame[[f'prob_{label}' for label in CLASSES]].to_numpy(dtype=float)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    assert np.allclose(probabilities.sum(axis=1), 1, atol=2e-6)
    truth = frame.true_label.map({name: index for index, name in enumerate(CLASSES)}).to_numpy()
    assert not pd.isna(truth).any()
    prediction = probabilities.argmax(axis=1)
    assert np.array_equal(prediction, frame.predicted_index.to_numpy())
    accuracy = float(np.mean(truth == prediction))
    f1 = []
    for index in range(len(CLASSES)):
        tp = int(((truth == index) & (prediction == index)).sum())
        fp = int(((truth != index) & (prediction == index)).sum())
        fn = int(((truth == index) & (prediction != index)).sum())
        denominator = 2*tp+fp+fn
        f1.append(2*tp/denominator if denominator else 0.0)
    macro_f1 = float(np.mean(f1))
    assert abs(accuracy-selected['validation_accuracy']) < 1e-12
    assert abs(macro_f1-selected['validation_macro_f1']) < 1e-12
    manifest = json.loads((ROOT/'evidence-manifest.json').read_text())
    for relative, digest in manifest['sha256'].items():
        assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest() == digest, relative
    return {'status':'passed', 'validation_rows':len(frame), 'accuracy':accuracy, 'macro_f1':macro_f1,
            'boundary':'Saved prediction evidence only; training and hidden-test inference were not rerun.'}

if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
