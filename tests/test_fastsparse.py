import numpy as np
from fastsparse.fit import fastsparse_fit
from fastsparse.predict import fastsparse_predict
from fastsparse.gensynthetic import gen_synthetic

def test_fastsparse_fit():
    X, y = gen_synthetic(100, 10)
    model = fastsparse_fit(X, y, 0.1)
    assert model is not None

def test_fastsparse_predict():
    X, y = gen_synthetic(100, 10)
    model = fastsparse_fit(X, y, 0.1)
    preds = fastsparse_predict(model, X, 0.1)
    assert len(preds) == len(y)
