import sklearn.base
import xgboost
import numpy as np

def make_xgboost_detection_ranking_model(n_estimators=100, **kwargs):
    return xgboost.sklearn.XGBClassifier(eval_metric="map", n_jobs=2,
                                    colsample_bytree=0.5, max_depth=4,
                                    base_score=0.1,
                                    reg_lambda=1.0, alpha=1.0,
                                    tree_method="approx",
                                    objective="rank:pairwise",
                                    n_estimators=n_estimators,
                                    **kwargs)

class XGBoostRankingClassifier(sklearn.base.BaseEstimator):

    def __init__(self, model=None):
        if model is None:
            model = make_xgboost_detection_ranking_model()
        self.model = model

    def fit(self, x, y, update=False):
        
        if not update:
            self.axis_normalizers_min_ = None
            self.axis_normalizers_max_ = None
            self.diff_normalizer_min_ = None
            self.diff_normalizer_max_ = None
            self.ylim_observed = (np.array([np.inf]*2),np.array([-np.inf]*2))
            self.ydiff_lim_observed = (np.inf,-np.inf)

        kwargs = dict()
        if update:
            kwargs["xgb_model"] = self.model.get_booster()
        self.model.fit(x, y, **kwargs)
        y_ = self.model.predict_proba(x)
        
        self.ylim_observed = (np.minimum(self.ylim_observed[0], y_.min(axis=0)),
                              np.maximum(self.ylim_observed[1], y_.max(axis=0)))

        self.axis_normalizers_min_ = self.ylim_observed[0]
        self.axis_normalizers_max_ = self.ylim_observed[1] - self.axis_normalizers_min_
        y_ = (y_ - self.axis_normalizers_min_) / self.axis_normalizers_max_
        y_[:, 1] -= y_[:, 0]

        self.ydiff_lim_observed = (min(self.ydiff_lim_observed[0], y_[:, 1].min()),
                                   max(self.ydiff_lim_observed[1], y_[:, 1].max()))

        self.diff_normalizer_min_ = self.ydiff_lim_observed[0]
        self.diff_normalizer_max_ = self.ydiff_lim_observed[1] - self.diff_normalizer_min_

        return self
    
    def fit_batch(self, x, y, xgb_model_path, iteration):
        self.fit(x, y, update=iteration > 0)
        self.model.save_model(xgb_model_path)
        return self

    def predict_proba(self, x):
        y = self.model.predict_proba(x)
        y = (y - self.axis_normalizers_min_) / self.axis_normalizers_max_
        y[:, 1] -= y[:, 0]
        y[:, 1] = (y[:, 1] - self.diff_normalizer_min_) / self.diff_normalizer_max_
        y[y[:, 1] < 0.0, 1] = 0.0
        y[y[:, 1] > 1.0, 1] = 1.0
        
        y[:, 0] = 1.0 - y[:, 1]
        
        return y
    
    def predict_cost(self, x):
        return self.predict_proba(x)[:, 0] # Cost is the inverse probability of the class being 1.

    def to_dict(self):
        return dict(model=self.model,
                    axis_normalizers=(self.axis_normalizers_min_, self.axis_normalizers_max_),
                    diff_normalizer=(self.diff_normalizer_min_, self.diff_normalizer_max_)
                   )
    
    @classmethod
    def load(cls, path):
        import joblib
        with open(path, "rb") as f:
            d = joblib.load(f)
        return cls.from_dict(d)
    
    @classmethod
    def from_dict(cls, d):
        classifier = XGBoostRankingClassifier()
        classifier.model = d["model"]
        classifier.axis_normalizers_min_, classifier.axis_normalizers_max_ = d["axis_normalizers"]
        classifier.diff_normalizer_min_, classifier.diff_normalizer_max_ = d["diff_normalizer"]
        
        return classifier
    
    def save(self, path):
        import joblib
        
        with open(path, "wb") as f:
            joblib.dump(self.to_dict(), f)
    
    
    @property
    def classes_(self):
        return self.model.classes_
    @property
    def feature_importances_(self):
        return self.model.feature_importances_


def make_detection_model(**kwargs):
    return XGBoostRankingClassifier(model=make_xgboost_detection_ranking_model(**kwargs))