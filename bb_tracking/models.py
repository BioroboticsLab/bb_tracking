import xgboost


# Load:  create common function, because both detection and tracklet model use the same thing
# if update or change in the future, edit the individual functions
def load_xgb_model(model_path):
    """
    Loads an XGBoost model from the specified file path.

    Args:
        model_path (str): The path to the model file.

    Returns:
        XGBClassifier: The loaded XGBoost model.
    """
    try:
        model = xgboost.XGBClassifier()
        model.load_model(model_path)
        return model
    except Exception as e:
        # Handle exceptions such as file not found, etc.
        print(f"Error loading model: {e}")
        return None
# Function aliases that are referenced in bb_behavior (change if differentiate them in the future)
load_detection_model = load_xgb_model
load_tracklet_model = load_xgb_model

# Functions for creating each model, with the parameters that were used
def make_tracklet_model():
    return xgboost.sklearn.XGBClassifier(eval_metric="auc", n_jobs=4,
                                    colsample_bytree=0.75, max_depth=8,
                                    #base_score=0.5,
                                    reg_lambda=10, alpha=1.0,
                                    tree_method="approx",
                                    objective="binary:logistic",
                                    n_estimators=100,
                                    #scale_pos_weight  = 0.05
                                    )

def make_detection_model():
    return xgboost.sklearn.XGBClassifier(eval_metric="auc", n_jobs=4,
                                    colsample_bytree=0.75, max_depth=8,
                                    #base_score=0.5,
                                    reg_lambda=10, alpha=1.0,
                                    tree_method="approx",
                                    objective="binary:logistic",
                                    n_estimators=30,
                                    #scale_pos_weight  = 0.05
                                    )

# note:  save tracklet and detection modesl with built in xgb method:  model.save_model("path.json")