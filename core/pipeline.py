from sklearn.pipeline import Pipeline as Sklearn_pipeline
from core import processors

preprocess_pipeline = Sklearn_pipeline(
    [
        (
            "clean_text", processors.CleanText(
                feature_column_name="short_descriptions",
                label_column_name="priority"
            )
        )
    ]
)
