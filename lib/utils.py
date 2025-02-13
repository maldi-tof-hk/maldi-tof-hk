import pandas as pd
import shap
import tensorflow as tf
import numpy as np


def init_libraries():
    """
    Initialize the notebook with useful library settings.
    """

    # Display more columns for large data frames
    pd.set_option("display.max_columns", 500)

    # Set random states
    tf.keras.utils.set_random_seed(812)
    tf.config.experimental.enable_op_determinism()

    # Print in decimals, not scientific notation
    np.set_printoptions(suppress=True)

    # Initialize SHAP
    shap.explainers._deep.deep_tf.op_handlers["AddV2"] = (
        shap.explainers._deep.deep_tf.passthrough
    )
    shap.explainers._deep.deep_tf.op_handlers["LeakyRelu"] = (
        shap.explainers._deep.deep_tf.op_handlers["Relu"]
    )

    print("Libraries initialized")


def random_choice(samples, size, random_state=812):
    rng = np.random.default_rng(random_state)
    return rng.choice(samples, size=size, replace=False)
