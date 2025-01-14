import pandas as pd
import tensorflow as tf
import numpy as np

def init_libraries():
    """
    Initialize the notebook with useful library settings.
    """

    # Display more columns for large data frames
    pd.set_option('display.max_columns', 500)

    # Set random states
    tf.keras.utils.set_random_seed(812)
    tf.config.experimental.enable_op_determinism()

    # Print in decimals, not scientific notation
    np.set_printoptions(suppress=True)

    print("Libraries initialized")