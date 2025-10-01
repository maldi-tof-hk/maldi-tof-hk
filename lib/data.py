import numpy as np
import scipy.signal
import pybaselines
import pandas as pd

# Prepare an empty 2D array as a template for the binning step
# The array has 6000 elements, corresponding to 6000 bins
# Each element is a 2D vector where X is the m/z ratio of that bin and Y is the binned intensity
empty_bins = np.vstack(np.array([[i, 0] for i in range(2000, 20000, 3)]))


def preprocess_spectrum_numpy(spectrum):
    """
    Apply preprocessing to a spectrum in the form of a 2D numpy array.

    Preprocess spectrum using the same steps as in the DRIAMS dataset
    reference implementation in R:
      https://github.com/BorgwardtLab/maldi_amr/blob/ae0ccd73150986cbc50814aac5792bb3caac03d0/amr_maldi_ml/DRIAMS_preprocessing/DRIAMS-F_2019_preprocessed.r#L66-L79
    documentation of the R package:
      https://cran.r-project.org/web/packages/MALDIquant/MALDIquant.pdf
    """

    # Apply square root to the whole spectrum to reduce variance
    # R:   spectra = transformIntensity(myspec, method="sqrt")
    np.sqrt(spectrum[:, 1], out=spectrum[:, 1])

    # Apply smoothing to reduce noise
    # R:   spectra = smoothIntensity(spectra, method="SavitzkyGolay", halfWindowSize=10)
    spectrum[:, 1] = scipy.signal.savgol_filter(
        spectrum[:, 1], mode="constant", window_length=21, polyorder=3
    )

    # Normal MALDI-TOF spectra have a baseline intensity that is higher at low m/z values
    # and approach 0 at high m/z values. We are only interested in the peaks that rise above the baseline,
    # so we use the SNIP algorithm to detect and remove that baseline
    # R:   spectra = removeBaseline(removeBaseline(spectra, method="SNIP", iterations=20))
    baseline, dict = pybaselines.smooth.snip(
        spectrum[:, 1], max_half_window=20, decreasing=True
    )
    spectrum[:, 1] = spectrum[:, 1] - baseline

    # The intensity of the spectrum may vary between spectra but is not related to whether the
    # organism is MRSA. We normalize the intensity by setting Total Ion Current to 1, which means
    # that the sum of all intensities should equal 1.
    # R:   spectra = calibrateIntensity(spectra, method="TIC")
    tic = np.sum(spectrum[:, 1])

    # Some spectra seem to be broken and is all zero. This breaks further calculation
    # so we have to abandon these spectra here.
    if tic == 0 or np.isnan(tic):
        raise ValueError(f"TIC is {tic}")

    spectrum[:, 1] = spectrum[:, 1] / tic

    # The highest and lowest m/z values of each spectrum is not fixed.
    # Here we simply trim off values lower than 2000 or higher than 20000.
    # R:   spectra = trim(spectra[[1]], range=c(2000,20000))
    spectrum = spectrum[(spectrum[:, 0] >= 2000) & (spectrum[:, 0] < 20000)]

    # In the raw spectrum, the m/z values (aka the X coordinates) can be a decimal number,
    # but it is very hard to design a model that ingest arbitrary-sized data.
    # Also, the m/z value of the same peak in different spectra may be different, which is
    # called the peak shifting issue.
    # Binning solves these two issues by ensuring that there are always exactly 6000 bins and
    # peaks that have shifted by 1 or 2 Da still land in the same bin.
    spectrum[:, 0] = np.floor((spectrum[:, 0] - 2000) / 3) * 3 + 2000
    spectrum = np.concatenate((spectrum, empty_bins))
    spectrum = np.array(pd.DataFrame(spectrum).groupby(0, as_index=False).sum())

    # Wrong size breaks the model, so we need a safety check here
    if len(spectrum) != 6000:
        raise ValueError(f"Incorrect array size {len(spectrum)}")

    return spectrum


def preprocess_spectrum_file(path: str, sep=" ", header: int | None = None):
    """
    Load a spectrum from a file and preprocess it using the same steps as in the DRIAMS dataset
    reference implementation in R:
      https://github.com/BorgwardtLab/maldi_amr/blob/ae0ccd73150986cbc50814aac5792bb3caac03d0/amr_maldi_ml/DRIAMS_preprocessing/DRIAMS-F_2019_preprocessed.r#L66-L79
    documentation of the R package:
      https://cran.r-project.org/web/packages/MALDIquant/MALDIquant.pdf
    """

    # Retrieve the spectrum. Each column is separated by a space, not by comma
    spectrum = pd.read_csv(path, sep=sep, header=header).to_numpy()

    return preprocess_spectrum_numpy(spectrum)


def load_spectra(path: str = "data/samples_anonymized") -> pd.DataFrame:
    """
    Load spectrum data from the anonymized dataset

    Parameters
    ----------
    path : str
        Path to the folder containing the dataset

    Returns
    -------

    pandas.DataFrame
        Data frame with the 'bins' column containing the spectra
    """

    df = pd.read_csv(f"{path}/index.csv")

    progress = 0

    def get_spectra(row):
        nonlocal progress
        arr = np.loadtxt(f'{path}/spectra_processed/{row["sample_id"]}.txt')
        progress += 1
        if progress == len(df) or progress % 100 == 0:
            print("Load progress: ", progress, "/", len(df))
        return arr

    df["bins"] = df.apply(get_spectra, axis=1)
    return df


def train_val_split(df: pd.DataFrame, train_ratio=0.8, random_state=812):
    """
    Split the data frame into training and validation sets.

    Parameters
    ----------

    df : pandas.DataFrame
        Data frame containing the data to split.
    train_ratio : float
        Ratio of the data to use for training.
    random_state : int
        Random seed to use for the split.

    Returns
    -------

    tuple
        Tuple containing the training and validation sets in the following order:
        X_train, y_train, X_val, y_val
    """

    train = df.sample(frac=train_ratio, random_state=random_state)
    val = df.drop(train.index)

    return xy_split(train, val)


def xy_split(train: pd.DataFrame, val: pd.DataFrame):
    """
    Split the data frames into (x,y) training and validation sets.

    Parameters
    ----------

    train : pandas.DataFrame
        Data frame containing training data.
    val : pandas.DataFrame
        Data frame containing validation data.

    Returns
    -------

    tuple
        Tuple containing the training and validation sets in the following order:
        X_train, y_train, X_val, y_val
    """

    X_train, y_train = df_to_xy(train)
    X_val, y_val = df_to_xy(val)

    print("Data split:")
    print("  Training: ", X_train.shape, y_train.shape)
    print("  Validation: ", X_val.shape, y_val.shape)

    return X_train, y_train, X_val, y_val


def df_to_xy(df: pd.DataFrame):
    """
    Split the data frame into (x,y) sets.

    Parameters
    ----------

    df : pandas.DataFrame
        Data frame containing the data to split.

    Returns
    -------

    tuple
        Tuple containing the sets in the following order:
        X, y
    """

    X = np.vstack(df["bins"].to_numpy())
    y = df["is_mrsa"].to_numpy().astype(float)

    return X, y


def kfold_split(df: pd.DataFrame, n_splits=5, random_state=812):
    """
    Split the data frame into training and validation sets using KFold.

    Parameters
    ----------

    df : pandas.DataFrame
        Data frame containing the data to split.
    n_splits : int
        Number of splits to perform.
    random_state : int
        Random seed to use for the splits.

    Returns
    -------

    list
        List containing the training and validation sets in the following order:
        split_id, X_train, y_train, X_val, y_val
    """

    from sklearn.model_selection import KFold

    inputs = np.vstack(df["bins"].to_numpy())
    targets = df["is_mrsa"].to_numpy().astype(float)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    splits = []

    fold_id = 1
    for train_index, val_index in kf.split(inputs, targets):
        X_train = inputs[train_index]
        y_train = targets[train_index]
        X_val = inputs[val_index]
        y_val = targets[val_index]

        splits.append((fold_id, X_train, y_train, X_val, y_val))
        fold_id += 1

    return splits


def aggregate_intensities(samples, start, end):
    sum = None
    for arg in range(start, end + 1):
        if sum is None:
            sum = samples[:, arg] ** 2
        else:
            sum += samples[:, arg] ** 2
    return np.sqrt(sum)
