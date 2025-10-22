import streamlit as st
import hmac

MODEL_VERSION = 59

st.set_page_config(layout="wide")

def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["password"], st.secrets["PASSWORD"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password.
        else:
            st.session_state["password_correct"] = False

    # Return True if the password is validated.
    if st.session_state.get("password_correct", False):
        return True

    st.write("This app is password-protected. Please enter the password to continue.")

    # Show input for password.
    st.text_input(
        "Password", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
    return False

# if not check_password():
#     st.stop()  # Do not continue if check_password is not True.

# Main Streamlit app starts here
f"""
# AI-MALDI-TOF: MRSA

#### A deep learning model for the classification of MRSA and MSSA from MALDI-TOF mass spectra

- This model is currently in early testing only.
- The only valid input is a MALDI-TOF spectrum of Staphylococcus aureus. Model output is invalid for all other inputs.

*Model version: MRSA v{MODEL_VERSION}*
"""

with st.spinner('Initializing libraries...'):
    import numpy as np
    import pandas as pd
    import kagglehub
    import tensorflow as tf
    import tensorflow.keras as keras
    import keras as krs
    import scipy.signal
    import pybaselines.smooth
    import matplotlib
    import matplotlib.pyplot as plt
    from io import StringIO
    import joblib
    from sklearn.base import BaseEstimator
    from sklearn.base import ClassifierMixin
    import operator
    
    st.write(f"Debug information: tensorflow=={tf.__version__} keras=={krs.__version__}")


class EnsembleClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, keras_models, sk_models, weights=None):
        self.keras_models = keras_models
        self.sk_models = sk_models
        self.weights = weights

    def fit(self, X, y):
        for clf in self.keras_models:
            clf.fit(X, y)
        for clf in self.sk_models:
            clf.fit(X, y)

    def predict(self, X):
        return self.predict_proba(X) > 0.5

    def predict_proba(self, X):
        self.probas_ = [clf.predict(X).reshape((-1,)) for clf in self.keras_models]
        for clf in self.sk_models:
            y_pred = clf.predict_proba(X)
            y_pred = (y_pred/y_pred.sum(axis=1,keepdims=1))[:,1]
            self.probas_.append(y_pred)
        avg = np.average(self.probas_, axis=0, weights=self.weights)

        return avg

@st.cache_resource
def load_keras_model():
    return keras.models.load_model(keras_path + '/model-MRSA.keras')

@st.cache_resource
def load_sk_model():
    return joblib.load(sk_path + '/lightgbm.pkl')

@st.cache_resource
def load_ensemble_model():
    return EnsembleClassifier(
        keras_models=[keras_model],
        sk_models=[sk_model],
        weights=[26, 24]
    )

with st.spinner('Downloading model...'):
    keras_path = kagglehub.model_download(f'hlysine/ai-maldi-tof/keras/mrsa-v37')
    sk_path = kagglehub.model_download(f'hlysine/ai-maldi-tof/scikitLearn/mrsa-v52')
with st.spinner('Loading model...'):
    keras_model = load_keras_model()
    sk_model = load_sk_model()
    ensemble_model = load_ensemble_model()

# ============================================
# DATA PREPROCESSING
# ============================================

empty_bins = np.vstack(np.array([[i, 0] for i in range(2000, 20000, 3)]))
def process_spectrum(file):
    """Read a MALDI-TOF spectrum from the given file and apply preprocessing steps"""

    # Retrieve the spectrum. Each column is separated by a space, not by comma
    spectrum = pd.read_csv(file, sep=" ", header=None, dtype='float').to_numpy()

    # Validate the spectrum
    if spectrum.shape[1] != 2:
        raise ValueError(f"Invalid content: expected 2 columns, but got {spectrum.shape[1]}")
    if spectrum.shape[0] < 3:
        raise ValueError(f"Invalid content: expected at least 3 rows, but got {spectrum.shape[0]}")

    # Preprocess spectrum using the same steps as in the DRIAMS dataset
    # reference implementation in R:
    #   https://github.com/BorgwardtLab/maldi_amr/blob/ae0ccd73150986cbc50814aac5792bb3caac03d0/amr_maldi_ml/DRIAMS_preprocessing/DRIAMS-F_2019_preprocessed.r#L66-L79
    # documentation of the R package:
    #   https://cran.r-project.org/web/packages/MALDIquant/MALDIquant.pdf

    # Apply square root to the whole spectrum to reduce variance
    # R:   spectra = transformIntensity(myspec, method="sqrt")
    np.sqrt(spectrum[:,1], out=spectrum[:,1])

    # Apply smoothing to reduce noise
    # R:   spectra = smoothIntensity(spectra, method="SavitzkyGolay", halfWindowSize=10)
    spectrum[:,1] = scipy.signal.savgol_filter(spectrum[:,1], mode='constant', window_length=21, polyorder=3)

    # Normal MALDI-TOF spectra have a baseline intensity that is higher at low m/z values
    # and approach 0 at high m/z values. We are only interested in the peaks that rise above the baseline,
    # so we use the SNIP algorithm to detect and remove that baseline
    # R:   spectra = removeBaseline(removeBaseline(spectra, method="SNIP", iterations=20))
    baseline, dict = pybaselines.smooth.snip(spectrum[:,1], max_half_window=20, decreasing=True)
    spectrum[:,1] = spectrum[:,1] - baseline

    # The intensity of the spectrum may vary between spectra but is not related to whether the
    # organism is MRSA. We normalize the intensity by setting Total Ion Current to 1, which means
    # that the sum of all intensities should equal 1.
    # R:   spectra = calibrateIntensity(spectra, method="TIC")
    tic = np.sum(spectrum[:,1])
    # Some spectra seem to be broken and is all zero. This breaks further calculation
    # so we have to abandon these spectra here.
    if tic == 0 or np.isnan(tic):
        raise ValueError(f"TIC is {tic} in {file}")
    spectrum[:,1] = spectrum[:,1] / tic

    # The highest and lowest m/z values of each spectrum is not fixed.
    # Here we simply trim off values lower than 2000 or higher than 20000.
    # R:   spectra = trim(spectra[[1]], range=c(2000,20000))
    spectrum = spectrum[(spectrum[:,0] >= 2000) & (spectrum[:,0] < 20000)]

    # I cannot find reference implementation for this step but it is described in the DRIAMS paper
    # In the raw spectrum, the m/z values (aka the X coordinates) can be a decimal number,
    # but it is very hard to design a model that ingest arbitrary-sized data.
    # Also, the m/z value of the same peak in different spectra may be different, which is
    # called the peak shifting issue.
    # Binning solves these two issues by ensuring that there are always exactly 6000 bins and
    # peaks that have shifted by 1 or 2 Da still land in the same bin.
    spectrum[:,0] = np.floor((spectrum[:,0] - 2000) / 3) * 3 + 2000
    spectrum = np.concatenate((spectrum, empty_bins))
    spectrum = np.array(pd.DataFrame(spectrum).groupby(0, as_index=False).sum())

    # Wrong size breaks the model, so we need a safety check here
    if len(spectrum) != 6000:
        raise ValueError(f"Incorrect size {len(spectrum)} for {file}")

    return spectrum


# ============================================
# INFERENCE
# ============================================

with st.form(key='my_form'):
    left, sep, right = st.columns([9, 1, 9])

    with left:
        uploaded_file = st.file_uploader("Upload MALDI-TOF spectrum files", type=["txt"], accept_multiple_files=True)

    with sep:
        st.write("**or**")

    with right:
        content = st.text_area("Paste the content of a MALDI-TOF spectrum file")

    show_spectra = st.toggle("Display spectra", help="Whether to display a preview of all uploaded spectra (May cause lag)")

    submitted = st.form_submit_button(label='Submit', use_container_width=True, type="primary")


if submitted:
    files = {}
    errors = []
    def add_file(name, file):
        if name in files:
            errors.append(f"Duplicate file name: {name}")
        else:
            files[name] = file
    if uploaded_file:
        for file in uploaded_file:
            add_file(file.name, file)
    if content:
        add_file("Pasted content", StringIO(content))
    if errors:
        for error in errors:
            st.error(error)
        st.stop()
    processed = []
    for name, file in files.items():
        try:
            processed.append([name, process_spectrum(file)[:,1], 'Unknown', 0, 'Unknown', 0, 'Unknown', 0])
        except Exception as e:
            errors.append(f"Error processing {name}: {e}")
    if errors:
        for error in errors:
            st.error(error)
    if processed:
        processed = pd.DataFrame(processed, columns=['Name', 'Spectrum', 'v37 Prediction', 'v37 Confidence', 'v52 Prediction', 'v52 Confidence', 'v59 Prediction', 'v59 Confidence'])
        with st.spinner('Predicting...'):
            all_spectra = np.vstack(processed['Spectrum'].to_numpy())
            max = all_spectra.max()
            if max == 0:
                max = 1
            keras_predictions = keras_model.predict(all_spectra)
            sk_predictions = sk_model.predict_proba(all_spectra)
            sk_predictions = (sk_predictions/sk_predictions.sum(axis=1,keepdims=1))[:,1]
            ensemble_predictions = ensemble_model.predict_proba(all_spectra)

            def fill_predictions(ver, y_pred):
                processed[f'{ver} Prediction'] = np.where(y_pred > 0.5, 'MRSA', 'MSSA')
                processed[f'{ver} Confidence'] = np.where(y_pred > 0.5, y_pred, 1 - y_pred)
            
            fill_predictions('v37', keras_predictions)
            fill_predictions('v52', sk_predictions)
            fill_predictions('v59', ensemble_predictions)
        def color_cell(val):
            color = 'red' if val == 'MRSA' else 'blue'
            return 'background-color: %s' % color
        if not show_spectra:
            processed.drop(columns=['Spectrum'], inplace=True)

        st.dataframe(
            processed.style
                .map(color_cell, subset=pd.IndexSlice[:, ['v37 Prediction', 'v52 Prediction', 'v59 Prediction']])
                .background_gradient(cmap='winter', vmin=0.5, vmax=1, subset=pd.IndexSlice[:, ['v37 Confidence', 'v52 Confidence', 'v59 Confidence']]),
            column_config={
                "Name": st.column_config.TextColumn(
                    "Name", help="Name of the uploaded file, or 'Pasted content'"
                ),
                "Spectrum": st.column_config.LineChartColumn(
                    "Spectrum", y_min=0, y_max=max, width="large", help="Preprocessed spectrum"
                ),
                "v37 Prediction": st.column_config.TextColumn(
                    "v37 Prediction", help="Predicted class of the spectrum"
                ),
                "v37 Confidence": st.column_config.NumberColumn(
                    "v37 Confidence",
                    help="Confidence of the model in the prediction",
                ),
                "v52 Prediction": st.column_config.TextColumn(
                    "v52 Prediction", help="Predicted class of the spectrum"
                ),
                "v52 Confidence": st.column_config.NumberColumn(
                    "v52 Confidence",
                    help="Confidence of the model in the prediction",
                ),
                "v59 Prediction": st.column_config.TextColumn(
                    "v59 Prediction", help="Predicted class of the spectrum"
                ),
                "v59 Confidence": st.column_config.NumberColumn(
                    "v59 Confidence",
                    help="Confidence of the model in the prediction",
                ),
            },
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.warning("No valid spectra to process")
