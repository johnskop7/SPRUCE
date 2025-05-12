# piece.py
import matplotlib.pyplot as plt
import numpy as np
import torch
import pickle
import pandas as pd
from scipy import stats
from tqdm import tqdm
import math
from sklearn.mixture import GaussianMixture


###############################################################################
#                  HELPER: 1D GMM CDF
###############################################################################
def gmm_cdf_1d(x, gmm):
    """
    Compute the 1D CDF of a GaussianMixture model at scalar or array x.
    Assumes gmm is fitted on 1D data (covariances_.shape = [n_components, 1, 1]).

    x: float or 1D numpy array
    gmm: a fitted GaussianMixture model from sklearn
    Returns: array of the same shape as x, containing F(x).
    """
    weights = gmm.weights_.flatten()  # [n_components]
    means = gmm.means_.flatten()  # [n_components]
    variances = gmm.covariances_.reshape(-1)  # [n_components]

    x = np.atleast_1d(x)
    cdf_vals = np.zeros_like(x, dtype=float)

    for i, xi in enumerate(x):
        total = 0.0
        for k in range(len(weights)):
            w = weights[k]
            mu = means[k]
            sigma = math.sqrt(variances[k])
            # Gaussian CDF for component k at xi
            z = (xi - mu) / sigma
            comp_cdf = 0.5 * (1.0 + math.erf(z / np.sqrt(2.0)))
            total += w * comp_cdf
        cdf_vals[i] = total
    return cdf_vals


###############################################################################
#         GMM-BASED MODEL FOR RESNET50 (WITH BERNoulli FOR ZERO VS. NON-ZERO)
###############################################################################
class GMMResNet50:
    """
    A GMM-based model for ResNet50-style ReLU activations:
      - Bernoulli for zero vs. non-zero
      - GaussianMixture for non-zero data
      - Enforces positivity checks (data >= 0).
    """

    def __init__(self, data, value=0.0, p_value=0.05, n_components=2):
        data = np.array(data, dtype=float).flatten()
        if len(data) == 0 or np.isnan(data).all():
            raise ValueError("Data is empty or all values are NaN.")
        if np.any(data < 0):
            raise ValueError("*** ERROR: Must be non-negative data (ReLU) ***")

        self.data = data
        self.value = float(value)
        self.p_value = p_value

        # Bernoulli part: zero vs. non-zero
        self.filtered_data = self.data[self.data != 0]
        self.bern_param = len(self.filtered_data) / len(self.data)

        self.n_components = n_components

        # Fit a GMM for non-zero data if sufficient samples exist
        if len(self.filtered_data) > 1:
            self.gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
            X = self.filtered_data.reshape(-1, 1)
            self.gmm.fit(X)
            self.rv_name = f"GMM_{n_components}"
        else:
            # Not enough non-zero data
            self.gmm = None
            self.rv_name = "GMM_None"

    def __pdf(self, x):
        """ GMM PDF multiplied by bern_param """
        if self.gmm is None:
            return 0.0
        x_ = np.array(x, ndmin=1).reshape(-1, 1)
        logpdf = self.gmm.score_samples(x_)
        return np.exp(logpdf) * self.bern_param

    def __cdf(self, x):
        """
        GMM CDF multiplied by bern_param.
        If x <= 0, cdf=0 for the continuous portion.
        """
        if x <= 0:
            return 0.0
        if self.gmm is None:
            return self.bern_param  # Degenerate case
        F = gmm_cdf_1d([x], self.gmm)[0]
        return F * self.bern_param

    def get_expected_value(self):
        if self.gmm is None:
            return 0.0
        means = self.gmm.means_.flatten()
        weights = self.gmm.weights_.flatten()
        return np.sum(means * weights)

    def get_prob_of_value(self):
        """
        Probability of observing self.value in the 'GMM+Bernoulli' mixture.
        If value=0 => 1 - self.bern_param
        Otherwise => 2-tailed: min(cdf(value), bern_param - cdf(value))
        """
        if self.value == 0:
            return 1 - self.bern_param

        cdf_val = self.__cdf(self.value)
        upper = self.bern_param - cdf_val
        return min(cdf_val, upper)

    def bern_fail_sig(self):
        # Zero is 'unexpectedly rare'
        if (1 - self.bern_param) < self.p_value and self.value == 0:
            return True
        return False

    def bern_success_sig(self):
        # Non-zero is 'unexpectedly rare'
        if self.bern_param < (self.p_value * 2) and self.value > 0:
            return True
        return False

    def high_cont_sig(self):
        """
        For large outliers, check if upper tail is < p_value
        """
        if self.value == 0:
            return False
        if self.gmm is None:
            return False
        cdf_val = self.__cdf(self.value)
        upper_tail = self.bern_param - cdf_val
        return (upper_tail < self.p_value)

    def low_cont_sig(self):
        """
        For small outliers, check if cdf(value) < p_value
        """
        if self.value == 0:
            return False
        if self.gmm is None:
            return False
        if self.__cdf(self.value) < self.p_value:
            return True
        return False

    def test_fit(self):
        return None

    @property
    def rv(self):
        class DummyRV:
            def __init__(self, name):
                self.name = name

        return DummyRV(self.rv_name)

    @property
    def params(self):
        return None

    @property
    def fixed_location(self):
        return False


###############################################################################
#                 GMM-BASED MODEL FOR CONVNEXT (NEGATIVE ALLOWED)
###############################################################################
class GMMConvNeXt:
    """
    A GMM-based model for ConvNeXt:
      - Entire data is fit with a GaussianMixture (allows negative values)
      - 2-tailed probability = min(CDF(value), 1 - CDF(value))
    """

    def __init__(self, data, value=0.0, p_value=0.05, n_components=2):
        data = np.array(data, dtype=float).flatten()
        if len(data) == 0 or np.isnan(data).all():
            raise ValueError("Data is empty or all values are NaN.")

        self.data = data
        self.value = float(value)
        self.p_value = p_value
        self.n_components = n_components

        if len(data) > 1:
            self.gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
            X = data.reshape(-1, 1)
            self.gmm.fit(X)
            self.rv_name = f"GMM_{n_components}"
        else:
            self.gmm = None
            self.rv_name = "GMM_None"

    def get_expected_value(self):
        if self.gmm is None:
            return 0.0
        means = self.gmm.means_.flatten()
        weights = self.gmm.weights_.flatten()
        return np.sum(means * weights)

    def get_prob_of_value(self):
        """
        2-tailed check: min(CDF(value), 1 - CDF(value)).
        """
        if self.gmm is None:
            return 0.5
        F = gmm_cdf_1d([self.value], self.gmm)[0]
        return min(F, 1 - F)

    def bern_fail_sig(self):
        return False

    def bern_success_sig(self):
        return False

    def high_cont_sig(self):
        if self.gmm is None:
            return False
        F = gmm_cdf_1d([self.value], self.gmm)[0]
        return ((1 - F) < self.p_value)

    def low_cont_sig(self):
        if self.gmm is None:
            return False
        F = gmm_cdf_1d([self.value], self.gmm)[0]
        return (F < self.p_value)

    def test_fit(self):
        return None

    @property
    def rv(self):
        class DummyRV:
            def __init__(self, name):
                self.name = name

        return DummyRV(self.rv_name)

    @property
    def params(self):
        return None

    @property
    def fixed_location(self):
        return False


###############################################################################
#               PRECOMPUTE GMMs FOR A GIVEN CLASS
###############################################################################
def precompute_gmms_for_class(dist_data,
                              class_idx,
                              classifier_type='convnext',
                              n_components=2,
                              alpha=0.05):
    """
    For the given class 'class_idx', fit ONE GMM per feature, so we don't have
    to re-fit for every query image. Returns a list of model objects (one per feature).
    """
    data_for_class = dist_data[class_idx]  # shape: [N_samples, N_features]
    num_features   = data_for_class.shape[1]

    gmms = []
    # wrap the range in tqdm
    for fidx in tqdm(range(num_features),
                     desc="Precomputing GMMs",
                     unit="feature"):
        column = data_for_class[:, fidx]

        if classifier_type == 'resnet50':
            gmm_obj = GMMResNet50(column,
                                  value=0.0,
                                  p_value=alpha,
                                  n_components=n_components)
        elif classifier_type in ('convnext', 'efficientnet'):
            gmm_obj = GMMConvNeXt(column,
                                  value=0.0,
                                  p_value=alpha,
                                  n_components=n_components)
        else:
            raise ValueError(f"Unknown classifier_type: {classifier_type}")
        gmms.append(gmm_obj)
    return gmms


###############################################################################
#         ACQUIRE FEATURE PROBABILITIES, USING PRE-FITTED GMMs
###############################################################################
def acquire_feature_probabilities(
        target_class,
        classifier,
        original_query_img=None,
        alpha=0.05,
        precomputed_gmms=None,  # pass in the GMMs we already fit
        classifier_type='resnet50'
):
    """
    Acquire feature probabilities for a specific class & query image,
    but DO NOT re-fit GMMs. Instead, use 'precomputed_gmms' (one per feature).

    Returns a DataFrame of shape [n_features, ...].
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier.to(device)
    classifier.eval()

    if original_query_img is not None:
        original_query_img = original_query_img.to(device)

    # Forward pass -> feature vector
    with torch.no_grad():
        logits, features = classifier(original_query_img, return_features=True)
        query_features = features[0].detach().cpu().numpy()  # shape [n_features,]

    # Class-specific final-layer weights (for sign checks, etc.)

    if classifier_type == 'resnet50':
        digit_weights = classifier.model.fc.weight.detach().cpu().numpy()[target_class]
    elif classifier_type == 'convnext':
        digit_weights = classifier.convnext.classifier[2].weight.detach().cpu().numpy()[target_class]
    elif classifier_type == 'efficientnet':
      # our CustomEfficientNetB3 stores the head in backbone.classifier[1]
        digit_weights = classifier.backbone.classifier[1].weight.detach().cpu().numpy()[target_class]
    else:
        raise ValueError(f"Unknown classifier_type: {classifier_type}")

    fail_results, succeed_results = [], []
    high_results, low_results = [], []
    expected_values, probability = [], []
    distribution_type = []

    # We assume precomputed_gmms is a list of length == num features
    n_features = len(query_features)

    for i in range(n_features):
        feature_value = float(query_features[i])

        # Grab the GMM we fit once for this feature
        gmm_obj = precomputed_gmms[i]

        # Overwrite the 'value' and 'p_value' so it applies to *this* query
        gmm_obj.value = feature_value
        gmm_obj.p_value = alpha

        fail_results.append(gmm_obj.bern_fail_sig())
        succeed_results.append(gmm_obj.bern_success_sig())
        high_results.append(gmm_obj.high_cont_sig())
        low_results.append(gmm_obj.low_cont_sig())
        expected_values.append(gmm_obj.get_expected_value())
        probability.append(gmm_obj.get_prob_of_value())
        distribution_type.append(gmm_obj.rv.name)

    contributions = query_features * digit_weights
    df = pd.DataFrame({
        'Feature Map': list(range(n_features)),
        'Contribution': contributions,
        'Bern Fail': fail_results,
        'Bern Success': succeed_results,
        'Cont High': high_results,
        'Cont Low': low_results,
        'Expected Value': expected_values,
        'Probability of Event': probability,
        'Dist Type': distribution_type
    })

    pd.set_option('display.float_format', lambda x: '%.4f' % x)
    return df


###############################################################################
#             FILTER DF OF EXCEPTIONAL NOISE (SAME LOGIC)
###############################################################################
def filter_df_of_exceptional_noise(df, target_class, classifier, alpha=0.05, classifier_type='resnet50'):
    """
    Filter the dataframe for features deemed 'exceptional,'
    referencing the sign of final-layer weights and significance checks.
    """
    # Keep only rows that have Probability < alpha
    df = df[df['Probability of Event'] < alpha].copy()
    df['flag'] = np.zeros(df.shape[0])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if classifier_type == 'resnet50':
        digit_weights = classifier.model.fc.weight.detach().cpu().numpy()[target_class]
    elif classifier_type == 'convnext':
        digit_weights = classifier.convnext.classifier[2].weight.detach().cpu().numpy()[target_class]
    elif classifier_type == 'efficientnet':
      # our CustomEfficientNetB3 stores the head in backbone.classifier[1]
        digit_weights = classifier.backbone.classifier[1].weight.detach().cpu().numpy()[target_class]
    else:
        raise ValueError(f"Unknown classifier_type: {classifier_type}")

    for idx, row in df.iterrows():
        feature_idx = int(row['Feature Map'])
        cont_high = row['Cont High']
        cont_low = row['Cont Low']
        bern_fail = row['Bern Fail']

        # Original logic for resnet, else convnext
        if classifier_type == 'resnet50':
            if bern_fail and digit_weights[feature_idx] < 0:
                df.loc[idx, 'flag'] = 1
            if cont_high and digit_weights[feature_idx] > 0:
                df.loc[idx, 'flag'] = 1
            if cont_low and digit_weights[feature_idx] < 0:
                df.loc[idx, 'flag'] = 1
        else:
            if cont_high and digit_weights[feature_idx] > 0:
                df.loc[idx, 'flag'] = 1
            if cont_low and digit_weights[feature_idx] < 0:
                df.loc[idx, 'flag'] = 1

    df = df[df['flag'] == 0].drop('flag', axis=1)
    return df


###############################################################################
#            MODIFYING EXCEPTIONAL FEATURES (COUNTERFACTUAL STEP)
###############################################################################
def modifying_exceptional_features(df, target_class, query_activations):
    """
    Replace the 'exceptional' features with their expected values.
    """
    ideal_xp = query_activations.clone().detach()
    df_sorted = df.sort_values('Probability of Event', ascending=True)
    for idx, row in df_sorted.iterrows():
        feature_idx = int(row['Feature Map'])
        if feature_idx >= ideal_xp.size(0):
            print(f"Skipping index {feature_idx} as out-of-bounds for tensor size {ideal_xp.size(0)}")
            continue
        expected_value = row['Expected Value']
        if expected_value is not None:
            ideal_xp[feature_idx] = expected_value

    return ideal_xp


