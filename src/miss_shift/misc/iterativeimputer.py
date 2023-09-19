import numpy as np
from scipy import stats
from sklearn.base import clone
from sklearn.impute import IterativeImputer
from sklearn.utils import (_safe_indexing)


class FastIterativeImputer(IterativeImputer):
    """The IterativeImputer uses scipy.stats.truncatednorm(), which is 
    pretty slow. This implementation instaed samples from scipy.stats.norm()."""
    def _impute_one_feature(self,
                            X_filled,
                            mask_missing_values,
                            feat_idx,
                            neighbor_feat_idx,
                            estimator=None,
                            fit_mode=True):
        """See `IterativeImputer` for a detailed documentation
        """
        
        if estimator is None and fit_mode is False:
            raise ValueError("If fit_mode is False, then an already-fitted "
                             "estimator should be passed in.")

        if estimator is None:
            estimator = clone(self._estimator)

        missing_row_mask = mask_missing_values[:, feat_idx]
        if fit_mode:
            X_train = _safe_indexing(X_filled[:, neighbor_feat_idx],
                                     ~missing_row_mask)
            y_train = _safe_indexing(X_filled[:, feat_idx],
                                     ~missing_row_mask)
            estimator.fit(X_train, y_train)

        # if no missing values, don't predict
        if np.sum(missing_row_mask) == 0:
            return X_filled, estimator

        # get posterior samples if there is at least one missing value
        X_test = _safe_indexing(X_filled[:, neighbor_feat_idx],
                                missing_row_mask)
        if self.sample_posterior:
            mus, sigmas = estimator.predict(X_test, return_std=True)
            imputed_values = np.zeros(mus.shape, dtype=X_filled.dtype)
            # two types of problems: (1) non-positive sigmas
            # (2) mus outside legal range of min_value and max_value
            # (results in inf sample)
            positive_sigmas = sigmas > 0
            imputed_values[~positive_sigmas] = mus[~positive_sigmas]
            mus_too_low = mus < self._min_value[feat_idx]
            imputed_values[mus_too_low] = self._min_value[feat_idx]
            mus_too_high = mus > self._max_value[feat_idx]
            imputed_values[mus_too_high] = self._max_value[feat_idx]
            # the rest can be sampled without statistical issues
            inrange_mask = positive_sigmas & ~mus_too_low & ~mus_too_high
            mus = mus[inrange_mask]
            sigmas = sigmas[inrange_mask]
            
            normal = stats.norm(loc=mus, scale=sigmas)  # <-- change happened here.
            imputed_values[inrange_mask] = normal.rvs(
                random_state=self.random_state_)
        else:
            imputed_values = estimator.predict(X_test)
            imputed_values = np.clip(imputed_values,
                                     self._min_value[feat_idx],
                                     self._max_value[feat_idx])

        # update the feature
        X_filled[missing_row_mask, feat_idx] = imputed_values
        return X_filled, estimator