.. role:: raw-latex(raw)
   :format: latex
..

.. title:: Global Systemic Risk Indicators

Portfolio of international country indices
==========================================

This page demonstrates the application of several systemic risk measures
applied an portfolio of assets constituting of the following country
indices and their associated ticker symbols from Yahoo Finance
(https://au.finance.yahoo.com/lookup/indices):

-  USA (^GSPC)

-  China (000001.SS)

-  Japan (^N225)

-  UK (^N225)

-  Germany (^GDAXI)

-  France (^FCHI)

-  Australia (^AXJO)

-  Italy (EWI)

-  Canada (^GSPTSE)

These systemic risk measures are calculated on a daily basis and are
updated at 01:00 (AEST,UTC/GMT +10 hours).

Turbulence Index
================

.. bokeh-plot:: hti.py
   :source-position: none

Kritzman and Li (2010) introduce a measure that
captures the degree of multivariate asset “unusualness” through time. It
is based on the glsmd that was originally applied in the categorization
of human skulls (Mahalanobis, 1927). The has
been applied to analyze and stress-test portfolios during turbulent
market conditions Chow et al. (1999).

Intuitively, the captures the statistical “unusualness” of a portfolio
of assets through time. The financial turbulence score increases
significantly when asset prices behave in an unusual manner as given by
(1) extreme price volatility (2) decoupling of correlated assets (3)
convergence of uncorrelated assets.

Kritzman and Li (2010) define the for a portfolio of
:math:`N` assets as shown in Equation .

.. math::

   \label{eq:turbulence_index}
           TI_t=(y_t-{\mu}){\Sigma}^{-1}(y_t-\mu)'

:math:`TI_t` is the for time period :math:`t` and is a scalar value.
:math:`y_t` is the vector (:math:`1 \times N`) of asset returns for
period :math:`t`. :math:`\mu` is the sample average vector
(:math:`1 \times N`) of historical returns. :math:`\Sigma` is the sample
variance-covariance matrix (:math:`N \times N`) of historical returns.

Breaking down the Turbulence Index: Correlation Surprise & Magnitude Surprise
=============================================================================

.. bokeh-plot:: cms-correlation.py
   :source-position: none

.. bokeh-plot:: cms-magnitude.py
   :source-position: none

The Turbulence Index (TI) (Kritzman and Li, 2010) across a portfolio of
assets can be thought of as a multivariate z-score that measures the
statistical “unusualness” of a contemporaneous cross-section of asset
returns, relative to its historical distribution. Thus, it captures two
components, that is the extent to which (1) the risk-adjusted magnitude
of returns differ from their historical means (2) the interaction across
the assets is inconsistent with the historical correlation matrix.
Intuitively, Kinlaw and Turkington (2014) describe that
the captures (1) , the average degree of “unusualness” in individual
asset returns (2) , average degree of “unusualness” in pair-wise
interaction across assets.

Kinlaw and Turkington (2014) disentangle these two
components of the by first calculating and then . is given by the (See
Equation ) where all off-diagonal elements of the matrix are set to
zero. Therefore, is a correlation-blind , that completely ignores any
co-movement within a portfolio. is given by the (that includes all
correlation effects) by as shown in Equation . Therefore, is the
unusualness of interactions across asset returns of the portfolio on a
selected day relative to history.

.. math::

   \label{eq:correlation_surprise}
       \text{Correlation Surprise} = \frac{\text{Turbulence Index}}{\text{Magnitude Surprise}}

Absorption Ratio
================

.. bokeh-plot:: ari.py
   :source-position: none

Based on Principal Components Analysis (PCA), Kritzman et al. (2011) propose a
measure of systemic risk named the . is defined as the fraction of the
total variance of a set of asset returns explained or “absorbed” by a
fixed number of eigenvectors. Intuitively, captures the extent to which
markets are unified or tightly coupled. When markets are tightly
coupled, they are more fragile as negative shocks propagate more quickly
and broadly than when markets are loosely linked. Therefore, a high
value for the corresponds to a high level of systemic risk as it implies
that the sources of risk are more unified. A low indicates less systemic
risk, as sources of risk are more disparate. Therefore, is simply a
measure of market fragility and is a ratio that seeks to measure the
extent to which sources of risk are becoming more or less compact.

To calcuate the , is applied to the matrix (:math:`N \times N`) of
:math:`N` asset returns to obtain the corresponding eigenvectors and
eigenvalues. The for a portfolio of :math:`N` assets is given by
equation

.. math::

   \label{eq:absorption_ratio}
       \text{AR}= \frac{\sum_{i=1}^{n} \sigma^2_{Ei}}{\sum_{j=1}^{N} \sigma^2_{Aj}}

To calculate the , a 500-day sample window is taken to estimate the
matrix of assets. is applied to calculate the eigenvalues and
eigenvectors of the . :math:`n` is selected as the number of
eigenvectors corresponding to 1/5th of the number of assets within the
portfolio. The eigenvectors selected are those that correspond to the
largest eigenvalues of the . :math:`\sigma^2_{Ei}` is the variance of
the :math:`i^{th}` eigenvector and :math:`\sigma^2_{Aj}` is the variance
of the :math:`j^{th}` asset. Both variances :math:`\sigma^2_{Ei}` and
:math:`\sigma^2_{Aj}` are calculated using exponential weighting. Using
an exponential weight decay is analogous to assuming that the market’s
memory of prior events fades away as they recede further into the past.
The half-life of the exponential weight decay is set to equal half the
sample window (250 days).

Probit Forecast
================

.. bokeh-plot:: pf.py
   :source-position: none

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Phasellus tempor nisl diam, a tempor est vestibulum non. Praesent quis feugiat diam, id tempus risus. Vestibulum felis velit, rutrum et hendrerit non, ornare vel ipsum. Integer placerat, elit nec malesuada aliquet, turpis felis pulvinar orci, non pretium diam felis vel sapien. Aliquam semper tortor non purus tincidunt, eu efficitur est mollis. Morbi quis elementum lorem. Quisque arcu enim, hendrerit in eros ut, feugiat lobortis mauris. Morbi volutpat nulla vel ullamcorper dapibus. Maecenas eu vulputate nisi. Fusce tortor lorem, sollicitudin in lorem congue, ornare sagittis nisl. Maecenas pellentesque lorem sit amet lacus bibendum dictum at eget dui. Aliquam sed varius risus, non maximus nulla.

Bibliography
============

Chow, G., Jacquier, E., Kritzman, M., Lowry, K., 1999. Optimal Portfolios in
Good Times and Bad. Financial Analysts Journal 55 (3), 65-73.

Kinlaw, W., Turkington, D., 2014. Correlation Surprise. Journal of Asset Man-
agement 14 (6), 385-399.

Kritzman, M., Li, Y., 2010. Skulls, Financial Turbulence, and Risk Manage-
ment. Financial Analysts Journal 66 (5), 30-41.

Kritzman, M., Li, Y., Page, S., Rigobon, R., 2011. Principal Components as a
Measure of Systemic Risk. Journal of Portfolio Management 37 (4), 112-126.

Mahalanobis, P., 1927. Analysis of Race-Mixture in Bengal. Journal of the Asi-
atic Society of Bengal 23, 301-333.