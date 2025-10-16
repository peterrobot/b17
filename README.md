# Log-Pearson Flood Flow Frequency using USGS 17B

This is a python port for bulletin_17B origin written for MATLAB. (Version 1.7.0.0)

> Jeff Burkey (2025). Log-Pearson Flood Flow Frequency using USGS 17B [https://ww2.mathworks.cn/matlabcentral/fileexchange/22628-log-pearson-flood-flow-frequency-using-usgs-17b](<https://ww2.mathworks.cn/matlabcentral/fileexchange/22628-log-pearson-flood-flow-frequency-using-usgs-17b>), MATLAB Central File Exchange.

## Usage

```python
stream_flow = np.random.randn()# load some stream flow to memory (np.ndarray)

import b17
ret = b17.b17(stream_flow)

# Optional, extract result to dataframe
df = b17.extract_ret_df(ret)
```

## Origin description

This function estimates Flood Frequencies using U.S. Water Resources Council publication Flood Flow Frequencies, Bulletin #17B (1976, revised 1981,1982). This method uses a Log Pearson Type-3 distribution for estimating quantiles. See url: [bulletin_17B](<http://water.usgs.gov/osw/bulletin17b/bulletin_17B.html>) for further documentation.

NaN need to be removed in the dataset. If any years have missing data, it will still assume to include that year as part of the sample size-- as stipulated in 17B guidelines. An exmaple MAT file is provided for the user to test. Further down in these comments is some sample script to pre-process the examples.mat file provided.

There are only a couple of loops in this function and subfunctions, most of this is vectorized for speed.

A nice enhancement to this function is a plot of the analyses. It is plotted in probability space-- SKEWED probability space! Because data may not be normally distributed, plotting in skewed space maintains a straight line for the final frequency curve. Again, no need of any
special toolboxes to create this figure. All self contained in this function.

Outputs of this function include:
estimates of a final frequency (based on a weighted skew),
confidence intervals (95%) for the final frequency,
expected frequency curve based on an adjusted conditional
probability,
observed data with computed plotting positions using Gringorten and
Weibull techniques (no toolbox required),
various Skews,
mean of log10(Q),
standard deviation of log10(Q),
and the coup de grâce,
a probability plot that does not require a toolbox to create, but
also plots the probability space using the computed weighted skew
and not just the normal probability.

*This added feature yields a straight line plot for the final
frequency curve even if the data are not normally distributed.

The one important aspect not included in this funtion is the assumed generalized skew (which is variable throughout the country), which can be obtained from Plate 1 in the bulletin. Using the USGS program PKFQWin, this generalized skew is automatically estimated with given lat/long
coordinates. For this function, the user must specify a generalized skew, if no generalized skew is provided, 0.0 is assumed.

Even though this function computes probabilities, skews, etc., no toolboxes are required. All necessary tables are provided as additional MAT files supporting this function. These tables are created from the published USGS 17B manual, and not taken from any Matlab toolboxes, so there are no conflicts or copyright violations.

Other files required to support this function are:
KNtable.mat - using normal distribution, a table of 10-percent
significance level of K values per N sample size.
ktable.mat - Appendix 3 table Pearson distributions
PNtable.mat - Table 11-1 in Appendix 11. Table of probabilities
adjusted for sample size.
pscale.mat - table used to define tick/grid intervals when creating a
probability plot of the results. Can be modified by user if other than
the default values.
examples.mat - dataset presented in the 17B publication.

Parabolic interpolation of Pearson Distribution is dependant on function: lagrange.m (written by Jeff Burkey 3/23/2007). Can be downloaded from Mathworks user community.

Syntax
[dataout skews pp XS SS hp] =
b17(datain, gg, imgfile, gaugeName, plotref)

written by
Jeff Burkey
King County- Department of Natural Resources and Parks
Seattle, WA
email: [jeff.burkey@kingcounty.gov](<jeff.burkey@kingcounty.gov>)
January 6, 2009
