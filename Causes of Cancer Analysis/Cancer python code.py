#!/usr/bin/env python
# coding: utf-8

# In[2]:


from os.path import basename, exists


def download(url):
    filename = basename(url)
    if not exists(filename):
        from urllib.request import urlretrieve

        local, _ = urlretrieve(url, filename)
        print("Downloaded " + local)


download("https://github.com/AllenDowney/ThinkStats2/raw/master/code/thinkstats2.py")
download("https://github.com/AllenDowney/ThinkStats2/raw/master/code/thinkplot.py")
download("https://github.com/AllenDowney/ThinkStats2/raw/master/code/nsfg.py")

download("https://github.com/AllenDowney/ThinkStats2/raw/master/code/2002FemPreg.dct")
download(
    "https://github.com/AllenDowney/ThinkStats2/raw/master/code/2002FemPreg.dat.gz"
)


# In[42]:


import pandas as pd
import thinkplot
import pylab as pl
import thinkstats2
import scipy.stats
import numpy as np

from pandas import *


# In[4]:


Data = pd.read_csv('cancerPatientDataSets.csv')


# In[5]:


Data.head()


# In[186]:


Data.isnull().sum()


# In[59]:


Data.columns=['Patient_ID', 'Age','Gender','Air_Pollution','Alcohol_use', 'Dust Allergy', 'OccuPational_Hazards',
                'Genetic_Risk','chronic Lung Disease', 'Balanced_Diet', 'Obesity', 'Smoking', 'Passive_Smoker',
                'Chest_Pain', 'Coughing_of_Blood', 'Fatigue', 'Weight_Loss', 'Shortness_of_Breath', 'Wheezing',
                'Swallowing_Difficulty', 'Clubbing_of_Finger_Nails', 'Frequent_Cold', 'Dry_Cough', 'Snoring', 'Level']


# In[8]:


Data.hist('Age')
pl.title("Histogram of Age")
pl.xlabel("Age of Respondant With Cancer")
pl.ylabel("Individuals")


# In[9]:


Data.hist('Air_Pollution')
pl.title("Histogram of Air Pollution")
pl.xlabel("Respondant Exposed to Air Pollution")
pl.ylabel("Individuals")


# In[16]:


Data.hist('Obesity')
pl.title("Histogram of Obesity")
pl.xlabel("Respondant With Obesity")
pl.ylabel("Individuals")


# In[17]:


Data.hist('Smoking')
pl.title("Histogram of Smoking")
pl.xlabel("Respondant that Smokes")
pl.ylabel("Individuals")


# In[18]:


Data.hist('Passive_Smoker')
pl.title("Histogram of Passive Smoker")
pl.xlabel("Respondant that's a Passive Smoker")
pl.ylabel("Individuals")


# ## PMF

# In[137]:


smoker = Data.Smoking
smoker.head()


# In[138]:


age = Data.Age


# In[139]:


passiveSmoker = Data.Passive_Smoker


# In[140]:


pollution = Data.Air_Pollution


# In[141]:


obesity = Data.Obesity


# In[33]:


pmf = thinkstats2.Pmf(smoker)
pmf


# In[34]:


pmf2 = thinkstats2.Pmf(age)
pmf2


# In[142]:


pmf3 = thinkstats2.Pmf(passiveSmoker)


# In[143]:


pmf4 = thinkstats2.Pmf(pollution)


# In[144]:


pmf5 = thinkstats2.Pmf(obesity)


# In[190]:


thinkplot.PrePlot(2, cols=2)
thinkplot.Hist(pmf, align='right')
thinkplot.Hist(pmf2, align='left')
thinkplot.Config(xlabel='Smoking and Age', ylabel='probability')

thinkplot.PrePlot(2)
thinkplot.SubPlot(2)
thinkplot.Pmfs([pmf, pmf2])
thinkplot.Show(xlabel='Smoking and Age')


# In[145]:


thinkplot.PrePlot(2, cols=2)
thinkplot.Hist(pmf2, align='right')
thinkplot.Hist(pmf3, align='left')
thinkplot.Config(xlabel='Passive Smoking and Age', ylabel='probability')


# In[149]:


thinkplot.PrePlot(2, cols=2)
thinkplot.Hist(pmf2, align='right')
thinkplot.Hist(pmf4, align='left')
thinkplot.Config(xlabel='Pollution and Age', ylabel='probability')


# In[150]:


thinkplot.PrePlot(2, cols=2)
thinkplot.Hist(pmf2, align='right')
thinkplot.Hist(pmf5, align='left')
thinkplot.Config(xlabel='Obesity and Age', ylabel='probability')


# ## CDF

# In[157]:


cdf = thinkstats2.Cdf(pmf)


# In[158]:


cdf2 = thinkstats2.Cdf(pmf2)


# In[159]:


cdf3 = thinkstats2.Cdf(pmf3)


# In[160]:


cdf4 = thinkstats2.Cdf(pmf4)


# In[161]:


cdf5 = thinkstats2.Cdf(pmf5)


# In[162]:


thinkplot.Cdf(cdf)
thinkplot.Show(xlabel="Smoker", ylabel="CDF")


# In[163]:


thinkplot.Cdf(cdf2)
thinkplot.Show(xlabel="Age", ylabel="CDF")


# In[164]:


thinkplot.Cdf(cdf3)
thinkplot.Show(xlabel="Passive Smoker", ylabel="CDF")


# In[165]:


thinkplot.Cdf(cdf4)
thinkplot.Show(xlabel="Pollution", ylabel="CDF")


# In[166]:


thinkplot.Cdf(cdf5)
thinkplot.Show(xlabel="Obesity", ylabel="CDF")


# ## Distribution

# In[51]:


mu, var = thinkstats2.TrimmedMeanVar(age, p=0.01)
print("Mean, Var", mu, var)

sigma = np.sqrt(var)
print("Sigma", sigma)
xs, ps = thinkstats2.RenderNormalCdf(mu, sigma, low=0, high=12.5)

thinkplot.Plot(xs, ps, label="model", color="0.6")

cdf = thinkstats2.Cdf(age, label="data")

thinkplot.PrePlot(1)
thinkplot.Cdf(cdf)
thinkplot.Config(title="Age Of Cancer Patient", xlabel="Cancer Patient Age", ylabel="CDF")


# In[167]:


mu, var = thinkstats2.TrimmedMeanVar(smoker, p=0.01)
print("Mean, Var", mu, var)

sigma = np.sqrt(var)
print("Sigma", sigma)
xs, ps = thinkstats2.RenderNormalCdf(mu, sigma, low=0, high=12.5)

thinkplot.Plot(xs, ps, label="model", color="0.6")

cdf = thinkstats2.Cdf(smoker, label="data")

thinkplot.PrePlot(1)
thinkplot.Cdf(cdf)
thinkplot.Config(title="Smoker", xlabel="Smoker", ylabel="CDF")


# In[168]:


mu, var = thinkstats2.TrimmedMeanVar(passiveSmoker, p=0.01)
print("Mean, Var", mu, var)

sigma = np.sqrt(var)
print("Sigma", sigma)
xs, ps = thinkstats2.RenderNormalCdf(mu, sigma, low=0, high=12.5)

thinkplot.Plot(xs, ps, label="model", color="0.6")

cdf = thinkstats2.Cdf(passiveSmoker, label="data")

thinkplot.PrePlot(1)
thinkplot.Cdf(cdf)
thinkplot.Config(title="Passive Smoker", xlabel="Passive Smoker", ylabel="CDF")


# In[169]:


mu, var = thinkstats2.TrimmedMeanVar(pollution, p=0.01)
print("Mean, Var", mu, var)

sigma = np.sqrt(var)
print("Sigma", sigma)
xs, ps = thinkstats2.RenderNormalCdf(mu, sigma, low=0, high=12.5)

thinkplot.Plot(xs, ps, label="model", color="0.6")

cdf = thinkstats2.Cdf(pollution, label="data")

thinkplot.PrePlot(1)
thinkplot.Cdf(cdf)
thinkplot.Config(title="Pollution", xlabel="Pollution", ylabel="CDF")


# In[170]:


mu, var = thinkstats2.TrimmedMeanVar(obesity, p=0.01)
print("Mean, Var", mu, var)

sigma = np.sqrt(var)
print("Sigma", sigma)
xs, ps = thinkstats2.RenderNormalCdf(mu, sigma, low=0, high=12.5)

thinkplot.Plot(xs, ps, label="model", color="0.6")

cdf = thinkstats2.Cdf(obesity, label="data")

thinkplot.PrePlot(1)
thinkplot.Cdf(cdf)
thinkplot.Config(title="Obesity", xlabel="Obesity", ylabel="CDF")


# ## Scatter Plots

# In[52]:


thinkplot.Scatter(age, smoker)
thinkplot.Show(xlabel='Age', ylabel='Smoker')


# In[60]:





# In[61]:


thinkplot.Scatter(age, passiveSmoker)
thinkplot.Show(xlabel='Age', ylabel='Passive Smoker')


# In[171]:


thinkplot.Scatter(age, pollution)
thinkplot.Show(xlabel='Age', ylabel='Pollution')


# In[172]:


thinkplot.Scatter(age, obesity)
thinkplot.Show(xlabel='Age', ylabel='Obesity')


# In[64]:


def Jitter(values, jitter=0.5):
          n=len(values)
          return np.random.uniform(-jitter, +jitter, n) + values


# In[173]:


smokers = thinkstats2.Jitter(smoker)
passiveSmokers = thinkstats2.Jitter(passiveSmoker)
ages = thinkstats2.Jitter(age)
pollutions = thinkstats2.Jitter(pollution)
obesitys = thinkstats2.Jitter(obesity)


# In[68]:


thinkplot.Scatter(ages, smokers, alpha=0.2)


# In[70]:


thinkplot.Scatter(ages, passiveSmokers, alpha=0.2)


# In[174]:


thinkplot.Scatter(ages, pollutions, alpha=0.2)


# In[175]:


thinkplot.Scatter(ages, obesitys, alpha=0.2)


# In[72]:


def Cov(xs, ys, meanx=None, meany=None):
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    
    if meanx is None:
        meanx = np.mean(xs)
    if meany is None:
        meany = np.mean(ys)
        
    cov = np.dot(xs-meanx, ys-meany)/len(xs)
    return cov


# In[77]:


def Corr(xs, ys):
    xs = np.asarray(xs)
    ys = np.asarray(ys)

    meanx, varx = thinkstats2.MeanVar(xs)
    meany, vary = thinkstats2.MeanVar(ys)

    corr = Cov(xs, ys, meanx, meany) / np.sqrt(varx * vary)
    return corr


# In[78]:


Cov(smoker, age)


# In[79]:


Corr(smoker, age)


# In[176]:


Corr(pollution, age)


# In[177]:


Corr(obesity, age)


# In[80]:


Cov(passiveSmoker, age)


# In[81]:


Corr(passiveSmoker, age)


# In[178]:


Cov(pollution, age)


# In[179]:


Cov(obesity, age)


# In[85]:


def SpearmanCorr(xs, ys):
    xs = pd.Series(xs)
    ys = pd.Series(ys)
    return xs.corr(ys, method='spearman')


# In[86]:


SpearmanCorr(smoker, age)


# In[87]:


SpearmanCorr(passiveSmoker, age)


# In[180]:


SpearmanCorr(pollution, age)


# In[181]:


SpearmanCorr(obesity, age)


# ## Hypothesis test

# In[98]:


def ChiTest(x):
        observed = x
        n = sum(observed)
        expected = np.ones(1000) * n / 1000
        test_stat = sum((observed - expected)**2 / expected)
        return test_stat


# In[99]:


ChiTest(smoker)


# In[117]:


ChiTest(passiveSmoker)


# In[182]:


ChiTest(age)


# In[183]:


ChiTest(pollution)


# In[184]:


ChiTest(obesity)


# ## Regression Analysis

# In[191]:


import statsmodels.formula.api as smf

formula = 'Smoking ~ Age'
model = smf.ols(formula, data=Data)
results = model.fit()
results.summary()


# In[192]:


formula = 'Passive_Smoker ~ Age'
model = smf.ols(formula, data=Data)
results = model.fit()
results.summary()


# In[193]:


formula = 'Air_Pollution ~ Age'
model = smf.ols(formula, data=Data)
results = model.fit()
results.summary()


# In[194]:


formula = 'Obesity ~ Age'
model = smf.ols(formula, data=Data)
results = model.fit()
results.summary()


# In[ ]:




