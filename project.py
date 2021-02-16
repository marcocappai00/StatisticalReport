# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 11:34:14 2021
@author: marcocappai
"""
#%%
#Importing libraries and setting working directory
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import scipy.stats as stats
import seaborn as sns
from matplotlib import rc
from scipy.special import erf
import statsmodels.api as sm 
from statsmodels.graphics.gofplots import qqplot_2samples
import matplotlib.ticker as plticker


plt.style.use('seaborn-talk')
path="C:/Challenge/datasets"
os.chdir(path)
rc('text', usetex=True)

#%%
#Importing the three datasets we need
df_business = pd.read_csv('business.csv')
df_user = pd.read_csv('user.csv')
df_review = pd.read_csv('reviews.csv')

#%%
#Global vars and functions
n_points = 1000

def ECDF_function(dataset):
    sorted_x = np.sort(dataset)
    y = np.arange(1, len(dataset)+1)/len(dataset)
    return sorted_x, y
#%%
#nevada_business = df_business[df_business['full_address'].str.contains('NC')]
nevada_business = df_business[df_business['state']=='PA']
nevada_businessID = nevada_business['business_id']
review_businessID = df_review['business_id']
#Finding matching elements with business IDs
matching = review_businessID.isin(nevada_businessID) 
index_array = matching.where(matching==True).dropna().index
nevada_userID = df_review.loc[index_array]['user_id'].drop_duplicates() 
#filtered users leaving reviews in Nevada

#We have found the user IDs of the users leaving reviews in Nevada, 
#we now need to query the users datafile in order to find the specific 
#users which left reviews in the area of Nevada.

userIDs = df_user['user_id']
matching_userID = userIDs.isin(nevada_userID)
index_array_user = matching_userID.where(matching_userID==True).dropna().index
nevada_users = df_user.loc[index_array_user]
nevada_users.head() #this is all the users in the Nevada area

nevada_user_count = nevada_users['review_count']
nevada_user_count.describe()

#Calculating moments
user_mu = nevada_user_count.mean()
user_sd = nevada_user_count.std()
user_sk = nevada_user_count.skew()
user_kr = nevada_user_count.kurtosis()
print('Mean: {}\nStandard deviation: {}\nSkewness: {}\nKurtosis: {}'.format(user_mu,
                                                                            user_sd,
                                                                            user_sk,
                                                                            user_kr))

#%%  REVIEWS PER USER
#we must remove all datapoints where the review count is zero
nevada_user_count = nevada_user_count.where(nevada_user_count!=0.0)
nevada_user_count = nevada_user_count[~np.isnan(nevada_user_count)]

#Finding optimal value of alpha from the data (Maximum likelihood estimation)
xmin_user = np.min(nevada_user_count)
sorted_review_count = np.sort(nevada_user_count) #sorting return vector
alpha = len(sorted_review_count)/np.sum(np.log(sorted_review_count/xmin_user))
print('The fitted alpha for the power-law distribution is {}'.format(alpha))

points_to_use = len(nevada_user_count)

#Creating estimate of power law distribution
x = np.linspace(np.min(sorted_review_count), np.max(sorted_review_count), points_to_use)
y = (np.power(alpha*xmin_user,alpha))/np.power(x, alpha+1)
#Empirical data fitted to histogram points
h, base = np.histogram(sorted_review_count, bins=30, density=True)
base = 0.5*(base[1:]+base[:-1])

h_data, base_data = np.histogram(sorted_review_count, len(nevada_user_count), density=True)
base_data = 0.5*(base_data[1:]+base_data[:-1])


#Plotting histogram and comparing to power law fit
plt.figure(figsize=(9,6))

#sns.scatterplot(base_data, h_data, color='blue',
                #label='datapoints', marker='+', s=200)
sns.lineplot(x, y, label='Pareto distribution, alpha={}'.format(round(alpha, 3)))
sns.scatterplot(base, h,
                label='empirical PDF', marker='o',
                color='black')

plt.xlabel('Number of reviews', fontsize=20)
plt.ylabel('Frequency', fontsize=20)
plt.yscale('log')
plt.xscale('log')
plt.grid(linewidth=0.5)
plt.legend(fontsize=12)
plt.title('Fitting Pareto distribution - Users', fontsize=30)

#%% CUMULATIVE DISTRIBUTION FUNCTION USERS - rank-frequency plot

y_cdf_user = np.arange(1, len(nevada_user_count)+1, 1)
y_cdf_user = y_cdf_user/(len(nevada_user_count)+1) #rank-frequency plot

pareto_cdf_user = 1-np.power((xmin_user/x),alpha)

tempx = np.arange(1, len(sorted_review_count)+1) #x for bases of array of empirically
#computed CDFs

sns.lineplot(x, pareto_cdf_user, label='theoretical cdf',
             linewidth=2)
plt.plot(sorted_review_count, y_cdf_user, label='empirically plotted cdf',
         color='red')
#sns.lineplot(sorted_review_count, y_cdf_user, color='red',
         #linewidth=2, label='empirical cdf', ci=None)
plt.xscale('log')
plt.grid(linewidth=0.5)
#plt.yscale('log')
plt.legend()
plt.title('CDF comparison for Pareto fit - Users', fontsize=30)




#%% Tests for user dataset
##########KSTEST#################################
print(stats.ks_2samp(pareto_cdf_user, y_cdf_user))

#print(stats.kstest(y_cdf_user, 'pareto', (x, alpha)))
#################################################

#stats.probplot(nevada_user_count, (x, alpha), dist='pareto', plot=plt)

#%% CONFIDENCE INTERVALS FOR USERS
#We will now be computing confidence intervals for user amounts using boostrap sampling
bts_retain = 0.8 #the amount of data kept in each resampling
n_sample = 1000 #number of samples which will be conducted
confint = 0.95 #the confidence level we are trying to achieve (in this case 90%)
estimates = np.array([]) #collection of all the estimates in the right tail

for i in range(n_sample):
    sample = nevada_user_count.iloc[np.random.permutation(len(nevada_user_count))]
    sample = sample[0:int(np.round(bts_retain*len(sample)))] 
    sample = np.sort(sample)
    
    N_bts = len(sample)
    estimates = np.append(estimates,
                           N_bts/np.sum(np.log(sample/np.min(sample))))
estimates = np.sort(estimates)
print('{} confidence interval results: [{} - {}]'.format(confint, 
     estimates[int(np.round(0.5*(1-confint)*n_sample))],
     estimates[int(np.round(0.5*(1+confint)*n_sample))]))

#%% REVIEWS PER BUSINESS
#sorting numbers, removing values with 0 and nan
area_businesses = nevada_business['review_count'].where(nevada_business!=0.0)
area_businesses = area_businesses[~np.isnan(area_businesses)]
sorted_review_business = np.sort(area_businesses)

xmin = np.min(area_businesses)
alpha_business = len(sorted_review_business)/(np.sum(np.log(sorted_review_business/xmin)))
print('The fitted alpha for business reviews is {}'.format(alpha_business))

points_to_use = len(area_businesses)

#Creating estimate of power law distribution
x_business = np.linspace(np.min(sorted_review_business), 
                         np.max(sorted_review_business), points_to_use)
y_business = (np.power(alpha_business*xmin,alpha_business))/np.power(x_business, 
                                                            alpha_business+1)

#Empirical data fitted to histogram points
h_business, base_business = np.histogram(sorted_review_business, 
                                         bins=30, density=True)
base_business = 0.5*(base_business[1:]+base_business[:-1])

#Plotting histogram and comparing to power law fit
plt.figure(figsize=(9,6))
sns.scatterplot(base_business, h_business, color='black',
                label='empirical data', marker='o')
sns.lineplot(x_business, y_business, label='power-law distribution, alpha={}'.format(round(alpha_business, 3)))
plt.xlabel('Number of reviews', fontsize=20)
plt.ylabel('Frequency', fontsize=20)
plt.yscale('log')
plt.xscale('log')
plt.grid(linewidth=0.5)
plt.legend(fontsize=12)
plt.title('Fitting power law distribution - Users', fontsize=30)

#%% CUMULATIVE DISTRIBUTION FUNCTION - rank-frequency plot
from statsmodels.distributions.empirical_distribution import ECDF

y_cdf_business = np.arange(1, len(area_businesses)+1, 1)
y_cdf_business = y_cdf_business/(len(area_businesses)+1) #rank-frequency plot


cdf_array = np.array([])
ecdf = ECDF(area_businesses)
values = np.arange(1, len(area_businesses)+1,1) 
for i in values:
    cdf_array = np.append(cdf_array, ecdf(i))
    
pareto_cdf_business = 1-np.power((xmin/x_business),alpha_business)

tempx = np.arange(1,len(area_businesses)+1)

############################################################
sns.lineplot(x_business, pareto_cdf_business, label='theoretical cdf')
#plt.plot(tempx, cdf_array, '*',
         #color='red', label='empirically plotted cdf')

#sns.lineplot(sorted_review_business, y_cdf_business, color='red',
         #'linewidth=2, label='empirical cdf', markersize=5,
         #ci=None)
plt.plot(sorted_review_business,
         y_cdf_business)
plt.xscale('log')
#plt.yscale('log')
plt.margins(0.02)
plt.grid(linewidth=0.5)
plt.legend()
############################################################

#%% Tests for business dataset
#KSTEST########################################################
print(stats.ks_2samp(pareto_cdf_business, y_cdf_business))



#%% CONFIDENCE INTERVALS FOR BUSINESSES
#We will now be computing confidence intervals for business reviews using the bootstrap
bts_retain = 0.8 #the amount of data kept in each resampling
n_sample = 1000 #number of samples which will be conducted
confint = 0.95 #the confidence level we are trying to achieve (in this case 90%)
estimates_business = np.array([]) #collection of all the estimates in the right tail

for i in range(n_sample):
    sample = area_businesses.iloc[np.random.permutation(len(area_businesses))]
    sample = sample[0:int(np.round(bts_retain*len(sample)))] 
    sample = np.sort(sample)
    
    N_bts = len(sample)
    estimates_business = np.append(estimates_business,
                           N_bts/np.sum(np.log(sample/np.min(sample))))
estimates_business = np.sort(estimates_business)
print('{} confidence interval results: [{} - {}]'.format(confint, 
     estimates_business[int(np.round(0.5*(1-confint)*n_sample))],
     estimates_business[int(np.round(0.5*(1+confint)*n_sample))]))

#%% Figure 1 - subplots (2,2) - Users
fontsize = 10
linewidth = 1

fix, axs = plt.subplots(3,2, figsize=(12,14))
plt.subplots_adjust(left=None, 
                    bottom=0.08, 
                    right=None, 
                    top=0.98, 
                    wspace=None, 
                    hspace=None)
#plt.suptitle('Reviews sent by users in PA', fontsize=20)
plt.margins(0)
##########################
#[0,0]
sns.distplot(nevada_user_count, kde=False,
             hist_kws=dict(edgecolor="black", linewidth=linewidth),
             label='Empirical data users',
             ax=axs[0,0])
axs[0,0].set_yscale('log')
axs[0,0].set_ylabel('Frequency', fontsize=fontsize)
axs[0,0].set_xlabel('Reviews', fontsize=fontsize)
axs[0,0].legend(fontsize=fontsize)
axs[0,0].grid(linewidth=0.5)
axs[0,0].set_title('(1.1)', fontsize=12)
##########################
#[1,0]
sns.lineplot(x, y, label='Pareto distribution, alpha={}'.format(round(alpha, 3)),
             ax=axs[1,0])
sns.scatterplot(base, h,
                label='empirical PDF Users', marker='o',
                color='black',
                ax=axs[1,0])
axs[1,0].set_yscale('log')
axs[1,0].set_xscale('log')
axs[1,0].set_xlabel('Reviews', fontsize=fontsize)
axs[1,0].set_ylabel(r'$f(x)$', fontsize=fontsize)
axs[1,0].grid(linewidth=0.5)
axs[1,0].legend(fontsize=fontsize)
axs[1,0].set_title('(1.3)', fontsize=12)

##########################
#[2,0]
sns.lineplot(x, pareto_cdf_user, label='theoretical Pareto CDF, alpha={}'.format(round(alpha, 3)),
             linewidth=1.5,
             ax=axs[2,0])
axs[2,0].plot(sorted_review_count, y_cdf_user, label='empirical CDF Users',
         color='red', 
         linewidth=1.5)
axs[2,0].set_xscale('log')
#sns.lineplot(sorted_review_count, y_cdf_user, color='red',
         #linewidth=2, label='empirical cdf', ci=None)
axs[2,0].grid(linewidth=0.5)
axs[2,0].set_ylabel(r'$F(x)$', fontsize=fontsize)
axs[2,0].set_xlabel('Reviews', fontsize=fontsize)
axs[2,0].legend(fontsize=fontsize)
axs[2,0].set_title('(1.5)', fontsize=12)
axs[2,0].set_ylim([0,1.02])
axs[2,0].set_xlim([0.9, np.max(x)])

##################
#[0,1]
sns.distplot(area_businesses, kde=False,
             hist_kws=dict(edgecolor="black", linewidth=linewidth),
             label='Empirical data Businesses',
             ax=axs[0,1])
axs[0,1].set_yscale('log')
axs[0,1].set_ylabel('Frequency', fontsize=fontsize)
axs[0,1].set_xlabel('Reviews', fontsize=fontsize)
axs[0,1].legend(fontsize=fontsize)
axs[0,1].grid(linewidth=0.5)
axs[0,1].set_title('(1.2)', fontsize=12)
##################
#[1,1]
sns.lineplot(x_business, y_business, label='Pareto distribution, alpha={}'.format(round(alpha_business, 3)),
             ax=axs[1,1])
sns.scatterplot(base_business, h_business,
                label='empirical PDF Businesses', marker='o',
                color='black',
                ax=axs[1,1])
axs[1,1].set_yscale('log')
axs[1,1].set_xscale('log')
axs[1,1].set_xlabel('Reviews', fontsize=fontsize)
axs[1,1].set_ylabel(r'$f(x)$', fontsize=fontsize)
axs[1,1].grid(linewidth=0.5)
axs[1,1].legend(fontsize=fontsize)
axs[1,1].set_title('(1.4)', fontsize=12)
####################
#[2,1]
sns.lineplot(x_business, pareto_cdf_business, label='theoretical Pareto CDF, alpha={}'.format(round(alpha_business, 3)),
             linewidth=1.5,
             ax=axs[2,1])
axs[2,1].plot(sorted_review_business, y_cdf_business, 
              label='empirical CDF Businesses',
              color='red',
              linewidth=1.5)
axs[2,1].set_xscale('log')
#sns.lineplot(sorted_review_count, y_cdf_user, color='red',
         #linewidth=2, label='empirical cdf', ci=None)
axs[2,1].grid(linewidth=0.5)
axs[2,1].set_ylabel(r'$F(x)$', fontsize=fontsize)
axs[2,1].set_xlabel('Reviews', fontsize=fontsize)
axs[2,1].legend(fontsize=fontsize)
axs[2,1].set_title('(1.6)', fontsize=12)
axs[2,1].set_ylim([0,1.02])
axs[2,1].set_xlim([2.8, np.max(x_business)])

#%% PART 2
business_categories = df_business['categories']
bars_df = df_business[business_categories.str.contains('Bars')==True]
bars_NV = bars_df[bars_df['state']=='NV']
bars_AZ = bars_df[bars_df['state']=='AZ']
stars_NV = bars_NV['stars']
stars_AZ = bars_AZ['stars']

NV_stars_mu = stars_NV.mean()
NV_stars_sd = stars_NV.std()
NV_stars_sk = stars_NV.skew()
NV_stars_krt = stars_NV.kurtosis() #in this case this is calculating the excess kurtosis
#so kurtosis of a Gaussian is 0
NV_stars_med = stars_NV.median()
NV_stars_mode = stars_NV.mode()

print('#################################\nStatistics for Nevada ratings:')
print('Mean: {},\nStandard Deviation: {},\nSkewness: {},\nKurtosis: {}'.format(NV_stars_mu,
                                                                               NV_stars_sd,
                                                                               NV_stars_sk,
                                                                               NV_stars_krt))
AZ_stars_mu = stars_AZ.mean()
AZ_stars_sd = stars_AZ.std()
AZ_stars_sk = stars_AZ.skew()
AZ_stars_krt = stars_AZ.kurtosis()
AZ_stars_med = stars_AZ.median()
print('#################################\nStatistics for Arizona ratings:')
print('Mean: {},\nStandard Deviation: {},\nSkewness: {},\nKurtosis: {}'.format(AZ_stars_mu,
                                                                               AZ_stars_sd,
                                                                               AZ_stars_sk,
                                                                               AZ_stars_krt))
#KSTEST - however criticising that it is usually better to conduct in a continuous case
#and the probability distribution we are studying is discrete

#constructing rank-frequency plot for Nevada (NV)
sorted_NV_ratings = np.sort(stars_NV)
cdf_NV = np.arange(1, len(stars_NV)+1, 1)
cdf_NV = cdf_NV/(len(stars_NV)+1)

sorted_AZ_ratings = np.sort(stars_AZ)
cdf_AZ = np.arange(1, len(stars_AZ)+1, 1)
cdf_AZ = cdf_AZ/(len(stars_AZ)+1)


h_NV, base_NV = np.histogram(stars_NV, bins=8, density=True)
base_NV = 0.5*(base_NV[1:]+base_NV[:-1])

h_AZ, base_AZ = np.histogram(stars_AZ, bins=8, density=True)
base_AZ = 0.5*(base_AZ[1:]+base_AZ[:-1])

print(stats.ks_2samp(stars_NV, stars_AZ))


#%%
fig, axs = plt.subplots(2,2, figsize=(14,14))

sns.distplot(stars_NV, bins=8,
             norm_hist=True,
             hist_kws=dict(edgecolor="k", linewidth=0.5),
             color='lightblue',
             kde=False,
             label='NV ratings PDF',
             ax=axs[0,0])
sns.distplot(stars_AZ, bins=8,
             norm_hist=True,
             hist_kws=dict(edgecolor="k", linewidth=0.5),
             color='pink',
             kde=False,
             label='AZ ratings PDF',
             ax=axs[0,0])
axs[0,0].axvline(NV_stars_mu, ls='--',color='red', 
                 label='$\mu - NV$ = {}'.format(round(NV_stars_mu,3)),
                 linewidth=1)
axs[0,0].axvline(AZ_stars_mu, ls='--',color='black', 
                 label='$\mu - AZ$ = {}'.format(round(AZ_stars_mu,3)),
                 linewidth=1)
axs[0,0].grid(linewidth=0.5)
axs[0,0].set_ylabel(r'$p(x)$')
axs[0,0].legend()
axs[0,0].set_title('(2.1)', fontsize=15)
axs[0,0].set_xlabel('Stars')
#--------------------------------------------
colors=['lightblue', 'pink']
medianprops = dict(linestyle='-', linewidth=0.8, color='black')
box1 = axs[0,1].boxplot([x for x in [stars_NV, stars_AZ]],
                 widths=0.4, patch_artist=True, 
                 whiskerprops = dict(linestyle='-',linewidth=0.5),
                 medianprops=medianprops)
axs[0,1].grid(linewidth=0.5)
axs[0,1].set_xticklabels(['NV ratings', 'AZ ratings'])
for patch, color in zip(box1['boxes'], colors):
        patch.set_facecolor(color)
axs[0,1].set_title('(2.2)', fontsize=15)
axs[0,1].set_ylabel('Stars')
axs[0,1].set_xlabel('States')
#----------------------------------------------
qqplot_2samples(stars_NV, stars_AZ, ax=axs[1,0],
                line='45')
axs[1,0].set_title('')
axs[1,0].get_lines()[0].set_marker('o')
axs[1,0].get_lines()[0].set_markerfacecolor('none')
axs[1,0].get_lines()[0].set_markersize(8)
axs[1,0].get_lines()[0].set_markeredgecolor('black')
axs[1,0].get_lines()[0].set_markeredgewidth(0.5)
axs[1,0].get_lines()[0].set_label('empirical data')
axs[1,0].get_lines()[1].set_label('benchmark')
axs[1,0].get_lines()[1].set_linewidth(1)
axs[1,0].get_lines()[1].set_linestyle('--')
axs[1,0].get_lines()[1].set_color('red')
axs[1,0].grid(linewidth=0.5)
#axs[1,0].set_xticks(axs[1,1].get_xticks()[::2])
axs[1,0].legend(fontsize=fontsize)
axs[1,0].set_title('(2.3)', fontsize=15)
#------------------------------------------------
axs[1,1].plot(sorted_NV_ratings, cdf_NV,
              linewidth=1.5, 
              label='NV stars empirical CDF')
axs[1,1].plot(sorted_AZ_ratings, cdf_AZ,
              color='red',
              linewidth=1.5,
              label='AZ stars empirical CDF')
axs[1,1].legend(fontsize=fontsize)
axs[1,1].set_title('(2.4)', fontsize=15)
axs[1,1].set_xlabel('Stars')
axs[1,1].set_ylabel(r'$F(x)$')
axs[1,1].grid(linewidth=0.5)

#%% alternative 3rd plot
sns.scatterplot(base_NV, h_NV, ax=axs[1,0], s=50,
                color='black', label='NV PDF')
sns.scatterplot(base_AZ, h_AZ, ax=axs[1,0], s=50,
                color='red', label='AZ PDF')
axs[1,0].legend(fontsize=fontsize)
axs[1,0].set_title('(3.3)', fontsize=15)
axs[1,0].set_xlabel('Stars')
axs[1,0].set_ylabel(r'$p(x)$')
axs[1,0].grid(linewidth=0.5)










