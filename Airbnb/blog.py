# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 0.8.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # What can you do to improve Airbnb business?
#
# This is the project of Udacity's data scientist nanodegree program (Term2, Project1).  
# In this project, we are required to analysis data with **CRISP-DM** process. The CRISP-DM process is below.
#
# #### CRISP-DM (Cross-Industry Standard Process for Data Mining)
# - Business Understanding
# - Data Understanding
# - Data Preparation
# - Modeling
# - Evaluation
# - Deployment
#
# So, in this kernel I analysis data with following this process.

# ## Business Understanding
#
# Airbnb is a platform of accommodation which match the needs of staying and of lending.  
# Their main source of income is **fee for host**. Basically, as the number of transactions between the host and the guest increases, their profit also increases.  
# So, It is important to their business and I expect it to be one of their KPIs.
#
#
# <img src="https://bmtoolbox.net/wp-content/uploads/2016/06/airbnb.jpg" width=700>
#
# ref: https://bmtoolbox.net/stories/airbnb/
#
# #### What can we do to increase the transactions?
# I considered three below questions to explore its way.
#
# * **How long is the period available for lending by rooms?**  
# Is there rooms which is available all for years? or almost rooms are available on spot like one day or one week?  
# Here, I want to know the trend in the outline of the data.
#
# * **Is there a busy season?**  
# If the demand for accommodation is more than the number of rooms available for lending, it leads to the loss of business opportunities.  
# So, I want to know whether is there the busy season. If this is true, we must create a mechanism to increase the number of rooms available for lending during the busy season.
#
# * **Are there any trends of popular rooms?**  
# If this question's answer is true, we can suggest host to ways make the room easier to rent.  
# In this part, I'll use machine learning technique.

# ## Data Understanding
#
# We have three data.
#
# * `listings`: including full descriptions and average review score
# * `calendar`: including listing id and the price and availability for that day
# * `reviews`: including unique id for each reviewer and detailed comments
#
# In this part, I'll make some visualization and aggregation to understand the charactoristics of the data.

# +
# import necessary package
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %matplotlib inline
import warnings
warnings.filterwarnings('ignore')
# -

# load data
seattle_calendar = pd.read_csv('seattle/calendar.csv')
seattle_listing = pd.read_csv('seattle/listings.csv')
seattle_review = pd.read_csv('seattle/reviews.csv')

# ### calendar

# Let's look first 5 row of the data and column information.

seattle_calendar.head()

seattle_calendar.info()

# There are 4 columns.  
# Here, I found some charactoristics of the data.
#
# * Not only available days are stored in data, it seems to be stored not available days.
# * If the `available` values are `f`, the `price` values seems to be `NaN`. 
# * The `price` values are stored as object, not integer. This is caused the value stored like `$xx.xx`, and it is necessary to transform this column.

# In response to the result, now I have two question .
#
# 1. If the available values are f, the price values seems to be NaN. But it is only a hypothesis, is it true all data?
# 2. How many rows per each listing_id?
#
# Let's answer these questions with exploring data.

#  If the available values are f, the price values seems to be NaN. But it is only a hypothesis, is it true all data?
calendar_q1_df = seattle_calendar.groupby('available')['price'].count().reset_index()
calendar_q1_df.columns = ['available', 'price_nonnull_count']
calendar_q1_df

#  How many rows per each listing_id?
calendar_q2_df = seattle_calendar.groupby('listing_id')['date'].count().reset_index()
calendar_q2_df['date'].value_counts()

# Above, I can answer my question. The answer is
#
# ***If the available values are f, the price values seems to be NaN. But it is only a hypothesis, is it true all data?***  
# -> true !!
#
# ***How many rows per each listing_id?***  
# -> 365 days record. This is equal a year.

# Now, I almost understood the features of the data.  
# Finally, I'll research is there any trend of the listings price.

# +
# process data
calendar_q3_df = seattle_calendar.copy(deep=True)
calendar_q3_df.dropna(inplace=True)
calendar_q3_df['date'] = pd.to_datetime(calendar_q3_df['date'])
calendar_q3_df['price'] = calendar_q3_df['price'].map(lambda x: float(x[1:].replace(",", "")))

# apply aggregation
calendar_q3_df = calendar_q3_df.groupby('date')['price'].mean().reset_index()

# plot avg listings prices over time.
plt.figure(figsize=(15, 8))
plt.plot(calendar_q3_df.date, calendar_q3_df.price, color='b', marker='.', linewidth=0.9)
plt.title("Average listing price by date")
plt.grid()
# -

# This is interesting.  
# There are two trend of the data.
#
# 1. The average price rise from 2016/1 to 2016/7, and reach peak for three months, and getting lower. And the average proce of 2017/1 is higher than 1 years ago.
# 2. There is periodic small peak.

# The first trend can be split into two foctors. One is seasonal factor, and the other is overall factor.  
# The second trend looks like a weekly trend, so let's close look at!!

# plot more narrow range
plt.figure(figsize=(15, 8))
plt.plot(calendar_q3_df.date.values[:15], calendar_q3_df.price.values[:15], color='b', marker='o', linewidth=1.5)
plt.title("Average listing price by date")
plt.grid()

# It looks like a weekly trend as I thought.  
# Then, which does weekday have high price? 

# +
# create weekday column
calendar_q3_df["weekday"] = calendar_q3_df["date"].dt.weekday_name

# boxplot to see price distribution
plt.figure(figsize=(15, 8))
sns.boxplot(x = 'weekday',  y = 'price', data = calendar_q3_df, palette="Blues", width=0.6)
plt.show()
# -

# The weekend, Friday and Saturday has high prices. 

# #### Summary
#
# * Each listings has `365` days record in this data.
# * If `available` values are `f`, the `price` values are `NaN`.
# * There is the weekly trend which the listing prices in weekend are higher than other weekday.

# ### listings

# Let's begin with looking at first 5 row of the data and columns information.

seattle_listing.head()

print(list(seattle_listing.columns.values))

# There are many columns, so I can't explore each columns here.  
# Here I'll look at some columns of my interest.

# First, I'll investigate how many listings are in the data.

print("Num of listings: ", seattle_listing.id.count())
print("Num of rows: ", seattle_listing.shape[0])

# This shows the each rows represents unique listings.

# Next, I am interested in below columns.
#
# * review_scores_rating
# * price
# * maximum_nights
#
# What is the distribution of these values in each columns? Is the distribution skewed or normal?  
# Let's look at!

# #### review_scores_rating

seattle_listing['review_scores_rating'].describe().reset_index()

# +
# cleaning data
listings_q1_df = seattle_listing['review_scores_rating'].dropna()

# plot histgram
plt.figure(figsize=(15, 8))
plt.hist(listings_q1_df.values, bins=80, color='b')
plt.grid()
# -

# This is very right skewed distribution.  
# The 75% or more values are 90 points. And the most common thing is 100 points.  
# I can say the low score listings are minolity.

# #### price

# +
# cleaning data
listings_q2_df = seattle_listing.copy(deep=True)
listings_q2_df = listings_q2_df['price'].dropna().reset_index()
listings_q2_df['price'] = listings_q2_df['price'].map(lambda x: float(x[1:].replace(',', '')))

listings_q2_df['price'].describe().reset_index()
# -

plt.figure(figsize=(15, 8))
plt.hist(listings_q2_df.price, bins=100, color='b')
plt.grid()

# This is long tail distribution.  
# Almost values are from 0 to 200.

# #### maximum_nights

seattle_listing['maximum_nights'].describe().reset_index()

# +
# eliminate outliers because maximum values are very large.
listings_q3_df = seattle_listing[seattle_listing['maximum_nights'] <= 1500]

plt.figure(figsize=(15, 8))
plt.hist(listings_q3_df.maximum_nights, bins=100, color='b')
plt.grid()
# -

# This is very surprising because I expect it would be a week at most.  
# In fact, almost `maxmum_night` values are setted 1125.   
# I have not used Airbnb so I don't know, but maybe there may be something like the default value.  
# Or there maybe two segments, one is `spot available listings`, the other is `long term listings like normal rent`. 

# #### Summary
#
# * The listings data has 92 columns.
# * The `review_scores_rating` has right skewed distribution, and almost values are over 90 points.
# * The `price` has long tail distribution, almost values are around 100\$ but some values are much higher than other values.  
# * The `maximum_nights` has very special distribution. Their are two segments, one is about 3 years, the other is around 1week.
#
# OK, let's look at last data.

# ### reviews

# Let's begin with looking at first 5 row of the data and columns information.

seattle_review.head()

seattle_review.info()

# There are six columns, such as listing_id that received review, id of reviews, when review submitted, and so on.  
# I'm concerned that there are no review scores here. I think it might be in comments, so let's confirm this.

print("sample 1: ", seattle_review.comments.values[0], "\n")
print("sample 2: ", seattle_review.comments.values[3])

# From the above, the review score seems not to be included.

# Next, I want to see the time series change of the number of comments.

# +
# convert date column's data type to date from object
review_q1_df = seattle_review.copy(deep=True)
review_q1_df.date = pd.to_datetime(review_q1_df.date)

review_q1_df = review_q1_df.groupby('date')['id'].count().reset_index()

# plot avg listings prices over time.
plt.figure(figsize=(15, 8))
plt.plot(review_q1_df.date, review_q1_df.id, color='b', linewidth=0.9)
plt.title("Number of reviews by date")
plt.grid()
# -

# It is little noisy, but we can see an increase in the number of Airbnb users. (and the date range is wide than calendar data)  
# And I realize it seems to have a peak at about the same time of each year.  
# So, let's use moving averages to smooth the graph.

# +
# create rolling mean column
review_q1_df["rolling_mean_30"] = review_q1_df.id.rolling(window=30).mean()

# plot avg listings prices over time.
plt.figure(figsize=(15, 8))
plt.plot(review_q1_df.date, review_q1_df.rolling_mean_30, color='b', linewidth=2.0)
plt.title("Number of reviews by date")
plt.grid()
# -

# I tried thirty days (about 1 month) window.  
# The graph became smooth and the trend became clear, and my belief that the peaks were in the same place became stronger.  
# Next, I extract when the peak comes in each year.

# +
review_q1_df["year"] = review_q1_df.date.dt.year
years = review_q1_df.year.unique()

for year in years:
    if year >= 2010 and year < 2016:
        year_df = review_q1_df[review_q1_df.year == year]
        max_value = year_df.rolling_mean_30.max()
        max_date = year_df[year_df.rolling_mean_30 == max_value].date.dt.date.values[0]
        print(year, max_date, np.round(max_value, 1))
# -

# My hypothesis is correct.  
# The peak seems to be towards the beginning of September!!
# Is this summer vacation?

# ### Answer my Question

# ここまでで、私は冒頭で述べた三つの質問のうち二つに答えることができる。
# まず一つめの、質問から回答していこう。
# How long is the period available for lending by rooms?
#
# これは、listingデータを調査している時に示された。listingsには二つのグループがあった。それは、最大貸し出し日数が１週間以内のスポットで利用可能なlistingと、最大3年まで利用可能な、賃貸のような感覚のlistingだ。
#
# さらに考察を得るため、最大の貸し出し日数と、最低貸し出し日数の散布図をプロットしてみよう

# +
listings_q3_df["min_max_night_diff"] = listings_q3_df.maximum_nights - listings_q3_df.minimum_nights

plt.figure(figsize=(15, 8))
plt.plot(listings_q3_df.maximum_nights, listings_q3_df.minimum_nights, color='b', marker='o', linewidth=0, alpha=0.25)
plt.grid()
# -

# ここから、最低宿泊日数は最大宿泊日数と関係がなく、ほぼ一定な様子がわかる。
# つまり、最大期間が長い物件は、賃貸専用として貸し出しているのではなく、スポットでの利用から長期滞在まで幅広く対応していることがわかる。

# それでは、二個目の質問に答えよう。
# Is there a busy season?
#
# ユーザーの実際の滞在期間が出ていないので正確には言えないが、レビューの個数は一つの目安になると考えられる。（宿泊後レビューをするので、宿泊日数分のタイムラグが生じる）
# また、レビューの個数に一年ごとに周期的なピークが現れていたことから、その近辺が繁忙期だと解釈して差し支えなさそうである。
# 最も大きな繁忙期は9月の頭だとわかったが、いつからいつまでくらいが繁忙期なのだろうか？もう少し詳しくみてみよう

# +
review_q2_df = review_q1_df[review_q1_df.year == 2015]

plt.figure(figsize=(15, 8))
plt.plot(review_q2_df.date, review_q2_df.rolling_mean_30, color='b', linewidth=2.0)
plt.title("Number of reviews by date")
plt.grid()
# -

# ここから、繁忙期は九月の前後1ヶ月と言って良いのではないかと思う。

# +
# create weekday column
review_q2_df["weekday"] = review_q2_df["date"].dt.weekday_name

# boxplot to see price distribution
plt.figure(figsize=(15, 8))
sns.boxplot(x = 'weekday',  y = 'id', data = review_q2_df, palette="Blues", width=0.6)
plt.show()
# -

# ここから、日曜日、月曜日が他の曜日に比べてレビューの投稿数が多いことがわかる。これと前に述べた、週末は価格が高くなることを重ねると、週末は宿泊のニーズが高いことから、週末に利用する人は、１日や2日だけ利用することが多いのかもしれない

# ## Data Preparation

# ここからは、最後の質問：借りられやすい物件に何か傾向はあるのか？に答えるためにlistingデータをクレンジング、加工していく

# todo:
# 借りられやすさの定義
#
# * カレンダーの利用可能日数
# * accommodates
# * host_since

# まず、目的変数である`借りられやすさ`を定義する必要がある。
# これは、こう定義できると思った。
#
# 実際に借りられた回数 / （２０１６〜2017の一年間の利用可能日数 *　（2017 - 物件が公開された年））
#
# 実際に借りられた回数は、その物件が利用可能な日数に比例すると考えられるので、まずそこでスケーリングする必要があり、
# さらに借りられた回数は物件が公開された日が早ければ早いほど多くなると予想できるので、そこでもスケーリングする必要がある。
#
# ここで一つ考慮できていないのは、その物件が実際に借りられた期間だ。借りられた期間が長いほど、借りられた回数は小さくなる。しかしそれを考慮できるデータがなく、また最大宿泊日数と最低宿泊日数は互いに関係がないので、ここは一旦Airbnbの利用者のほとんどは短期での利用だと仮定して進める。

# それでは、まずはnull値のチェックから始める。あまりにもnullが多いカラムは使えないからだ。

prepare_df = seattle_listing.copy(deep=True)

# +
# check null count
df_length = prepare_df.shape[0]

for col in prepare_df.columns:
    null_count = prepare_df[col].isnull().sum()
    if null_count == 0:
        continue
        
    null_ratio = np.round(null_count/df_length * 100, 2)
    print("{} has {} null values ({}%)".format(col, null_count, null_ratio))
# -

# 見ると、90%以上の値がnullのものもあるが、ほとんどのカラムがnull_ratio０〜30%の間に収まっているようだ。そこで、ここでは30%以上nullの値のカラムは分析対象から外すことにする。
# また、目的変数の算出に使うhost_sinceに二個だけnullがあるようだ。これは除く必要がある

# +
# detect need drop columns
drop_cols = [col for col in prepare_df.columns if prepare_df[col].isnull().sum()/df_length >= 0.3]

# drop null
prepare_df.drop(drop_cols, axis=1, inplace=True)
prepare_df.dropna(subset=['host_since'], inplace=True)

# check after
for col in prepare_df.columns:
    null_count = prepare_df[col].isnull().sum()
    if null_count == 0:
        continue
        
    null_ratio = np.round(null_count/df_length * 100, 2)
    print("{} has {} null values ({}%)".format(col, null_count, null_ratio))
# -

# 良い感じに処理できたようだ。次に借りられやすさの特徴にならなそうなカラムを削除していく。今回は自然言語処理は含まないので、コメント系のカラムも削除する

# +
drop_cols = ['listing_url', 'scrape_id', 'last_scraped', 'name', 'summary', 'space', 'description', 'neighborhood_overview',
                'transit', 'medium_url', 'picture_url', 'xl_picture_url', 'host_id', 'host_url', 'host_name', 'host_about', 'host_thumbnail_url',
                'host_picture_url', 'street', 'city', 'state', 'zipcode', 'market', 'smart_location', 'country_code', 'country', 'latitude', 'longitude',
                'calendar_updated', 'calendar_last_scraped', 'first_review', 'last_review', 'amenities', 'host_verifications']

prepare_df.drop(drop_cols, axis=1, inplace=True)
# -

prepare_df.columns

# さらに、単一の値しか持たないカラムは特徴量の意味がないので削除する

# +
drop_cols = []
for col in prepare_df.columns:
    if prepare_df[col].nunique() == 1:
        drop_cols.append(col)
        
prepare_df.drop(drop_cols, axis=1, inplace=True)
prepare_df.columns
# -

# 大方の必要なカラムだけ残せたので、目的変数を作成する

# +
# available days count each listings
listing_avalilable = seattle_calendar.groupby('listing_id')['price'].count().reset_index()
listing_avalilable.columns = ["id", "available_count"]

# merge
prepare_df = prepare_df.merge(listing_avalilable, how='left', on='id')

# create target column
prepare_df['host_since_year'] = pd.to_datetime(prepare_df['host_since']).dt.year
prepare_df["easily_accomodated"] = prepare_df.accommodates / (prepare_df.available_count+1) / (2017 - prepare_df.host_since_year)
# -

# 次に、目的変数に直接関わりのありそうなカラムを削除する（num_reviewなど）

# +
print("Before: {} columns".format(prepare_df.shape[1]))

drop_cols = ['host_since', 'accommodates', 'availability_30', 'availability_60', 'availability_90', 'availability_365',
                'number_of_reviews', 'review_scores_rating', 'available_count', 'reviews_per_month', 'host_since_year', 'review_scores_value']

prepare_df.drop(drop_cols, axis=1, inplace=True)
print("After: {} columns".format(prepare_df.shape[1]))
# -

# さて、ここからはデータをモデルが学習できるような形式に変えていく。
# まずはカテゴリ値をダミー化していく。

# +
# convert true or false value to 1 or 0
dummy_cols = ['host_is_superhost', 'require_guest_phone_verification', 'require_guest_profile_picture', 'instant_bookable', 
              'host_has_profile_pic', 'host_identity_verified', 'is_location_exact']

for col in dummy_cols:
    prepare_df[col] = prepare_df[col].map(lambda x: 1 if x == 't' else 0)

# create dummy valuables
dummy_cols = ['host_location', 'host_neighbourhood', 'neighbourhood', 'neighbourhood_cleansed', 'neighbourhood_group_cleansed',
             'property_type', 'room_type', 'bed_type', 'cancellation_policy', 'host_response_time']

prepare_df = pd.get_dummies(prepare_df, columns=dummy_cols, dummy_na=True)

# +
df_length = prepare_df.shape[0]

for col in prepare_df.columns:
    null_count = prepare_df[col].isnull().sum()
    if null_count == 0:
        continue
        
    null_ratio = np.round(null_count/df_length * 100, 2)
    print("{} has {} null values ({}%)".format(col, null_count, null_ratio))
# -

# 残りのカラムはnullを変換し、一部数値に変えなければならないものもある

prepare_df["is_thumbnail_setted"] = 1 - prepare_df.thumbnail_url.isnull()
prepare_df.drop('thumbnail_url', axis=1, inplace=True)
prepare_df.host_response_rate = prepare_df.host_response_rate.fillna('0%').map(lambda x: float(x[:-1]))
prepare_df.host_acceptance_rate = prepare_df.host_acceptance_rate.fillna('0%').map(lambda x: float(x[:-1]))
prepare_df.bathrooms.fillna(0, inplace=True)
prepare_df.bedrooms.fillna(0, inplace=True)
prepare_df.beds.fillna(0, inplace=True)
prepare_df.cleaning_fee.fillna('$0', inplace=True)
prepare_df.review_scores_accuracy.fillna(0, inplace=True)
prepare_df.review_scores_cleanliness.fillna(0, inplace=True)
prepare_df.review_scores_checkin.fillna(0, inplace=True)
prepare_df.review_scores_communication.fillna(0, inplace=True)
prepare_df.review_scores_location.fillna(0, inplace=True)

# 最後に、数値として入っているべき値が文字列として認識されているので、それを変換する

for col in prepare_df.columns:
    if prepare_df[col].dtypes == 'object':
        print(col)

prepare_df.price = prepare_df.price.map(lambda x: float(x[1:].replace(',', '')))
prepare_df.cleaning_fee = prepare_df.cleaning_fee.map(lambda x: float(x[1:].replace(',', '')))
prepare_df.extra_people = prepare_df.extra_people.map(lambda x: float(x[1:].replace(',', '')))

# ## Modeling, Evaluation

# +
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

X = prepare_df.drop(['id', 'easily_accomodated'], axis=1)
y = prepare_df.easily_accomodated.values

rf = RandomForestRegressor(n_estimators=100, max_depth=5)
scores = cross_val_score(rf, X, y, cv=5)
# -

scores

# 全然予想できていない。なぜだろうか？
# ここで、予測値と実際の値の散布図をプロットしてみよう

rf.fit(X, y)
predictions = rf.predict(X)

# +
plt.figure(figsize=(8, 8))

plt.plot((0, 4), (0, 4), color='gray')
plt.plot(y, predictions, linewidth=0, marker='o', alpha=0.5)
plt.grid()
plt.xlim((-0.2, 4.2))
plt.ylim((-0.2, 4.2))
plt.xlabel("Actual values")
plt.ylabel("Predicted values")
plt.show()
# -

# 見事に当たっていないことがわかる。また、予測が0付近の小さい値に引っ張られていて、大きい方の値に対して小さな予測をしてしまっていることもわかる


