# Airbnb data analysis

This repositry is project of Udacity's Data Scientist Nanodegree.

## packages used

Please see Pipfile's packages section.
https://github.com/shyaginuma/DataAnalysis/blob/master/Pipfile

## Project motivation

Also I wrote in the previous part, this is project of Udacity's Data Scientist Nanodegree.
Here, I want to practice the CRISP-DM process on the real dataset to improve my skills.

## Summary

In the blog.ipynb, I walk through the data with CRISP-DM process. And make three questions about the data and answer these.
These questions and answers are

1. How long is the period available for lending by rooms?

The histogram of maximum nights shows that there are two groups.
One is a listing that can be used at spots such as the maximum number of nights within a week.
The other is a listing that supports a wide range of stay from the super long-term stay of the maximum number of stays for three years or more and the minimum number of nights for around two days to the spot use.

2. Is there a busy season?

The answer is Yes.
Apart from the increase in the number of Airbnb users, there was definitely a timely increase in the number of reviews at the same time each year.
It is thought that it is about one month around early September and overlaps with the summer vacation time. It is important that the number of properties that can be provided at this time exceeds demand.

3. Are there any trends of popular rooms?

I could not derive it from my analysis.
However, I learned that the score improves by logarithmic transformation. I will find time in the future and try to improve.

## Explanation of the file

```txt
Airbnb/
├── extensions
│   └── tools.py
└── notebooks
    ├── EDA(draft).ipynb
    ├── EDA(draft).py
    ├── blog.ipynb
    └── blog.py
```

* extensions: There is the python script which contains useful functions.
* notebooks: There are notebooks and python script that is used in notebooks.
    * EDA(draft): This is my draft file, please don't see because the notebooks is durty.
    * blog: This is also uploaded to [Kaggle Kernel](https://www.kaggle.com/yaginun/crisp-dm-process-on-the-airbnb-dataset). Please see on the kaggle because it is look better on there.

The dataset file is not included in this repositry.
Please download dataset from [kaggle](https://www.kaggle.com/airbnb/seattle).