

def transform_calendar(calendar_df):

    calendar_df['price'] = calendar_df['price'].map(lambda x: float(x[1:]))

    calendar_df['date'] = pd.to_datetime(calendar_df['date'])
    calendar_df['weekday'] = calendar_df.date.dt.weekday
    calendar_df['weekday_name'] = calendar_df.date.dt.weekday_name
