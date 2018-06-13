from datetime import date
import numpy as np
import pandas as pd

def convertDate(row, column, nan_value='NaT'):
    """
        Function to convert string date to integer and hour in seconds
        column  : name of columns, string
        row     : dataframe row
    """
    if row[column] != nan_value:
        temp = row[column].split()
        hour = temp[1].split(':')
        date = temp[0].split('-')
        hour_int = int(hour[0]) * 3600 + int(hour[1]) * 60 + int(hour[2])
        date_int = int(date[0]) * 3600 + int(date[1]) * 60 + int(date[2])
    else:
        hour_int = np.nan
        date_int = np.nan
        
    return pd.Series([date_int, hour_int])

def convertInt(row, column, nan_value='-'):
    """
        Function to convert string time to seconds
        column  : name of columns, string
        row     : dataframe row
    """
    if row[column] != nan_value:    
        H = True if 'H' in row[column] else False
        M = True if 'M' in row[column] else False

        time = row[column].replace('H', ' ').replace('M', ' ').split()
        if H and M:
            days = int(time[0]) * 3600 + int(time[1]) * 60 + int(time[2])
        elif H and not M:
            days = int(time[0]) * 3600 + int(time[1]) * 60
        elif not H and M:
            days = int(time[0]) * 60 + int(time[1])
        else:
            days = int(time[0])
    else:
        days = np.nan
        
    return days

def apply_cats(df, trn):
    """
        Changes any columns of strings in df (DataFrame) into categorical variables
        using trn (DataFrame) as a template for the category codes (inplace).
    """
    for n,c in df.items():
        if (n in trn.columns) and (trn[n].dtype.name=='category'):
            df[n] = pd.Categorical(c, categories=trn[n].cat.categories, ordered=True)           
    
def date_diff(d1, d2):
    """
        Days between d1 and d2, expressed as integers
    """
    return (date(d1 // 10000, (d1 // 100) % 100, d1 % 100) - \
            date(d2 // 10000, (d2 // 100) % 100, d2 % 100)).days
    
def days_since(day_df, trades, keys, nan_date=20170701):
    """
        Get number of days between last *keys* and day_df date
    """
    last_trades = pd.Series(trades.drop_duplicates(keys, keep='first') \
            .set_index(keys)['TradeDateKey']).to_dict()
    return day_df.apply(lambda r: date_diff(r['TradeDateKey'],
            last_trades.get(tuple(r[k] for k in keys) if len(keys) > 1 else r[keys[0]],
            nan_date)), axis=1)
    
# Count without considering weekdays
def add_datediffs(day_df, trades):
    """
        Adds datediffs features to a dataset (representing a single day/week)
        from the information of trades. Adds #DaysSinceBuySell (the corresponding
        one) #DaysSinceTransaction (either buy or sell), #DaysSinceCustomerActivity
        (since last customer interaction) #DaysSinceBondActivity (since last bond
        interaction)
    """
    trades = trades[trades.CustomerInterest == 1]
    date = sorted(day_df['TradeDateKey'].unique())[0]
    trades = trades[trades.TradeDateKey < date]
    trades = trades.sort_values('TradeDateKey', ascending=False)
    
    day_df['DaysSinceBuySell'] = days_since(day_df, trades, 
                                            ['CustomerIdx', 'IsinIdx', 'BuySell'])
    day_df['DaysSinceCustomerBuySell'] = days_since(day_df, trades, 
                                            ['CustomerIdx', 'BuySell'])
    day_df['DaysSinceTransaction'] = days_since(day_df, trades, 
                                            ['CustomerIdx', 'IsinIdx'])
    day_df['DaysSinceCustomerActivity'] = days_since(day_df, trades, ['CustomerIdx'])
    day_df['DaysSinceBondActivity'] = days_since(day_df, trades, ['IsinIdx'])
    day_df['DaysSinceBondBuySell'] = days_since(day_df, trades, ['IsinIdx', 'BuySell'])

def days_count(day_df, trades, keys):
    '''Get frequency *keys* in historical trades before day_df'''
    day_counter = trades.groupby(keys).size().to_dict()
    return day_df.apply(lambda r: \
            day_counter.get(tuple(r[k] for k in keys) if len(keys) > 1 else r[keys[0]], 
            0), axis=1)
    
def add_dayscount(day_df, trades):
    '''Adds dayscount features to a dataset (representing a single day/week)
    from the information of trades'''
    trades = trades[trades.CustomerInterest == 1]
    date = sorted(day_df['TradeDateKey'].unique())[0]
    trades = trades[trades.TradeDateKey < date]
    
    day_df['DaysCountBuySell'] = days_count(day_df, trades,
                                    ['CustomerIdx', 'IsinIdx', 'BuySell'])
    day_df['DaysCountCustomerBuySell'] = days_count(day_df, trades,
                                    ['CustomerIdx', 'BuySell'])
    day_df['DaysCountTransaction'] = days_count(day_df, trades,
                                    ['CustomerIdx', 'IsinIdx'])
    day_df['DaysCountCustomerActivity'] = days_count(day_df, trades, ['CustomerIdx'])
    day_df['DaysCountBondActivity'] = days_count(day_df, trades, ['IsinIdx'])
    day_df['DaysCountBondBuySell'] = days_count(day_df, trades, ['IsinIdx', 'BuySell'])
    