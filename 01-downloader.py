import datetime
#from pandas_datareader import data # for newer python versions
from pandas.io import data
import pandas as pd
import io
import urllib
import urllib2 

"""
Imported data with available start date (* = latest available start date = start date for all)
Germany     ^GDAXI   26/11/1990
US          ^GSPC    3/1/1950
Japan       ^N225    4/1/1984
UK          ^FTSE    3/1/1984
France      ^FCHI    1/3/1990
Italy       EWI      1/4/1996    *
Canada      ^GSPTSE  29/6/1979
Australia   ^AORD    3/8/1984
"""
start = datetime.datetime(1996, 4, 1)

# Get data from yahoo (daily interval)
f = data.DataReader(['^GDAXI', '^GSPC', '^N225', '^FTSE', '^FCHI', '^GSPTSE', '^AORD'], 'yahoo', start)
#f.to_excel('01-downloader-a.xls')
#print('[TEMP: File 01-downloader-a.xls saved with all received data from yahoo]')

# Only get adjusted close rates
f = f['Adj Close']
#f.to_excel('01-downloader-b.xls')
#print('[TEMP: File 01-downloader-b.xls saved with only adjusted close data]')

# Interpolate NaN values
f=f.interpolate()
#f.to_excel('01-downloader-c.xls')
#print('[TEMP: File 01-downloader-b.xls saved with interpolation where value are NaN]')

# Save to pickle file
f.to_pickle('sourcedata.pickle')


#### PROBIT section
start = datetime.datetime(1977, 2, 15)
# Get data from yahoo (daily interval)
f = data.DataReader(['^GSPC', '^TYX'], 'yahoo', start)
# Only get adjusted close rates
f = f['Adj Close']
# Drop non existent-files
f=f.dropna()
# Save to pickle file
f.to_pickle('probit_portfolio.pickle')

# Get recession values
data = {
    'form[native_frequency]': 'Monthly',
    'form[frequency]': 'Monthly',
    'form[obs_start_date]': '1961-02-01',
    'form[obs_end_date]': '2014-06-01',
    'form[file_format]': 'csv',
    'form[download_data_1]': 'Download Data',
    'form[units]': 'lin',
}

req = urllib2.Request(url="https://research.stlouisfed.org/fred2/series/OECDNMERECM/downloaddata",
                      data=urllib.urlencode(data), 
                      headers={"Content-type": "application/x-www-form-urlencoded"}) 
response = urllib2.urlopen(req)

recession_values = pd.read_csv(io.BytesIO(response.read()), index_col=0, sep=',')
recession_values.to_pickle("recession_values.pickle")
