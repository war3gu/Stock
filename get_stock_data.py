import os
import datetime
import urllib3
from dateutil.parser import parse
import threading

#assert 'QUANDL_KEY' in os.environ
quandl_api_key = ''

FIELD_DATE = 'trade_date'

import tushare as ts

ts.set_token('51295be6098fe565f6f727019e280ba4821ad5554b551c311bc33ae3')
pro = ts.pro_api()

class nasdaq():
	def __init__(self):
		self.output = './stock_data'
		self.company_list = './companylist.csv'

	def build_url(self, symbol):
		url = symbol
		return url

	def symbols(self):
		symbols = []
		with open(self.company_list, 'r') as f:
			next(f)
			for line in f:
				symbols.append(line.split(',')[0].strip())
		return symbols

def download(i, symbol, url, output):
	df1 = ts.pro_bar(ts_code=url, adj='qfq', start_date="19880101", end_date="20020101")
	df2 = ts.pro_bar(ts_code=url, adj='qfq', start_date="20020101", end_date="20211111")

	df = df2

	if df1 is not None:
		df = df1.append(df2)

	df.sort_values(by=FIELD_DATE, ascending=False, inplace=True)  # inplace is important

	df = df.reset_index(drop=True)

	print(df)
	fullPath = os.path.join(output, symbol)
	df.to_csv('{}.csv'.format(fullPath))
	print('download')

'''
def download(i, symbol, url, output):
	print('Downloading {} {}'.format(symbol, i))
	try:
		response = urllib3.urlopen(url)
		quotes = response.read()
		lines = quotes.strip().split('\n')
		with open(os.path.join(output, symbol), 'w') as f:
			for i, line in enumerate(lines):
				f.write(line + '\n')
	except Exception as e:
		print('Failed to download {}'.format(symbol))
		print(e)
'''

def download_all():
	if not os.path.exists('./stock_data'):
	    os.makedirs('./stock_data')

	nas = nasdaq()
	for i, symbol in enumerate(nas.symbols()):
		url = nas.build_url(symbol)
		download(i, symbol, url, nas.output)

if __name__ == '__main__':
	download_all()