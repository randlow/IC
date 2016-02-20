import bokeh.resources as bokeh_res
import urllib
import os

cdn_urls = bokeh_res._get_cdn_urls()
for obj in cdn_urls:
   for url in cdn_urls[obj]:
      urllib.urlretrieve(url, 'build/html/' + os.path.basename(url))


import fileinput
import glob
import sys

for line in fileinput.input(glob.glob('build/html/bokeh-plot-*.js'), inplace=True):
   sys.stdout.write(line.replace('http://cdn.pydata.org/bokeh/release/', ''))
   