from bokeh.plotting import figure, show
import pandas

PF = pandas.read_pickle('pf.pickle')

from bokeh.plotting import figure, output_file, show
gra = figure(title="Probit Forecast",
           x_axis_label="Year",
           y_axis_label="Index",
           x_axis_type="datetime",
           y_range=[-4, 4],
           plot_width=900,
           plot_height=300,
           logo=None)

# Set the footer labels (including zoomed-state)
gra.below[0].formatter.formats = dict(years=['%Y'],
                                    months=['%b %Y'],
                                    days=['%d %b %Y'])

# Paint graph
gra.segment(x0=PF.index,
          y0=0,
          x1=PF.index,
          y1=PF.values.flatten(),
          color='#000000')

#output_file("pf.html", title="Probit Forecast")
show(gra)
