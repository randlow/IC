from bokeh.plotting import figure, show
import pandas

Corr_sur = pandas.read_pickle('cms-correlation.pickle')

# Define graph
gra = figure(title="Correlation Surprise",
             x_axis_label="Year",
             y_axis_label="Index",
             x_axis_type="datetime",
             plot_width=800,
             plot_height=300,
             logo=None)

# Set the footer labels (including zoomed-state)
gra.below[0].formatter.formats = dict(years=['%Y'],
                                      months=['%b %Y'],
                                      days=['%d %b %Y'])

# Paint graph
gra.segment(x0=Corr_sur.index,
            y0=0,
            x1=Corr_sur.index,
            y1=Corr_sur.values,
            color='#000000')

#output_file("cms-correlation.html", title="Correlation Surprise")
show(gra)