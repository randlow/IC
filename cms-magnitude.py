from bokeh.plotting import figure, show
import pandas

Mag_sur = pandas.read_pickle('cms-magnitude.pickle')

# Define graph
gra = figure(title="Magnitude Surprise",
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
gra.segment(x0=Mag_sur.index,
            y0=0,
            x1=Mag_sur.index,
            y1=Mag_sur.values,
            color='#0000AA')

#output_file("cms-correlation.html", title="Correlation Surprise")
show(gra)