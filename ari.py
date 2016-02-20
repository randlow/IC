from bokeh.plotting import figure, show
import pandas

AR = pandas.read_pickle('ari.pickle')

# Define graph
gra = figure(title="Absorption Ratio Index",
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
gra.segment(x0=AR.index,
            y0=0,
            x1=AR.index,
            y1=AR.values,
            color='#000000')

#output_file("ari.html", title="Absorption Ratio Index")
show(gra)