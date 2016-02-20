from bokeh.plotting import figure, show
import pandas

Monthly_Mah_Returns = pandas.read_pickle('hti-quiet.pickle')
Monthly_Mah_Turbulent_Returns = pandas.read_pickle('hti-turbulent.pickle')

# Define graph
gra = figure(title="Historical Turbulence Index",
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

# Paint quiet graph
gra.segment(x0=Monthly_Mah_Returns.index,
          y0=0,
          x1=Monthly_Mah_Returns.index,
          y1=Monthly_Mah_Returns.values,
          color='#000000',legend='Quiet')

# Paint turbulent graph
gra.segment(x0=Monthly_Mah_Turbulent_Returns.index,
          y0=0,
          x1=Monthly_Mah_Turbulent_Returns.index,
          y1=Monthly_Mah_Turbulent_Returns.values,
          color='#BB0000',legend='Turbulent')

#output_file("hti.html", title="Historical Turbulence Index")
show(gra)