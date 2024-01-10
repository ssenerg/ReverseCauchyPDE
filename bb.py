from ANNPDE.PDE.shapes import LineShape


a = LineShape(
    seed=1,
    n=256,
    start_point=0,
    end_point=1,
    cross_sample_generate=1,
    even_sample=True
)
a.plot()