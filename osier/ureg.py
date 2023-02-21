from unyt import UnitRegistry, Unit
from unyt import day
from unyt.dimensions import dimensionless
reg = UnitRegistry()
reg.add("dollar", base_value=1.0, dimensions=dimensionless,
         tex_repr=r"\$")
reg.add("Mdollar", base_value=1e6, dimensions=dimensionless,
         tex_repr=r"\$")

u = Unit('dollar', registry=reg)
u2 = Unit('Mdollar', registry=reg)
data = 3*u2
print((data/day).units)