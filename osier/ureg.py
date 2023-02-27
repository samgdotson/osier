from unyt import UnitRegistry, Unit
from unyt.dimensions import dimensionless, mass
import unyt as u
from unyt import kg

kgs_per_tonne = 1e3
tonnes_per_megatonne = 1e6
dollars_per_Mdollars = 1e6

reg = UnitRegistry()
reg.add("dollar", base_value=1.0, dimensions=dimensionless,
         tex_repr=r"\$", prefixable=False)
reg.add("Mdollars", base_value=dollars_per_Mdollars, dimensions=dimensionless,
         tex_repr=r"M\$", prefixable=False)
# reg.add("Tonne", base_value=kgs_per_tonne, dimensions=mass, tex_repr="\rm{T}")
# reg.add("MTonne", base_value=tonnes_per_megatonne, dimensions=mass,tex_repr="\rm{MT}")

dollar = Unit('dollar', registry=reg)
Mdollars = Unit('Mdollars', registry=reg)

u.define_unit("tonne", value=kgs_per_tonne*kg, tex_repr="\rm{T}")
u.define_unit("Mtonne", value=tonnes_per_megatonne*u.tonne, tex_repr="\rm{T}")

data = 1*Mdollars
data2 = 1*dollar

print(data/data2)
print(data2, data)
print(data/(3*u.Mtonne))