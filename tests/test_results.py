from osier.equations import (annualized_capital_cost, annualized_fixed_cost, total_cost, 
                             annual_emission, per_unit_capacity, per_unit_energy, volatility)
from osier import Technology, DispatchModel
from osier.results import get_objective_names, rename_duplicates
import numpy as np
import pytest
import sys
import functools

N = 5
duplicate_list = ['a']*N + ['b']*N
print(duplicate_list)
print(rename_duplicates(duplicate_list))

objective_list = ['pct_renewable', 
                  annualized_capital_cost,
                  annualized_fixed_cost,
                  total_cost,
                  functools.partial(annual_emission, emission='co2'),
                  functools.partial(annual_emission, emission='so2'),
                  functools.partial(volatility, m=3, delay=100),
                  functools.partial(per_unit_capacity, attribute='land_use'),
                  functools.partial(per_unit_capacity, attribute='employment'),
                  functools.partial(per_unit_energy, attribute='water_use'),
                  functools.partial(per_unit_energy, attribute='death_rate'),
                  functools.partial(volatility, m=4, delay=60),
                  ]

def test_rename_duplicates():
    expected = [f'a_{i}' for i in range(N)] + [f'b_{i}' for i in range(N)] 

    actual = rename_duplicates(duplicate_list)

    assert expected == actual


def test_get_objective_names():
    expected = ['pct_renewable', 
                'annualized_capital_cost',
                'annualized_fixed_cost',
                'total_cost',
                'co2',
                'so2',
                'volatility_0',
                'land_use',
                'employment',
                'water_use',
                'death_rate',
                'volatility_1',]
    
    actual = get_objective_names(objective_list=objective_list)

    assert expected == actual