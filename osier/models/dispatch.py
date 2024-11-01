import pandas as pd
import numpy as np
import xarray as xr
from linopy import Model
from unyt import unyt_array, hr, MW, kW, GW, kWh
import itertools as it
from osier import Technology
from osier.technology import _validate_quantity, _validate_unit
from osier.utils import synchronize_units
import warnings
import logging
import time


_freq_opts = {'D': 'day',
              'H': 'hour',
              'S': 'second',
              'T': 'minute'}
LARGE_NUMBER = 1e20
MEDIUM_NUMBER = 1e10
BLACKOUT_COST = 10 * (1 / kWh)  # M$/kWh


curtailment_tech = Technology(technology_name='Curtailment',
                              technology_type='removal',
                              dispatchable=True,
                              capacity=MEDIUM_NUMBER)
reliability_tech = Technology(technology_name='LoadLoss',
                              technology_type='matching',
                              dispatchable=True,
                              fuel_cost=BLACKOUT_COST,
                              capacity=MEDIUM_NUMBER)


class DispatchModel():
    """
    The :class:`DispatchModel` class creates and solves a basic dispatch
    model from the perspective of a "grid operator." The model uses
    `pyomo <https://pyomo.readthedocs.io/en/stable/index.html>`_
    to create and solve a linear programming model.
    The mathematical formulation for this problem is:

    Minimize


    .. math::
        \\text{C}_{\\text{total}} = \\sum_t^T \\sum_u^U
        \\left[c^{\\text{fuel}}_{u,t} + c^{\\text{om,var}}_{u,t}\\right]x_{u,t}

    Such that,

    1. The generation meets demand within a user-specified tolerance
    (undersupply and oversupply)

    .. math::
        \\sum_u^Ux_{u,t} &\\geq \\left(1-\\text{undersupply}\\right)\\text{D}_t
        \\ \\forall \\ t \\in T

        \\sum_u^Ux_{u,t} &\\leq \\left(1+\\text{oversupply}\\right)\\text{D}_t
        \\ \\forall \\ t \\in T

    2. A technology's generation (:math:`x_u`) does not exceed its capacity to
    generate at any time, :math:`t`.

    .. math::
        x_{u,t} \\leq \\textbf{CAP}_{u}\\Delta t \\ \\forall \\ u,t \\in U,T

    3. Technologies may not exceed their ramp up rate,

    .. math::
        \\frac{x_{r,t} - x_{r,t-1}}{\\Delta t} = \\Delta P_{r,t} \\leq
        (\\text{ramp up})\\textbf{CAP}_u\\Delta t \\ \\forall \\ r,t
        \\in R \\subset U, T

    or ramp down rate,

    .. math::
        \\frac{x_{r,t} - x_{r,t-1}}{\\Delta t} = \\Delta P_{r,t} \\leq
        -(\\text{ramp down})\\textbf{CAP}_u\\Delta t \\ \\forall \\ r,t
        \\in R \\subset U, T .

    Parameters
    ----------
    technology_list : list of :class:`osier.Technology`
        The list of :class:`Technology` objects to dispatch -- i.e. decide
        how much energy each technology should produce.
    net_demand : list, :class:`numpy.ndarray`, :class:`unyt.array.unyt_array`, :class:`pandas.DataFrame`
        The remaining energy demand to be fulfilled by the technologies in
        :attr:`technology_list`. The `values` of an object passed as
        `net_demand` are used to create a supply constraint. See
        :attr:`oversupply` and :attr:`undersupply`.
        If a :class:`pandas.DataFrame` is passed, :mod:`osier` will try
        inferring a `time_delta` from the dataframe index. Otherwise, the
        :attr:`time_delta` must be passed or the default is used.
    time_delta : str, :class:`unyt.unyt_quantity`, float, int
        Specifies the amount of time between two time slices. The default is
        one hour. Can be overridden by specifying a unit with the value. For
        example:

        >>> time_delta = "5 minutes"
        >>> from unyt import days
        >>> time_delta = 30*days

        would both work.
    power_units : str, :class:`unyt.unit_object`
        Specifies the units for the power demand. The default is :attr:`MW`.
        Can be overridden by specifying a unit with the value.
    solver : str
        Indicates which solver to use. May require separate installation.
        Accepts: ['cplex', 'cbc']. Other solvers will be added in the future.
    lower_bound : float
        The minimum amount of energy each technology can produce per time
        period. Default is 0.0.
    oversupply : float
        The amount of allowed oversupply as a percentage of demand.
        Default is 0.0 (no oversupply allowed).
    undersupply : float
        The amount of allowed undersupply as a percentage of demand.
        Default is 0.0 (no undersupply allowed).
    verbosity : Optional, int
        Sets the logging level for the simulation. Accepts `logging.LEVEL`
        or integer where LEVEL is {10:DEBUG, 20:INFO, 30:WARNING, 40:ERROR, 50:CRITICAL}.
    curtailment : boolean
        Indicates if the model should enable a curtailment option.
    allow_blackout : boolean
        If True, a "reliability" technology is added to the model that will
        fulfill the mismatch in supply and demand. This reliability technology
        has a variable cost of 1e4 $/MWh. The value must be higher than the
        variable cost of any other technology to prevent a pathological
        preference for blackouts. Default is False.

    Attributes
    ----------
    model : :class:`pyomo.environ.ConcreteModel`
        The :mod:`pyomo` model class that converts python code into a
        set of linear equations.
    objective : float
        The result of the model's objective function. Only instantiated
        after :meth:`DispatchModel.solve()` is called.
    n_timesteps : int
        The number of timesteps in the model.
    upper_bound : float
        The upper bound for all decision variables. Chosen to be equal
        to the maximum capacity of all technologies in :attr:`tech_set`.
    storage_upper_bound : float
        The upper bound for storage decision variables.
    penalty : float
        The penalty applied to the objective function to eliminate
        simultaneous charging and discharging. Users may need to tune
        this parameter. Default is 1e-4.
    model_initialized : bool
        Indicates whether :attr:`DispatchModel.model` has been populated
        with equations yet. This is set to ``True`` after
        :meth:`DispatchModel._write_model_equations()` has been called.
    indices : list of tuples
        The list of tuples representing the product of the
        :attr:`tech_set` and :attr:`time_set` attributes.
    tech_set : list of str
        A list of the unique technology names in the simulation.
    capacity_dict : dict
        A dictionary of {name : capacity} pairs.
    efficiency_dict : dict
        A dictionary of {name : efficiency} pairs.
    time_set : iterable
        The result of ``range(self.n_timesteps)``.
    cost_params : list
        The set of cost parameters for each technology. Corresponds to
        a list of values. Size is equal to the product of the number of
        timesteps and the number of technologies in the model.
    ramp_up_params : list
        The set of ramp_up parameters. Only initialized if there is a
        ramping technology.
    ramp_down_params : list
        The set of ramp_down parameters. Only initialized if there is a
        ramping technology.
    ramping_techs : list
        The subset of :attr:`tech_set` containing ramping technologies.

    Notes
    -----
    1. Technically, :attr:`solver` will accept any solver that :class:`pyomo`
    can use. We only list two solvers because those are the only solvers
    in the :mod:`osier` test suite.

    2. The default value for :attr:`time_delta` in :attr:`__init__` is
    `None`. This is replaced by a setter method in :class:`Dispatch`.

    3. This formulation uses a :attr:`penalty` parameter  to prevent unphysical
    behavior because it preserves the problem's linearity. Some formulations
    may use a binary variable to prevent simultaneous charging and discharing.
    However, this changes the problem to a mixed-integer linear program which
    requires a more sophisticated solver such as ``gurobi``.

    4. :meth:`_write_model_equations` may be called before :meth:`solve` if
    users wish to add their own constraints or parameters to the problem.
    """

    def __init__(self,
                 technology_list,
                 net_demand,
                 time_delta=None,
                 solver='cbc',
                 lower_bound=0.0,
                 oversupply=0.0,
                 undersupply=0.0,
                 verbosity=50,
                 penalty=1e-4,
                 power_units=MW,
                 curtailment=True,
                 allow_blackout=False,
                 **kwargs):
        self.net_demand = net_demand
        self.time_delta = time_delta
        self.lower_bound = lower_bound
        self.oversupply = oversupply
        self.undersupply = undersupply
        self.penalty = penalty
        self.model = Model()
        self.solver = solver
        self.results = None
        self.objective = None
        self.model_initialized = False
        self.verbosity = verbosity
        if self.verbosity > 10:
            self.verbose = False
        else: 
            self.verbose = True
        self.curtailment = curtailment
        self.allow_blackout = allow_blackout

        if isinstance(net_demand, unyt_array):
            self.power_units = net_demand.units
        else:
            self.power_units = power_units

        if ((self.curtailment) and (self.allow_blackout)):
            sync_list = technology_list + [curtailment_tech, reliability_tech]
        elif self.curtailment:
            sync_list = technology_list + [curtailment_tech]
        elif self.allow_blackout:
            sync_list = technology_list + [reliability_tech]
        else:
            sync_list = technology_list

        self.technology_list = synchronize_units(
            sync_list,
            unit_power=self.power_units,
            unit_time=self.time_delta.units)
        
        self.technology_list = [t for t in self.technology_list if t.dispatchable]

        logging.basicConfig(level=verbosity, format='%(message)s')


    @property
    def time_delta(self):
        return self._time_delta

    @time_delta.setter
    def time_delta(self, value):
        if value:
            valid_quantity = _validate_quantity(value, dimension='time')
            self._time_delta = valid_quantity
        else:
            if isinstance(self.net_demand, pd.DataFrame):
                try:
                    freq_list = list(self.net_demand.index.inferred_freq)
                    freq_key = freq_list[-1]
                    try:
                        value = float(freq_list[0])
                    except ValueError:
                        warnings.warn((f"Could not convert value "
                                       f"{freq_list[0]} to float. "
                                       "Setting to 1.0."),
                                      UserWarning)
                        value = 1.0
                    self._time_delta = _validate_quantity(
                        f"{value} {_freq_opts[freq_key]}", dimension='time')
                except KeyError:
                    warnings.warn(
                        (f"Could not infer time delta with freq {freq_key} "
                         "from pandas dataframe. Setting delta to 1 hour."),
                        UserWarning)
                    self._time_delta = 1 * hr
            else:
                self._time_delta = 1 * hr

    @property
    def power_units(self):
        return self._power_units

    @power_units.setter
    def power_units(self, value):
        if value:
            valid_quantity = _validate_unit(value, dimension='power')
            self._power_units = valid_quantity
        else:
            warnings.warn(f"Could not infer power units. Unit set to MW.")
            self._power_units = MW

    @property
    def n_timesteps(self):
        return len(self.net_demand)

    @property
    def tech_set(self):
        tech_names = [t.technology_name for t in self.technology_list]
        return tech_names

    @property
    def capacity_dict(self):
        capacity_set = unyt_array([t.capacity for t in self.technology_list])
        return dict(zip(self.tech_set, capacity_set))

    @property
    def efficiency_dict(self):
        efficiency_set = [t.efficiency for t in self.technology_list]
        return dict(zip(self.tech_set, efficiency_set))

    @property
    def time_set(self):
        return range(self.n_timesteps)

    @property
    def indices(self):
        return list(it.product(self.tech_set, self.time_set))

    @property
    def cost_params(self):
        v_costs = np.array([
            (t.variable_cost_ts(self.n_timesteps))
            for t in self.technology_list
        ]).flatten()
        return dict(zip(self.indices, v_costs))

    @property
    def storage_techs(self):
        return [t.technology_name
                for t in self.technology_list
                if hasattr(t, "storage_capacity")]

    @property
    def storage_upper_bound(self):
        caps = unyt_array([t.storage_capacity
                           for t in self.technology_list
                           if hasattr(t, "storage_capacity")])
        return caps.max().to_value()

    @property
    def max_storage_params(self):
        storage_dict = {t.technology_name: t.storage_capacity.to_value()
                        for t in self.technology_list
                        if hasattr(t, "storage_capacity")}
        return storage_dict

    @property
    def initial_storage_params(self):
        storage_dict = {t.technology_name: t.initial_storage.to_value()
                        for t in self.technology_list
                        if hasattr(t, "initial_storage")}
        return storage_dict

    @property
    def ramping_techs(self):
        return [t.technology_name
                for t in self.technology_list
                if hasattr(t, 'ramp_up')]

    @property
    def ramp_up_params(self):
        rates_dict = {
            t.technology_name: (
                t.ramp_up *
                self.time_delta).to_value() for t in self.technology_list
            if hasattr(t, 'ramp_up')}
        return rates_dict

    @property
    def ramp_down_params(self):
        rates_dict = {
            t.technology_name: (
                t.ramp_down *
                self.time_delta).to_value() for t in self.technology_list
            if hasattr(t, 'ramp_up')}
        return rates_dict

    @property
    def upper_bound(self):
        caps = unyt_array([t.capacity for t in self.technology_list])
        return caps.max().to_value()


    def _create_demand_param(self):
        self.demand = dict(zip(self.time_set, np.array(self.net_demand)))


    def _create_cost_param(self):
        self.variable_cost = self.cost_params


    def _create_model_parameters(self):
        self.model.add_parameters(
            ramp_up=self.ramp_up_params,
            ramp_down=self.ramp_down_params,
            storage_capacity=self.max_storage_params,
            initial_storage=self.initial_storage_params
        )


    # def _create_model_variables(self):
    #     breakpoint()
    #     self.model.add_variables(
    #         'x', dims=(self.tech_set, self.time_set),
    #         lower=self.lower_bound, upper=self.upper_bound,
    #     )

    #     if len(self.storage_techs) > 0:
    #         self.model.add_variables(
    #             'storage_level', dims=(self.storage_techs, self.time_set),
    #             lower=self.lower_bound, upper=self.storage_upper_bound,
    #         )
    #         self.model.add_variables(
    #             'charge', dims=(self.storage_techs, self.time_set),
    #             lower=self.lower_bound, upper=self.storage_upper_bound,
    #         )


    def _create_model_variables(self):
        lower_bounds = xr.DataArray(self.lower_bound, 
                                    dims=('tech_set', 'time_set'), 
                                    coords={'tech_set': self.tech_set, 
                                            'time_set': self.time_set})
        upper_bounds = xr.DataArray(self.upper_bound, 
                                    dims=('tech_set', 'time_set'), 
                                    coords={'tech_set': self.tech_set, 
                                            'time_set': self.time_set})
        breakpoint()
        self.model.add_variables(name='x', 
                                #  dims=(self.tech_set, self.time_set), 
                                 lower=lower_bounds, upper=upper_bounds)

        if len(self.storage_techs) > 0:
            storage_lower_bounds = xr.DataArray(self.lower_bound, 
                                                dims=('storage_techs', 'time_set'), 
                                                coords={'storage_techs': self.storage_techs, 
                                                        'time_set': self.time_set})
            storage_upper_bounds = xr.DataArray(self.storage_upper_bound, 
                                                dims=('storage_techs', 'time_set'), 
                                                coords={'storage_techs': self.storage_techs, 
                                                        'time_set': self.time_set})

            self.model.add_variables(name='storage_level', 
                                     lower=storage_lower_bounds, 
                                     upper=storage_upper_bounds)
            self.model.add_variables(name='charge', 
                                     lower=storage_lower_bounds, 
                                     upper=storage_upper_bounds)



    def _objective_function(self):
        expr = sum(self.variable_cost[g, t] * self.model.variables['x'][g, t]
                for g in self.tech_set for t in self.time_set)
        if len(self.storage_techs) > 0:
            expr += sum(self.model.variables['x'][s, t] + self.model.variables['charge'][s, t]
                        for s in self.storage_techs
                        for t in self.time_set) * self.penalty
        self.model.objective = self.model.minimize(expr)


    def _supply_constraints(self):
        for t in self.time_set:
            generation = sum(self.model.x[g, t] for g in self.tech_set if g != 'Curtailment')
            if self.curtailment:
                generation -= self.model.x['Curtailment', t]
            if len(self.storage_techs) > 0:
                generation -= sum(self.model.charge[s, t] for s in self.storage_techs)
            over_demand = self.demand[t] * (1 + self.oversupply)
            under_demand = self.demand[t] * (1 - self.undersupply)
            self.model.add_constraints(
                oversupply=(generation <= over_demand),
                undersupply=(generation >= under_demand)
            )

    def _generation_constraint(self):
        for g in self.tech_set:
            unit_capacity = (self.capacity_dict[g] * self.time_delta).to_value()
            for t in self.time_set:
                unit_generation = self.model['x'][g, t]
                self.model.add_constraints(
                    gen_limit=(unit_generation <= unit_capacity)
                )

    def _ramping_constraints(self):
        for r in self.ramping_techs:
            ramp_up = self.model['ramp_up'][r]
            ramp_down = self.model['ramp_down'][r]
            for t in self.time_set:
                if t != self.time_set[0]:
                    t_prev = self.time_set[t - 1]
                    previous_gen = self.model.x[r, t_prev]
                    current_gen = self.model.x[r, t]
                    delta_power = (current_gen - previous_gen) / self.time_delta.to_value()
                    self.model.add_constraints(
                        ramp_up_limit=(delta_power <= ramp_up),
                        ramp_down_limit=(delta_power >= -ramp_down)
                    )

    def _storage_constraints(self):
        for s in self.storage_techs:
            efficiency = self.efficiency_dict[s]
            storage_cap = self.model.storage_capacity[s]
            unit_capacity = (self.capacity_dict[s] * self.time_delta).to_value()
            initial_storage = self.model.initial_storage[s]
            for t in self.time_set:
                self.model.add_constraints(
                    charge_rate_limit=(self.model.charge[s, t] <= unit_capacity)
                )
                if t == self.time_set[0]:
                    self.model.add_constraints(
                        set_storage=(self.model.storage_level[s, t] == initial_storage)
                    )
                self.model.add_constraints(
                    discharge_limit=(self.model.x[s, t] <= initial_storage),
                    charge_limit=(self.model.charge[s, t] <= storage_cap - initial_storage)
                )
                if t != self.time_set[0]:
                    t_prev = self.time_set[t - 1]
                    previous_storage = self.model.storage_level[s, t_prev]
                    current_discharge = self.model.x[s, t]
                    current_charge = self.model.charge[s, t]
                    self.model.add_constraints(
                        set_storage=(self.model.storage_level[s, t] == previous_storage + np.sqrt(efficiency) * current_charge - np.sqrt(efficiency) * current_discharge),
                        charge_limit=(np.sqrt(efficiency) * current_charge <= storage_cap - previous_storage),
                        discharge_limit=(current_discharge <= previous_storage),
                        storage_limit=(self.model.storage_level[s, t] <= storage_cap)
                    )

    def _write_model_equations(self):

        # self._create_model_indices()
        # self._create_demand_param()
        # self._create_cost_param()
        # self._create_model_parameters()
        self._create_demand_param()
        self._create_cost_param()
        self._create_model_variables()
        self._objective_function()
        self._supply_constraints()
        self._generation_constraint()
        if len(self.ramping_techs) > 0:
            # self._create_ramping_params()
            self._ramping_constraints()
        if len(self.storage_techs) > 0:
            # self._create_init_storage_params()
            # self._create_max_storage_params()
            self._storage_constraints()
        self.model_initialized = True

    def _format_results(self):
        df = pd.DataFrame(index=self.time_set)
        for g in self.tech_set:
            if g == 'Curtailment':
                df[g] = [-1 * self.model.x[g, t].value for t in self.time_set]
            else:
                df[g] = [self.model.x[g, t].value for t in self.time_set]
        if len(self.storage_techs) > 0:
            for s in self.storage_techs:
                df[f"{s}_charge"] = [-1 * self.model.charge[s, t].value for t in self.time_set]
                df[f"{s}_level"] = [self.model.storage_level[s, t].value for t in self.time_set]
        df["Cost"] = np.array([sum(self.variable_cost[g, t] * self.model.x[g, t].value 
                                   for g in self.tech_set) for t in self.time_set])
        return df

    def solve(self, solver=None):
        """
        Executes the model solve. Model equations are written at the
        time the method is called.
        Parameters
        ----------
        solver : str
            Indicates which solver to use. If no solver is specified,
            the default :attr:`DispatchModel.solver` attribute is used.
            Default is ['cbc'].
        """
        if not self.model_initialized:
            self._write_model_equations()
        if solver:
            self.solver = solver
        results = self.model.solve(solver=self.solver, verbose=self.verbose)
        try:
            self.objective = self.model.objective_value
        except ValueError:
            if self.verbosity <= 30:
                warnings.warn(f"Infeasible or no solution. Objective set to {LARGE_NUMBER}")
            self.objective = LARGE_NUMBER
        try:
            self.results = self._format_results()
        except TypeError:
            self.results = None

