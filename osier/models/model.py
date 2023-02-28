from osier import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
from sklearn.preprocessing import normalize
from pymoo.core.problem import Problem
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from pymoo.termination.ftol import MultiObjectiveSpaceTermination
from pymoo.visualization.scatter import Scatter
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.termination.robust import RobustTermination
from pymoo.core.parameters import set_params, hierarchical

from pygenesys.utils.tsprocess import aggregate

from copy import deepcopy
import unyt as u
import time
import dill

LARGE_NUMBER = 1e40

class OsierModel(object):
    """
    This class constitutes a complete energy system model with
    capacity expansion from start to finish. Users can specify
    parameters for the capacity expansion model, the genetic 
    algorithm, and run the model. 

    Parameters
    ----------
    technology_list : list of :class:`osier.Technology` objects
        Defines the technologies used in the model and the number
        of decision variables.
    demand : :class:`numpy.ndarray`
        The demand curve that needs to be met by the technology mix.
    objectives : list of str or functions
        Specifies the number and type of objectives. A list of strings
        must correspond to preset objective functions. Users may optionally
        write their own functions and pass them to `osier` as items in the
        list.
    constraints : dictionary of string : float or function : float
        Specifies the number and type of constraints. String key names
        must correspond to preset constraints functions. Users may optionally
        write their own functions and pass them to `osier` as keys in the
        list. The values must be numerical and represent the value that the function
        should not exceed. See notes for more information about constraints.
    prm : Optional, float
        The "planning reserve margin" (`prm`) specifies the amount
        of excess capacity needed to meet reliability standards.
        See :attr:`capacity_requirement`. Default is 0.0.
    solar : Optional, :class:`numpy.ndarray`
        The curve that defines the solar power provided at each time
        step. Automatically normalized with the infinity norm
        (i.e. divided by the maximum value).
    wind : Optional, :class:`numpy.ndarray`
        The curve that defines the wind power provided at each time
        step. Automatically normalized with the infinity norm
        (i.e. divided by the maximum value).
    power_units : str, :class:`unyt.unit_object`
        Specifies the units for the power demand. The default is :attr:`MW`.
        Can be overridden by specifying a unit with the value.
    penalty : Optional, float
        The penalty for infeasible solutions. If a particular set
        produces an infeasible solution for the :class:`osier.DispatchModel`,
        the corresponding objectives take on this value.
    curtailment : boolean
        Indicates if the model should enable a curtailment option.
    allow_blackout : boolean
        If True, a "reliability" technology is added to the dispatch model that will
        fulfill the mismatch in supply and demand. This reliability technology
        has a variable cost of 1e4 $/MWh. The value must be higher than the
        variable cost of any other technology to prevent a pathological
        preference for blackouts. Default is True.
    algorithm : str
        Specifies the genetic algorithm to be used. Default is UNSGA3.

    """

    def __init__(self,
                 technology_list,
                 demand,
                 objectives,
                 constraints={},
                 solar=None,
                 wind=None,
                 prm=0.0,
                 penalty=LARGE_NUMBER,
                 power_units=u.MW,
                 curtailment=True,
                 allow_blackout=False,
                 algorithm='UNSGA3',
                 **kwargs) -> None:
        pass