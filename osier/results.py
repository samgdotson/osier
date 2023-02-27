import pandas as pd
import numpy as np
import dill
from types import FunctionType, BuiltinFunctionType
from functools import partial

_func_types = (FunctionType, BuiltinFunctionType, partial)


def rename_duplicates(names_list):
    """
    This function renames duplicate names by appending a number
    to the end of each duplicate.
    
    Parameters
    ----------
    names_list : list of str
        List of names
    
    Returns
    -------
    names_list : list of str
        A list of names without duplicates
    """

    duplicates = {i:names_list.count(i) 
                  for i in names_list 
                  if names_list.count(i) > 1}
    


    name_array = np.array(names_list)
    print('name_array:', name_array)

    for dup, count in duplicates.items():
        print('duplicate:',dup)
        locs = np.where(name_array == dup)
        print(locs)
        for i,loc in enumerate(locs[0]):
            print(i, names_list[loc])
            names_list[loc] = f'{dup}_{i}' 
            print(i, names_list[loc])
    
    return names_list




def get_objective_names(objective_list):
    """
    This function generates a list of function names
    used in an Osier model.

    
    Parameters
    ----------
    objective_list : :class:`osier.CapacityExpansion`
        A list of objectives in :class:`osier`
    
    Returns
    -------
    objective_names : list of str
        The list of objective names.

    Notes
    -----
    This function tries to generate a unique list of objective names.
    Since some objectives are formulated generically and may be passed 
    with :func:`functools.partial`, the name may be ambiguous. To avoid
    repeated names, the keyword argument will be used instead. If there
    are multiple keyword arguments, then the naming is still ambiguous
    and :func:`get_objective_names` reverts to using the name of the function.

    If there are duplicate entries a numerical value will be appended to 
    each duplicate name.

    Examples
    --------

    >>> objectives = [func]

    """
    objective_names = []

    for obj in objective_list:
        if isinstance(obj, str):
            objective_names.append(obj)
        elif isinstance(obj, _func_types):
            if (isinstance(obj, partial)) and (len(obj.keywords.items()) > 1):
                objective_names.append(obj.func.__name__)
            elif (isinstance(obj, partial)) and (len(obj.keywords.items()) == 1):
                objective_names.append(str(list(obj.keywords.values())[0]))
            else:
                objective_names.append(obj.__name__)
    
    if (len(set(objective_names)) != len(objective_names)):
        objective_names = rename_duplicates(objective_names)

    return objective_names