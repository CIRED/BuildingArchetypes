#%%
import pandas as pd

# import Res-IRF
from project.thermal import conventional_energy_3uses
from project.model import get_config
from project.read_input import read_inputs


list_values = {'Wall': pd.Series([2.5, 1, 0.5, 0.4, 0.2, 0.1]),
               'Roof': pd.Series([2.5, 0.5, 0.3, 0.2, 0.1]),
               'Windows': pd.Series([4.3, 3, 2.6, 1.6, 1.3, 0.8]),
               'Floor': pd.Series([2, 0.9, 0.5, 0.3, 0.2])
               }

list_values_simple = {'Wall': pd.Series([2.5, 1, 0.5, 0.1]),
                      'Roof': pd.Series([2.5, 0.5, 0.1]),
                      'Windows': pd.Series([3, 1.6, 1.3]),
                      'Floor': pd.Series([2, 0.5, 0.2])
                      }

U_VALUES = {'simple': list_values_simple,
            'medium': list_values}

KEY = 'medium'

def replace_closest(data, list_values):
    temp = pd.Series(list_values.values, list_values.values).reindex(data.values, method='nearest')
    temp.index = data.index
    return temp


if __name__ == '__main__':
    #%% md
    # Export from ADEME DPE Database
    #%%
    inputs = read_inputs(get_config())

    df = pd.read_hdf('input/district_level_diagnosis_data_latest.hdf')
    df.set_index('td001_dpe_id', drop=False, inplace=True)
    df.drop_duplicates('td001_dpe_id', inplace=True)
    #%% md

    #%%
    astype = {
        'wall_u_value': 'float',
        'roof_u_value': 'float',
        'floor_u_value': 'float',
        'wall_window_u_value': 'float',
        'main_heating_system_efficiency': 'float',
        'heating_system': 'string'
    }

    df = df.astype(astype)

    replace = {'[1991, 2005]': '[1991, 2005]',
               '[1971, 1990]': '[1971, 1990]',
               '[1946, 1970]': '[1000, 1970]',
              '[1000, 1918]': '[1000, 1970]',
              '[2006, 2100]': '[2006, 2100]',
              '[1919, 1945]': '[1000, 1970]'
    }

    df['construction_period'] = df['construction_year_class'].replace(replace)

    list_values = U_VALUES[KEY]
    df['wall_u_value'] = replace_closest(df['wall_u_value'], list_values['Wall'])
    df['roof_u_value'] = replace_closest(df['roof_u_value'], list_values['Roof'])
    df['floor_u_value'] = replace_closest(df['floor_u_value'], list_values['Floor'])
    df['wall_window_u_value'] = replace_closest(df['wall_window_u_value'], list_values['Windows'])

    ## Heating system
    #%%
    heating_system = ['electric heater', 'electric heat pump', 'fossil gas boiler', 'oil boiler', 'biomass/coal boiler',
                      'urban heat network']
    df = df.loc[df['heating_system'].isin(heating_system), :]
    df['main_heating_system_efficiency_round'] = df['main_heating_system_efficiency'].copy()

    # direct electric
    idx = df[df['heating_system'] == 'electric heater'].index
    df.loc[idx, 'main_heating_system_efficiency_round'] = 0.95

    # fossil boiler
    # TODO use inputs['efficiency']
    efficiency = pd.Series([0.6, 0.76])
    idx = df[df['heating_system'].isin(['fossil gas boiler', 'oil boiler', 'biomass/coal boiler'])].index
    temp = pd.Series(efficiency.values, efficiency.values).reindex(df.loc[idx, 'main_heating_system_efficiency'].values, method='nearest')
    temp.index = idx
    df.loc[idx, 'main_heating_system_efficiency_round'] = temp

    # heat pump
    idx = df[df['heating_system'] == 'electric heat pump'].index
    df.loc[idx, 'main_heating_system_efficiency_round'] = 2.5

    # Calculating energy consumption and DPE based on Res-IRF thermal function
    # %%

    ratio_surface = inputs['ratio_surface']

    # %%
    vars = {'residential_type': 'Housing type',
            'main_heating_energy': 'Energy',
            'wall_u_value': 'Wall',
            'floor_u_value': 'Floor',
            'roof_u_value': 'Roof',
            'wall_window_u_value': 'Windows',
            'main_heating_system_efficiency_round': 'Efficiency',
            'construction_period': 'Period',
            'air_change_rate': 'Air rate',
            'heating_system': 'Heating system'
            }

    buildings = df[vars.keys()].copy()

    replace = {'house': 'Single-family',
               'apartment': 'Multi-family',
               'electricity': 'Electricity',
               'oil': 'Oil fuel',
               'fossil_gas': 'Natural gas',
               'biomass': 'Wood fuel',
               'electric heater': 'Electricity-Performance boiler',
               'fossil gas boiler': 'Natural gas-Standard boiler',
               'oil boiler': 'Oil fuel-Standard boiler',
               'biomass/coal boiler': 'Wood fuel-Standard boiler',
               'electric heat pump': 'Electricity-Heat pump',
               'urban heat network': 'Heating-District heating'
               }

    # %%
    buildings.loc[:, 'residential_type'] = buildings.loc[:, 'residential_type'].replace(replace)
    buildings.loc[:, 'main_heating_energy'] = buildings.loc[:, 'main_heating_energy'].replace(replace)
    buildings.loc[:, 'heating_system'] = buildings.loc[:, 'heating_system'].replace(replace)
    idx = buildings[buildings['main_heating_system_efficiency_round'] == 0.76].index
    buildings.loc[idx, 'heating_system'] = buildings.loc[idx, 'heating_system'].str.replace('Standard', 'Performance')

    buildings['main_heating_energy'] = buildings['main_heating_energy'].astype(str)
    buildings.loc[buildings['heating_system'] == 'Heating-District heating', 'main_heating_energy'] = 'Heating'

    buildings['air_change_rate'] = buildings['air_change_rate'].astype('float')

    buildings = buildings.rename(columns=vars)
    buildings = buildings.set_index(['Housing type', 'Energy', 'Heating system'])

    # %%
    dpe, energy_primary = conventional_energy_3uses(buildings['Wall'], buildings['Floor'], buildings['Roof'], buildings['Windows'], ratio_surface, buildings['Efficiency'], buildings.index, air_rate=buildings['Air rate'])
    result = pd.concat((buildings, dpe.rename('DPE'), energy_primary.rename('PE')), axis=1)
    result.set_index('DPE', append=True, inplace=True)
    # %%
    result.to_csv('output/data_parsed_{}.csv'.format(KEY))
