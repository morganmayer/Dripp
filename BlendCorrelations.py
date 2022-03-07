#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mayermo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pint
from pint import UnitRegistry

class BlendCorrelations:
    
    def __init__(self, df):

        # property name dictionary
        self.prop_name_dict = {"flashpoint": ["fp_c", "fp_c_coalesced", "Flash_Point"], 
                     "meltingpoint": ["mp_c", "mp_c_coalesced", "Freezing_Point"],
                     "freezingpoint": ["mp_c", "mp_c_coalesced", "Freezing_Point"], 
                     "cn": ["cn", "cn_coalesced"],
                    "boilingpoint_0": ["Boiling_Point_0"], 
                     "boilingpoint_10": ["Boiling_Point_10"],
                    "boilingpoint_50": ["Boiling_Point_50"],
                    "boilingpoint_90": ["Boiling_Point_90"],
                    "boilingpoint_100": ["Boiling_Point_100"],
                    "NHOC": ["Net_Heat_of_Combustion", "NHOC", "NHOC_kjmol25"],
                     "density":["density", "density_rel", "density_20C"],
                               "avg_MW": ["MW", "avg_MW"],
                               "H/C": ["H/C"],
                               "viscosity": ["viscosity_mm2s_20"],
                               "ysi":["ysi"],
                               "refractive_index":["refractive_index_20C"],
                               "n_C": ["id_n_C", "n_C"]
                              }
        
        self.prop_units = {"flashpoint": "degC", # units in csv datasets
                          "meltingpoint": "degC",
                          "freezingpoint": "degC",
                          "cn": "dimensionless",
                          "boilingpoint_0": "degC",
                           "boilingpoint_10": "degC",
                          "boilingpoint_50": "degC",
                          "boilingpoint_90": "degC",
                          "boilingpoint_100": "degC",
                          "Net_Heat_of_Combustion": "kilojoule/mol", # change to use units column
                          "density": "g/cm^3",
                           "avg_MW": "g/mol",
                           "H/C": "dimensionless",
                          "viscosity": "mm^2/s",
                          "ysi": "dimensionless",
                          "H/C": "dimensionless",
                          "n_C": "dimensionless"}
            
        # neat property df
        self.df = df # neat fuel properties
        self.ureg = UnitRegistry()
        self.Q_ = self.ureg.Quantity
        
        # dictionaries of neat properties
        self.pure_freezing_points = self.get_dict_of_pure_components("meltingpoint")
        self.pure_flash_points = self.get_dict_of_pure_components("flashpoint")
        self.pure_cn_dict = self.get_dict_of_pure_components("cn")
        self.pure_density = self.get_dict_of_pure_components("density")
        self.pure_MW = self.get_dict_of_pure_components("avg_MW")
        self.pure_HC = self.get_dict_of_pure_components("H/C")
        self.pure_classes = self.combined_labels("id_class")
        self.pure_n_C = self.get_dict_of_pure_components("n_C")
        
    def remove_zero(self, comp):
        
        comp_ = comp.copy()
        for key, value in comp.items():
            if float(value) == 0.0:
                del comp_[key]
                #break
                
        return comp_
        
    def get_dict_of_pure_components(self, property_):
        
        prop_dict  = {}
        for species in self.df.index.to_list():
            for prop_name_ in self.prop_name_dict[property_]: # go through list of property names
                try: # look in pure species dataset
                    #print(prop_name_, species)
                    prop = float(self.df[str(prop_name_)][str(species)])
                    prop = self.Q_(prop, self.prop_units[property_])
                except: # here if species or prop_name_ not in df
                    pass

            try:
                prop # species, prop_name_ not found in dataset
            except:
                raise ValueError(str(species) + " " + str(property_) + " not in dataset")

            prop_dict[str(species)] = prop # save in single species dict
            prop = None # remove for next time in loop

        return prop_dict
    
    def combined_labels(self, category):
        
        label_dict  = {}
        for species in self.df.index.to_list():
            label = self.df[str(category)][str(species)]
            try:
                label_dict[str(species)] = label
                label = None
            except:
                pass
            
        return label_dict
    
    def n_C_avg(self, comp):
        
        comp = self.remove_zero(comp)
        
        moles_total = sum([self.Q_(comp[species], "dimensionless")*self.pure_density[species]/self.pure_MW[species] for species in comp])
        
        nC = sum([(self.pure_n_C[species]*self.Q_(comp[species], "dimensionless")*self.pure_density[species]/self.pure_MW[species])/moles_total for species in comp])
        
        return nC.to("dimensionless")
    
    def molecule_classes(self, comp):
        
        comp = self.remove_zero(comp)
        classes = ""
        for species in comp:
            classes += str(self.pure_classes[species]) + " "
        
        return classes
    
    def HC_ratio(self, comp):
        
        comp = self.remove_zero(comp)
        moles_total = sum([self.Q_(comp[species], "dimensionless")*self.pure_density[species]/self.pure_MW[species] for species in comp])
        
        HC = sum([(self.pure_HC[species]*self.Q_(comp[species], "dimensionless")*self.pure_density[species]/self.pure_MW[species])/moles_total for species in comp])
        
        return HC.to("dimensionless")
    
    def avg_MW(self, comp):
        
        comp = self.remove_zero(comp)
        
        grams_total = sum([self.Q_(comp[species], "dimensionless")*self.pure_density[species] for species in comp])
        moles_total = sum([self.Q_(comp[species], "dimensionless")*self.pure_density[species]/self.pure_MW[species] for species in comp])
        
        avgMW = grams_total / moles_total
        avgMW = avgMW.to("grams/mol")
        
        return avgMW
    
    def CN(self, comp):
        ''' CN assumed weighted to volume fraction ''' 
        comp = self.remove_zero(comp)
        
        cn = 0
        for species in comp:            
            cn += float(comp[str(species)]) * float(self.pure_cn_dict[str(species)])
        return cn
    
    def flash_point(self, comp):
        #ureg = UnitRegistry()
        #Q_ = ureg.Quantity        
        comp = self.remove_zero(comp)
        
        #uses volume fraction
        I_pure = {}
        for species in comp:
            temp = self.pure_flash_points[species].to('degF')
            I = 10 ** (-6.1188 + ((4345.2)/(temp.magnitude + 383)))
            I_pure[str(species)] = I
            
        I_blend = sum([I_pure[species] * comp[species] for species in I_pure])
            
        fp = (4345.2/(np.log10(I_blend) + 6.1188)) - 383    
        fp = self.Q_(fp, 'degF')        
        return fp.to('degC')
    
    def freezing_point(self, comp):
        #ureg = UnitRegistry()
        #Q_ = ureg.Quantity
        comp = self.remove_zero(comp)
        
        # convert Celsius to Farenheight
        #uses volume fraction for I
        I_pure = {}
        for species in comp:
            temp = self.pure_freezing_points[species].to('kelvin')
            I = float(3.23e-6 * (1.067 ** temp.magnitude)) # typo in Dayton paper
            I_pure[str(species)] = I
        
        I_blend = sum([I_pure[species] * comp[species] for species in I_pure])

        freezing_point = 193.798 + 15.379 * np.log(I_blend) # convert back to Celsius
        freezing_point = self.Q_(freezing_point, 'kelvin')    
        freezing_point = freezing_point.to('degC')
       # print(freezing_point)
        return freezing_point
