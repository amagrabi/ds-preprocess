#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Helper functions to process DataFrames.

@author: amagrabi

'''

import pandas as pd
import numpy as np


def remove_rows(df, column, value):
    '''Removes rows from a DataFrame with a certain value.
    
    If the value is a list, all single values in the list will be checked.
    The function cannot remove cell entries that are lists themselves.
    
    Args:
        df: Input DataFrame
        column: column in which the value is checked
        value: target value that should be removed from the DataFrame (list, str or numeric)
    
    Returns:
        DataFrame with removed rows.
    
    '''
    if type(value) is list:
        df = df[~(df[column].isin(value))]
    else:
        df = df[df[column]!=value]
    df = df.reset_index(drop=True)
    return df

    
def remove_mcol_duplicates(df, columns, ind_keep=0):
    '''Removes multi-column duplicates from DataFrame rows.
    
    Multi-column duplicates: rows in which the same combination of 
    values occurs in multiple rows.
    
    Args:
        df: Input DataFrame
        column: columns in which values are checked
        ind_keep: index for the multi-column combination that is kept in the DataFrame.
    
    Returns:
        DataFrame with removed duplicates.
    
    '''
    inds_delete = []
    # Count occurrences of img-category pairs
    df_counts = df.groupby(columns).size().reset_index().rename(columns={0:'count'})
    # Get indices img-category pairs that occur multiple times
    inds = df_counts[df_counts['count']>1].index.tolist()
    # Combine duplicate values
    duplicates = []
    duplicate = []
    for ind in inds:
        for col in columns:
            duplicate.append(df_counts.ix[ind,col])
        duplicates.append(duplicate)
        duplicate = []
    # Find duplicates in original df
    for duplicate in duplicates:
        # Get indices of duplicate in original df
        bools = [True]*len(df)
        for i, val in enumerate(duplicate):
            bools = bools & (df[columns[i]]==val)
        # Remove all occurrences of duplicate values except one (index of ind_keep)
        inds = df[bools].index.tolist()
        inds.remove(inds[ind_keep])
        inds_delete.extend(inds)
#    inds_delete = sorted(inds_delete)
    # Remove duplicates from DataFrame
    df = df.drop(inds_delete)
    df = df.reset_index(drop=True)
    return df

    