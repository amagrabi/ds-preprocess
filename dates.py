#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Helper function to process datetimes.
    
@author: amagrabi


"""

import re
import dateutil.parser


def date_to_us(date_str):
    '''Convert date to US format.
    
    Args:
        date_str: Input date.
        
    Returns:
        Converted date.
    '''
    date = dateutil.parser.parse(date_str)
    dateformat = '%d-%m-%y %I:%M:%S.%f %p'
    return date.strftime(dateformat)