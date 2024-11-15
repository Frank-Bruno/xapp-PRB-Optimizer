#!/bin/bash

# Using the Pycrate compiler (installed with pip)
pycrate_compiler="/home/openran-br/.local/bin/pycrate_asn1compile.py"
asn1_file="e2sm-kpm-rc.asn"
python3 $pycrate_compiler -i $asn1_file -o e2sm_kpm_rc