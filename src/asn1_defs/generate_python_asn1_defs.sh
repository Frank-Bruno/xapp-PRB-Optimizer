#!/bin/bash

# Using the Pycrate compiler (installed with pip)
pycrate_compiler="/home/openran-br/.local/bin/pycrate_asn1compile.py"
asn1_files="E2SM-KPM-v01.00.asn"
output_file="e2sm_kpm_1_0"
python3 $pycrate_compiler -i $asn1_files -o $output_file