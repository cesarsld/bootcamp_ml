#!/bin/bash
VERSION=$(python -V 2>&1)
if echo $VERSION | grep -q 3.7; then
	echo 3.7 detected
else
	echo 3.7 not detected
	echo installing python 3.7
	mkdir tmp_miniconda
	cd tmp_miniconda
	curl -o Miniconda3_installer.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
	bash Miniconda3_installer.sh -b -p /goinfre/miniconda -f
	rm -rf tmp_miniconda
	export PATH="/goinfre/miniconda/bin:$PATH"
	echo 'python 3.7 installed !'
fi