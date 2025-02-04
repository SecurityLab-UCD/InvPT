#!/bin/bash
# get_data.sh 
# Downloads and augments BCB data
while getopts "" opt; do
    case $opt in
	\?)
	    echo "Invalid option: -$OPTARG" >&2
	    ;;
    esac
done
if [[ "$#" -ne 0 ]]; then
    echo "Illegal number of parameters" >&2
    exit 1
fi

echo '= Downloading original dataset ='
cd dataset
wget https://raw.githubusercontent.com/microsoft/CodeXGLUE/refs/heads/main/Code-Code/Clone-detection-BigCloneBench/dataset/data.jsonl
mv data.jsonl original_data.jsonl
cd ..

echo '= Cleaning previous files ='
cd dataset
rm -rf augmented_data.jsonl augtest.txt
cd augmentation
rm -rf artifacts/*
mkdir -p artifacts/original artifacts/augmented

# Augment dataset

echo '= Augmenting dataset ='
python3 preprocess.py \
    --data_jsonl ../original_data.jsonl \
    --extracted_java_path artifacts/original \
    --metadata_jsonl_path artifacts/metadata.jsonl \
    --java_colname func \
    --id_colname idx \
    --no-drop_duplicates
./augment.sh 0 1 2 3 6 7
python3 postprocess.py \
    artifacts/metadata.jsonl \
    artifacts/augmented_only.jsonl \
    artifacts/augmented \
    --id_colname idx
python3 merge.py \
    --augmented_path artifacts/augmented_only.jsonl \
    --original_path ../original_data.jsonl \
    --output_path ../augmented_data.jsonl
cp ../augmented_data.jsonl ../data.jsonl

# Augment test split

echo '= Augmenting test split ='
python3 aug_test.py \
    --augmented_jsonl ../augmented_data.jsonl \
    --test_txt ../test.txt \
    --output_txt ../augtest.txt
cd ..
