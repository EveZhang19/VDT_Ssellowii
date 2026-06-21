#!/bin/bash

# ==============================================================================
# RNA-Seq Pipeline: Trimmomatic -> Kallisto
# Usage: bash run_rnaseq_pipeline_preprocessing.sh <INPUT_RAW_DIR> <OUTPUT_BASE_DIR> <REFERENCE_FASTA>
# ==============================================================================

# 1. check paras
if [ "$#" -ne 3 ]; then
    echo "Error: Missing arguments!"
    echo "Usage: bash $0 <INPUT_RAW_DIR> <OUTPUT_BASE_DIR> <REFERENCE_FASTA>"
    exit 1
fi

INPUT_RAW_DIR="$1"
OUTPUT_BASE_DIR="$2"
REF_FASTA="$3"

TRIMMOMATIC_JAR="~/tools/Trimmomatic-0.39/trimmomatic-0.39.jar"
ADAPTER_FILE="~/tools/Trimmomatic-0.39/TruSeq3-PE.fa"

TRIM_OUT_DIR="$OUTPUT_BASE_DIR/trim"
EXP_OUT_DIR="$OUTPUT_BASE_DIR/exp"
LOG_FILE="$OUTPUT_BASE_DIR/pipeline_run.log"

mkdir -p "$TRIM_OUT_DIR"
mkdir -p "$EXP_OUT_DIR"

LEADING=3
TRAILING=3
MINLEN=36
ILLUMINACLIP="ILLUMINACLIP:$ADAPTER_FILE:2:30:10:2:True"

echo "Starting Pipeline at $(date)" | tee "$LOG_FILE"
echo "Input Data: $INPUT_RAW_DIR" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

# ==============================================================================
# STEP 1: TRIMMOMATIC
# ==============================================================================
echo "STEP 1: Running Trimmomatic..." | tee -a "$LOG_FILE"

for sample_dir in "$INPUT_RAW_DIR"/*; do
    if [ -d "$sample_dir" ]; then
        sample_name=$(basename "$sample_dir")

        fq1="$sample_dir/${sample_name}_1.fq.gz"
        fq2="$sample_dir/${sample_name}_2.fq.gz"

        if [ -f "$fq1" ] && [ -f "$fq2" ]; then
            echo "   -> Trimming $sample_name..." | tee -a "$LOG_FILE"

            if [ -f "$TRIM_OUT_DIR/${sample_name}_forward_paired.fq.gz" ]; then
                echo "      [SKIPPED] Trimmed files already exist for $sample_name." | tee -a "$LOG_FILE"
                continue
            fi

            java -jar "$TRIMMOMATIC_JAR" PE \
                "$fq1" "$fq2" \
                "$TRIM_OUT_DIR/${sample_name}_forward_paired.fq.gz" "$TRIM_OUT_DIR/${sample_name}_forward_unpaired.fq.gz" \
                "$TRIM_OUT_DIR/${sample_name}_reverse_paired.fq.gz" "$TRIM_OUT_DIR/${sample_name}_reverse_unpaired.fq.gz" \
                "$ILLUMINACLIP" LEADING:"$LEADING" TRAILING:"$TRAILING" MINLEN:"$MINLEN" \
                2>&1 | tee -a "$LOG_FILE"
        else
            echo "   Warning: Paired files not found for $sample_name, skipping..." | tee -a "$LOG_FILE"
        fi
    fi
done

# ==============================================================================
# STEP 2: KALLISTO
# ==============================================================================
echo "" | tee -a "$LOG_FILE"
echo "STEP 2: Running Kallisto..." | tee -a "$LOG_FILE"

fasta_name=$(basename "$REF_FASTA" .fasta)
IDX_FILE="$EXP_OUT_DIR/${fasta_name}.idx"

echo "   -> Building Kallisto Index..." | tee -a "$LOG_FILE"
kallisto index -i "$IDX_FILE" "$REF_FASTA" 2>&1 | tee -a "$LOG_FILE"

for fq1 in "$TRIM_OUT_DIR"/*_forward_paired.fq.gz; do

    fq2="${fq1/_forward_paired/_reverse_paired}"
    sample_name=$(basename "$fq1" _forward_paired.fq.gz)

    if [[ -f "$fq1" && -f "$fq2" ]]; then
        echo "   -> Quantifying $sample_name..." | tee -a "$LOG_FILE"
        kallisto quant -i "$IDX_FILE" -o "$EXP_OUT_DIR/${sample_name}" "$fq1" "$fq2" 2>&1 | tee -a "$LOG_FILE"
    else
        echo "   Skipping $sample_name due to missing paired trimmed files." | tee -a "$LOG_FILE"
    fi
done

echo "========================================" | tee -a "$LOG_FILE"
echo "Preprocessing Pipeline completed successfully at $(date)!" | tee -a "$LOG_FILE"

# ==============================================================================
# STEP 3: AUTOMATIC QC EXTRACTION (DYNAMIC INTEGRATION)
# ==============================================================================
echo "" | tee -a "$LOG_FILE"
echo "STEP 3: Extracting QC Summary Metrics..." | tee -a "$LOG_FILE"

python3 - <<EOF
import re
import csv

log_file = "${LOG_FILE}"
output_csv = "${OUTPUT_BASE_DIR}/QC_summary.csv"

stats = {}
current_sample = None

with open(log_file, 'r') as f:
    for line in f:
        m_sample = re.search(r'-> (?:Trimming|Quantifying) ([\w_]+)\.\.\.', line)
        if m_sample:
            current_sample = m_sample.group(1)
            if current_sample not in stats:
                stats[current_sample] = {'Raw_Reads': 'NA', 'Survived_%': 'NA', 'Mapped_%': 'NA', 'Fragment_Length': 'NA'}

        if not current_sample:
            continue

        if "Input Read Pairs:" in line:
            m_trim = re.search(r'Input Read Pairs: (\d+) Both Surviving: \d+ \(([\d\.]+)\%\)', line)
            if m_trim:
                stats[current_sample]['Raw_Reads'] = m_trim.group(1)
                stats[current_sample]['Survived_%'] = m_trim.group(2)

        if "reads pseudoaligned" in line:
            m_map = re.search(r'processed ([\d,]+) reads, ([\d,]+) reads pseudoaligned', line)
            if m_map:
                proc = float(m_map.group(1).replace(',', ''))
                pseudo = float(m_map.group(2).replace(',', ''))
                stats[current_sample]['Mapped_%'] = str(round((pseudo / proc) * 100, 2))

        if "estimated average fragment length:" in line:
            m_frag = re.search(r'estimated average fragment length: ([\d\.]+)', line)
            if m_frag:
                stats[current_sample]['Fragment_Length'] = m_frag.group(1)

with open(output_csv, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Sample', 'Raw_Reads', 'Survived_%', 'Mapped_%', 'Fragment_Length'])
    for sample in sorted(stats.keys(), key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else x):
        writer.writerow([sample, stats[sample]['Raw_Reads'], stats[sample]['Survived_%'], stats[sample]['Mapped_%'], stats[sample]['Fragment_Length']])

print("   -> QC_summary.csv has been successfully generated in the output directory!")
EOF

# ==============================================================================
# STEP 4: AUTOMATIC PACKAGING FOR LOCAL DOWNLOAD
# ==============================================================================
echo "" | tee -a "$LOG_FILE"
echo "STEP 4: Packaging results for local transfer..." | tee -a "$LOG_FILE"

OUTPUT_DIR_NAME=$(basename "$OUTPUT_BASE_DIR")
TAR_FILE_NAME="${OUTPUT_DIR_NAME}_kallisto_results.tar.gz"
FINAL_TAR_PATH="$OUTPUT_BASE_DIR/$TAR_FILE_NAME"

cd "$(dirname "$OUTPUT_BASE_DIR")"

#  exp folder + QC_summary.csv
tar -czf "$FINAL_TAR_PATH" "$OUTPUT_DIR_NAME/exp" "$OUTPUT_DIR_NAME/QC_summary.csv"

echo "   -> Final package created successfully!" | tee -a "$LOG_FILE"
echo "   -> Location: $FINAL_TAR_PATH" | tee -a "$LOG_FILE"






               
