# RNA-Seq Preprocessing & Quantification Pipeline

## Overview
This automated pipeline is designed for high-throughput RNA-seq data preprocessing and transcript-level quantification. It integrates quality control (Trimmomatic), ultra-fast pseudoalignment (Kallisto), automatic QC metric extraction, and result packaging into a single, robust Bash script. 

Designed for fault tolerance, it features a built-in checkpoint mechanism that automatically skips completed steps during pipeline restarts.

---

## Quick Start / Usage

Execute the pipeline using `nohup` to ensure it runs safely in the background:

```bash
nohup bash run_rnaseq_pipeline_preprocessing.sh \
  <INPUT_RAW_DIR> \
  <OUTPUT_BASE_DIR> \
  <TRANSCRIPTOME_FASTA> \
  > <OUTPUT_BASE_DIR>/pipeline.log 2>&1 &
```

### Argument Definitions:
1. **`INPUT_RAW_DIR`**: The directory containing raw Paired-End (PE) sequencing files. 
   * *Requirement*: Files must be formatted as `${sample_name}_1.fq.gz` and `${sample_name}_2.fq.gz`.
2. **`OUTPUT_BASE_DIR`**: The target directory where all processed data, logs, and final tarballs will be stored. (Will be created automatically if it does not exist).
3. **`TRANSCRIPTOME_FASTA`**: The reference transcriptome FASTA file. The script will automatically build a Kallisto .idx file from this FASTA during STEP 2.

---

## 📂 Output Directory Structure

Upon successful completion, the `OUTPUT_BASE_DIR` will be structured as follows:

```text
OUTPUT_BASE_DIR/
├── trim/                           # Quality-controlled FASTQ files
│   ├── XX_1_forward_paired.fq.gz   # [Keep] Cleaned reads (Read 1)
│   ├── XX_1_reverse_paired.fq.gz   # [Keep] Cleaned reads (Read 2)
│   ├── XX_1_forward_unpaired.fq.gz # [Discard] Orphan reads
│   └── ...
├── exp/                            # Kallisto quantification results
│   ├── XX_1/
│   │   ├── abundance.tsv           # [Core] Transcript-level counts & TPM
│   │   ├── abundance.h5
│   │   └── run_info.json
│   └── ...
├── QC_summary.csv                  # Aggregated QC metrics for all samples
├── pipeline_run.log                # Master execution log
└── [ProjectName]_kallisto_results.tar.gz  # Final package for local transfer
```

---

## ⚙️ Core Parameters & Internal Logic

### STEP 1: Quality Control (Trimmomatic)
The pipeline utilizes Trimmomatic (v0.39) in Paired-End (PE) mode with the following parameters:
* `ILLUMINACLIP:TruSeq3-PE.fa:2:30:10:2:True`: Detects and removes Illumina adapters. Allows 2 seed mismatches, palindrome clip threshold of 30, and simple clip threshold of 10.
* `LEADING:3` & `TRAILING:3`: Trims low-quality bases (Phred score < 3) from the 5' and 3' ends of the reads.
* `MINLEN:36`: Absolute length threshold. Any read shorter than 36 bp after trimming is dropped to prevent ambiguous multi-mapping.
* *Checkpoint Logic*: If `${sample_name}_forward_paired.fq.gz` is detected in the `trim/` directory, this step is automatically bypassed.

### STEP 2: Quantification (Kallisto)
* **Dynamic Indexing**: Automatically builds a Kallisto index (`.idx`) from the provided `TRANSCRIPTOME_FASTA` before quantification.
* Employs Kallisto in `quant` mode to calculate transcript abundances.
* Automatically estimates average fragment length for each paired-end sample based on input data.
* Employs the Expectation-Maximization (EM) algorithm to resolve multi-mapping probabilities.

### STEP 3: Automated QC Extraction
A dynamic Python module scans the master log to extract critical biological viability metrics, generating `QC_summary.csv`:
* **Raw_Reads**: Total sequenced fragment pairs.
* **Survived_%**: Percentage of read pairs passing Trimmomatic QC (indicates initial sequencing quality).
* **Mapped_%**: Pseudoalignment rate (vital metric for evaluating reference genome quality and potential sample contamination).
* **Fragment_Length**: Estimated library insert size.

### STEP 4: Tarball Packaging
For seamless local transfer via `scp`, the script securely packages the `exp/` directory and `QC_summary.csv` into a `.tar.gz` archive within the output directory, ensuring data integrity during transfer.
