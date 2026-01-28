---
layout: post
title: "Scalable Genomics Pipelines in Python"
date: 2025-01-15
description: "A beginner-friendly guide to representing DNA, analyzing sequencing data, and building genomics pipelines in Python."
---

# Scalable Genomics Pipelines in Python

When I started working with genomic data, I was overwhelmed. The file formats were unfamiliar, the data was massive, and the biology was complex. But at its core, genomics is just string processing at scale. Once I realized that, everything clicked.

This guide breaks down the fundamentals: how to represent DNA in Python, how to work with sequencing data, and how to build pipelines that can handle real-world datasets. Whether you're a programmer curious about bioinformatics or a biologist learning to code, this is where I wish I had started.

## Part 1: Representing DNA in Python

DNA is just a string of four letters: A, T, G, and C. These represent the four nucleotide bases that make up your genetic code. In Python, we can represent DNA the same way we represent any text.

```python
# A simple DNA sequence
dna = "ATGCGATCGATCGATCG"

# DNA is just a string, so all string operations work
print(len(dna))           # 17 bases
print(dna[0:3])           # "ATG" - first three bases (a codon)
print(dna.count("G"))     # Count how many G's
```

### The Complement Rule

DNA has a beautiful property: A always pairs with T, and G always pairs with C. This is called base pairing, and it's how the double helix holds together. We can write a function to find the complement of any sequence:

```python
def complement(dna):
    """Return the complementary DNA strand."""
    pairs = {
        'A': 'T',
        'T': 'A',
        'G': 'C',
        'C': 'G'
    }
    return ''.join(pairs[base] for base in dna)

# Example
sequence = "ATGC"
print(complement(sequence))  # "TACG"
```

### Reverse Complement

In genomics, we often need the reverse complement. DNA strands run in opposite directions (called 5' to 3'), so to get the sequence of the opposite strand, we complement AND reverse:

```python
def reverse_complement(dna):
    """Return the reverse complement of a DNA strand."""
    return complement(dna)[::-1]

# Example
sequence = "ATGCGA"
print(reverse_complement(sequence))  # "TCGCAT"
```

This function is incredibly useful. When sequencing reads align to the reverse strand of a reference genome, you need the reverse complement to compare them properly.

### From DNA to Protein

DNA gets transcribed to RNA (replace T with U), then translated to protein. Here's a simple translator:

```python
# The genetic code: three RNA bases = one amino acid
CODON_TABLE = {
    'AUG': 'M',  # Methionine (start codon)
    'UUU': 'F', 'UUC': 'F',  # Phenylalanine
    'UUA': 'L', 'UUG': 'L',  # Leucine
    'UCU': 'S', 'UCC': 'S', 'UCA': 'S', 'UCG': 'S',  # Serine
    'UAU': 'Y', 'UAC': 'Y',  # Tyrosine
    'UGU': 'C', 'UGC': 'C',  # Cysteine
    'UGG': 'W',  # Tryptophan
    'UAA': '*', 'UAG': '*', 'UGA': '*',  # Stop codons
    # ... (full table has 64 entries)
}

def transcribe(dna):
    """Convert DNA to RNA."""
    return dna.replace('T', 'U')

def translate(rna):
    """Convert RNA to protein sequence."""
    protein = []
    for i in range(0, len(rna) - 2, 3):
        codon = rna[i:i+3]
        amino_acid = CODON_TABLE.get(codon, '?')
        if amino_acid == '*':  # Stop codon
            break
        protein.append(amino_acid)
    return ''.join(protein)
```

## Part 2: Reading Sequencing Data

Real genomic data comes in specific file formats. The two most common are FASTA (sequences only) and FASTQ (sequences plus quality scores).

### FASTA Format

FASTA is simple: a header line starting with `>`, followed by the sequence:

```
>sequence_1 Human chromosome 1 fragment
ATGCGATCGATCGATCGATCG
ATCGATCGATCGATCGATCGA
>sequence_2 Another fragment
GGGGCCCCAAAATTTT
```

Here's how to read it:

```python
def read_fasta(filename):
    """Parse a FASTA file into a dictionary of sequences."""
    sequences = {}
    current_name = None
    current_seq = []

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                # Save the previous sequence
                if current_name:
                    sequences[current_name] = ''.join(current_seq)
                # Start a new sequence
                current_name = line[1:].split()[0]  # Get ID (first word)
                current_seq = []
            else:
                current_seq.append(line)

        # Don't forget the last sequence
        if current_name:
            sequences[current_name] = ''.join(current_seq)

    return sequences

# Usage
genome = read_fasta("reference.fasta")
print(f"Loaded {len(genome)} sequences")
```

### FASTQ Format

FASTQ adds quality scores. Each read has four lines: header, sequence, separator (+), and quality scores:

```
@read_001
ATGCGATCGATCGATCG
+
IIIIIIIIIIIIIIIII
```

The quality scores are ASCII characters encoding the confidence of each base call. Higher is better.

```python
def read_fastq(filename):
    """Parse a FASTQ file, yielding (name, sequence, quality) tuples."""
    with open(filename, 'r') as f:
        while True:
            header = f.readline().strip()
            if not header:
                break
            sequence = f.readline().strip()
            f.readline()  # Skip the '+' line
            quality = f.readline().strip()

            name = header[1:]  # Remove the '@'
            yield name, sequence, quality

def quality_to_scores(quality_string):
    """Convert quality string to numeric scores (Phred+33 encoding)."""
    return [ord(char) - 33 for char in quality_string]

# Usage
for name, seq, qual in read_fastq("reads.fastq"):
    scores = quality_to_scores(qual)
    avg_quality = sum(scores) / len(scores)
    print(f"{name}: {len(seq)} bases, avg quality {avg_quality:.1f}")
```

## Part 3: Split Read Analysis

Here's where it gets interesting. When we sequence DNA, we don't get the whole genome at once. We get millions of short "reads" (typically 100-300 bases) that we need to piece back together.

Sometimes a single read spans a junction, like where two exons meet in RNA, or where a structural variant joins two distant parts of the genome. These are called split reads, and analyzing them reveals important biology.

### What Are Split Reads?

Imagine you're reading a book, but someone cut out a section and taped two non-adjacent pages together. A split read is like a sentence that spans that cut: part of it matches one location, and part matches another.

```python
class SplitRead:
    """Represents a read that aligns to multiple locations."""

    def __init__(self, name, sequence, alignments):
        self.name = name
        self.sequence = sequence
        self.alignments = alignments  # List of (chrom, start, end, segment)

    def is_split(self):
        """Check if this read has multiple alignment segments."""
        return len(self.alignments) > 1

    def get_junction(self):
        """Return the junction point if this is a split read."""
        if not self.is_split():
            return None

        # Sort alignments by position in the read
        sorted_alns = sorted(self.alignments, key=lambda x: x[3])

        # The junction is between the first and second segments
        first = sorted_alns[0]
        second = sorted_alns[1]

        return {
            'left_chrom': first[0],
            'left_pos': first[2],  # End of first segment
            'right_chrom': second[0],
            'right_pos': second[1],  # Start of second segment
        }
```

### Detecting Split Reads

When aligning reads to a reference genome, tools like BWA or STAR can produce split alignments. Here's a simplified detector:

```python
def find_split_reads(reads, reference, min_segment=20):
    """
    Find reads that don't align contiguously.

    This is a simplified version. Real aligners use sophisticated
    algorithms like Smith-Waterman with affine gap penalties.
    """
    split_reads = []

    for name, sequence, quality in reads:
        # Try to align the full read
        full_alignment = align(sequence, reference)

        if full_alignment.has_large_gap():
            # This read might be split
            # Try aligning each half separately
            mid = len(sequence) // 2
            left_half = sequence[:mid]
            right_half = sequence[mid:]

            left_aln = align(left_half, reference)
            right_aln = align(right_half, reference)

            if left_aln.is_good() and right_aln.is_good():
                split = SplitRead(
                    name=name,
                    sequence=sequence,
                    alignments=[
                        (left_aln.chrom, left_aln.start, left_aln.end, 'left'),
                        (right_aln.chrom, right_aln.start, right_aln.end, 'right')
                    ]
                )
                split_reads.append(split)

    return split_reads
```

### Why Split Reads Matter

Split reads help us detect:

1. **Exon-exon junctions** in RNA-seq (how genes are spliced)
2. **Structural variants** like translocations and large deletions
3. **Fusion genes** where two genes are joined together (important in cancer)

```python
def classify_split_read(split_read):
    """Classify what kind of event a split read represents."""
    junction = split_read.get_junction()

    if junction['left_chrom'] != junction['right_chrom']:
        return 'translocation'  # Spans two chromosomes

    distance = abs(junction['right_pos'] - junction['left_pos'])

    if distance < 1000:
        return 'small_deletion'
    elif distance < 100000:
        return 'exon_junction'  # Likely RNA splicing
    else:
        return 'large_structural_variant'
```

## Part 4: Building a Pipeline

Now let's put it all together. A genomics pipeline takes raw sequencing data through multiple processing steps:

```python
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

class GenomicsPipeline:
    """A simple genomics analysis pipeline."""

    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def quality_filter(self, reads, min_quality=20):
        """Filter reads by average quality score."""
        passed = []
        failed = 0

        for name, seq, qual in reads:
            scores = quality_to_scores(qual)
            if sum(scores) / len(scores) >= min_quality:
                passed.append((name, seq, qual))
            else:
                failed += 1

        print(f"Quality filter: {len(passed)} passed, {failed} failed")
        return passed

    def trim_adapters(self, reads, adapter="AGATCGGAAGAG"):
        """Remove adapter sequences from the end of reads."""
        trimmed = []

        for name, seq, qual in reads:
            # Simple adapter finding (real tools use approximate matching)
            idx = seq.find(adapter)
            if idx != -1:
                seq = seq[:idx]
                qual = qual[:idx]
            trimmed.append((name, seq, qual))

        return trimmed

    def run(self, fastq_file, reference_file):
        """Run the full pipeline."""
        print(f"Starting pipeline for {fastq_file}")

        # Step 1: Load reference genome
        print("Loading reference genome...")
        reference = read_fasta(reference_file)

        # Step 2: Read and filter raw data
        print("Reading and filtering reads...")
        reads = list(read_fastq(fastq_file))
        reads = self.quality_filter(reads)
        reads = self.trim_adapters(reads)

        # Step 3: Align reads (simplified)
        print("Aligning reads...")
        alignments = self.align_reads(reads, reference)

        # Step 4: Find split reads
        print("Detecting split reads...")
        split_reads = find_split_reads(reads, reference)

        # Step 5: Generate report
        print("Generating report...")
        self.write_report(alignments, split_reads)

        print("Pipeline complete!")
        return alignments, split_reads
```

## Part 5: Scaling Up

The simple pipeline above works for small datasets, but real genomics data is massive. A single sequencing run can produce hundreds of gigabytes. Here's how to scale:

### Process in Chunks

Don't load everything into memory at once:

```python
def process_fastq_chunked(filename, chunk_size=10000):
    """Process a large FASTQ file in chunks."""
    chunk = []

    for read in read_fastq(filename):
        chunk.append(read)

        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []

    if chunk:  # Don't forget the last partial chunk
        yield chunk

# Usage
for chunk in process_fastq_chunked("huge_file.fastq"):
    results = process_chunk(chunk)
    save_results(results)
```

### Parallelize with Multiple Cores

Genomics is embarrassingly parallel. Each read can be processed independently:

```python
from multiprocessing import Pool

def process_read(read):
    """Process a single read (runs in parallel)."""
    name, seq, qual = read
    # Quality check, trim, align, etc.
    return result

def parallel_process(reads, num_workers=8):
    """Process reads in parallel."""
    with Pool(num_workers) as pool:
        results = pool.map(process_read, reads)
    return results
```

### Use NumPy for Speed

String operations in pure Python are slow. NumPy can help:

```python
import numpy as np

def fast_quality_filter(sequences, qualities, min_score=20):
    """Fast quality filtering using NumPy."""
    # Convert quality strings to numeric arrays
    qual_arrays = [
        np.array([ord(c) - 33 for c in q])
        for q in qualities
    ]

    # Calculate mean quality for each read
    mean_quals = np.array([q.mean() for q in qual_arrays])

    # Boolean mask of reads that pass
    passing = mean_quals >= min_score

    return [s for s, p in zip(sequences, passing) if p]
```

## Wrapping Up

Genomics can seem intimidating, but at its heart, it's string processing. DNA is text. Sequencing reads are text with quality scores. Alignment is pattern matching. Once you see it that way, you can apply all the programming skills you already have.

Start small: write functions to complement DNA, read FASTA files, filter by quality. Then build up to more complex analyses. The biology will make more sense as you handle real data.

The code in this article is intentionally simple. Production pipelines use optimized tools like BWA, STAR, and samtools. But understanding the fundamentals helps you know what those tools are actually doing, and when to write custom code for your specific research questions.

---

*Questions about genomics, Python, or bioinformatics? Feel free to reach out. I'm always happy to help people get started in this field.*
