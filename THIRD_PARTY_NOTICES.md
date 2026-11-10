# Third-party material

- `src/georc/templates/judge_repo.txt` is the judge prompt from `score.py` in
  [talrejamohit03/GeoRC](https://github.com/talrejamohit03/GeoRC), copied verbatim.
  Copyright the GeoRC authors, Apache License 2.0.
- `patches/authors_a7a1679.patch` modifies files of the same repository (commit a7a1679) and
  quotes them as diff context; the modified lines are marked "REPRO PATCH". Copyright the GeoRC
  authors, Apache License 2.0.
- `src/georc/templates/judge_paper.txt` and `src/georc/templates/gen_paper.txt` transcribe Figures 14
  and 21 of *GeoRC: A Benchmark for Geolocation Reasoning Chains* (Talreja et al., ACL 2026,
  arXiv:2601.21278).
- The GeoRC dataset (`mohit-talreja/GeoRC` on Hugging Face, Apache License 2.0) is downloaded at run
  time and is not included in this repository.
