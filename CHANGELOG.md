# CHANGELOG
All notable changes to this project will be documented in this file.
 
The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

## [Unreleased] 

### Added 

### Changed 

### Fixed

## [0.0.7] - 2025-09-09

### Added 
- linalg_tools.extract_lower_rank_solution: new algorithm to extract
  lower-rank solution from a solution to an SDP.

### Changed 
- sdp_solvers.solve_low_rank_sdp: now returns the actuall success status of IPOPT.

### Fixed

## [0.0.6] - 2025-09-01

### Added 
- sdp_solvers.solve_low_rank_sdp: expose options of ipopt.

### Fixed
- rank_project working with p>1.
- rank_project working with almost-symmetric matrices.
- rank_project working with non-symmetric matrices.

## [0.0.5] - 2025-05-27

### Added 
- This CHANGELOG, to keep track of new releases in the future. 
 
### Changed
- Remove unnecessary dependencies (pandas, pylgmath, gurobipy)
- Make some dependencies optional (plotly, sparseqr)
- More consistent use of install_requires, requirements.txt, environment.yml
