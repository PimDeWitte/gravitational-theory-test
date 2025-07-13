# 📂 Open‑Access Datasets for the Compression‑Hypothesis Programme  

| # | Dataset (click to open) | Enables experiments | What you get (typical volume) | Caveats / first‑steps |
|---|-------------------------|---------------------|------------------------------|----------------------|
| 1 | [Cassini Radio‑Science ODF / TNF](https://pds.nasa.gov/ds-view/pds/viewProfile.jsp?dsid=CO-SS-RSS-1-SCC8-V1.0) | S‑1, S‑3 | X‑/Ka‑band Doppler & range, 1997‑2017 (~40 GB) | Filter solar‑plasma noise; convert TNF→ODF :contentReference[oaicite:0]{index=0} |
| 2 | [NAIF SPICE kernel repository](https://naif.jpl.nasa.gov/naif/data.html) | S‑1…S‑5, P‑1…P‑5 | Trajectory/geometry files (GB‑scale) | Always load latest `naif*.tls` leapsecond file :contentReference[oaicite:1]{index=1} |
| 3 | [INPOP21a ephemeris pack](https://www.imcce.fr/recherche/equipes/asd/inpop/download21a) | S‑4 | Binary/TXT/SPICE planetary ephemerides (60 MB–600 MB) | Use CALCEPH‑Python or SPICE reader :contentReference[oaicite:2]{index=2} |
| 4 | [ILRS Lunar‑Laser‑Ranging normal points](https://cddis.nasa.gov/Data_and_Derived_Products/SLR/Normal_point_data.html) | S‑2 | Daily shot‑averaged LLR points (≈200 MB) | Apply retro‑reflector thermal model :contentReference[oaicite:3]{index=3} |
| 5 | [MESSENGER Radio‑Science](https://pds-geosciences.wustl.edu/missions/messenger/rs.htm) | S‑3 | Raw Doppler/range + gravity models (6 GB) | Weak signal near periapsis :contentReference[oaicite:4]{index=4} |
| 6 | [NANOGrav 15‑year timing data](https://zenodo.org/records/8423265) | P‑3, D1 | Narrow/wide‑band TOAs & solutions (2 GB) | Install `enterprise` + TEMPO2 :contentReference[oaicite:5]{index=5} |
| 7 | [EPTA Data‑Release 2](https://www.epta.eu.org/epta-dr2.html) | P‑4 | 25 MSP timing sets (0.3 GB) | Clock models differ from NANOGrav :contentReference[oaicite:6]{index=6} |
| 8 | [LIGO/Virgo O3 strain (GWOSC)](https://gwosc.org/O3/o3_details/) | P‑5, M‑1 | Calibrated h(t) segments, 2019‑20 (≤12 TB) | Down‑select with `gwpy` before grab :contentReference[oaicite:7]{index=7} |
| 9 | [GW170817 posterior samples](https://gwosc.org/eventapi/html/GWTC-1-confident/GW170817/v3/) | M‑1 | HDF5 chains (~20 MB) | Match calibration version to strain :contentReference[oaicite:8]{index=8} |
|10 | [Fermi‑GBM GRB 170817A TTE/CSPEC](https://gcn.gsfc.nasa.gov/other/G298048.gcn3) | M‑1 | 128‑channel photon events (~200 MB) | Align trigger with LIGO GPS time :contentReference[oaicite:9]{index=9} |
|11 | [SN 1987A neutrino tables (Kam/IMB)](https://arxiv.org/pdf/2307.03549) | M‑2 | 29 event times & energies (<1 MB) | UTC leap‑second audit needed :contentReference[oaicite:10]{index=10} |
|12 | [CHIME/FRB Catalogue v1](https://www.chime-frb.ca/catalog) | M‑3 | 536 burst dynamic spectra (80 GB) | Host‑DM uncertainties dominate :contentReference[oaicite:11]{index=11} |
|13 | [XMM‑Newton Science Archive (XSA)](https://nxsa.esac.esa.int/) | AGN‑1, AGN‑3 | EPIC event lists (GB each target) | Filter high‑background intervals :contentReference[oaicite:12]{index=12} |
|14 | [NICER GX 339‑4 set (HEASARC)](https://heasarc.gsfc.nasa.gov/cgi-bin/db-perl/W3Browse/w3table.pl?tablehead=name%3Dnicerscience) | AGN‑2 | 0.05‑100 keV events, 2021‑22 (~50 GB) | Gain‑correction script mandatory  |
|15 | [Water‑tank horizon videos (Zenodo)](https://zenodo.org/record/15223412) | Lab‑1 | 1 k fps stereo MP4 & crest CSV (<20 GB) | Embed frame‑time metadata :contentReference[oaicite:14]{index=14} |
|16 | [Fiber‑optic event‑horizon spectra (Supplement)](https://www.science.org/doi/abs/10.1126/science.1153625) | Lab‑2 | MATLAB spectra & pump params (100 MB) | Convert to Δt via FFT :contentReference[oaicite:15]{index=15} |
|17 | [Wee‑g MEMS gravimeter project](https://wee-g.com/) | Lab‑3, Field‑1 | STL CAD, PCB GERBER, sample tide data | Long‑period drift calibration :contentReference[oaicite:16]{index=16} |
|18 | [OMEGA VISAR shot database†](https://www.lle.rochester.edu/media/publications/presentations/documents/APS07/Boehly_APS07.pdf) | Lab‑7 | Shock‑timing ASCII + streak imgs (2 GB) | Approval needed; IDL pipeline :contentReference[oaicite:17]{index=17} |
|19 | [gMeterPy gravity‑processing code](https://github.com/opengrav/gmeterpy) | Lab‑3, Field‑1 | Python lib (few MB) | Install `pandas`, `pyproj`, etc. :contentReference[oaicite:18]{index=18} |
|20 | [InterRidge Hydrothermal Vent DB](https://usinterridge.org/vents-database/) | Field‑1 | JSON of active vents (<5 MB) | No gravity values—need your survey :contentReference[oaicite:19]{index=19} |
|21 | [TAG‑field AUV gravimetry report (WHOI)](https://www2.whoi.edu/site/tag/) | Field‑1 | Simulated 1 mGal grids (200 MB PDF+MAT) | Model, not measurement (baseline) :contentReference[oaicite:20]{index=20} |
|22 | [Glasgow MEMS tide demo data](https://www.gla.ac.uk/research/beacons/nanoquantum/wee-gglasgowsgravimeter/) | Crowd‑1 | 4‑day Earth‑tide CSV (15 MB) | Good QC template :contentReference[oaicite:21]{index=21} |
|23 | [UCI HAR smartphone accelerometer set](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones) | Crowd‑1 | 561‑feature accel logs (200 MB) | Use for bias modelling only  |
|24 | [GRACE hybrid mass‑anomaly grids](https://www.nature.com/articles/s41597-023-02887-5) | Context | Monthly 1° Δg (1994‑2021, 5 GB) | Re‑grid to μGal before compare :contentReference[oaicite:23]{index=23} |
|25 | [Hydrodynamic sloshing tank dataset](https://zenodo.org/records/15223412) | Lab‑1 control | Free‑surface elevation CSV & video | Not a horizon flow but good testbed :contentReference[oaicite:24]{index=24} |
|26 | [Juno DSN range‑rate archive](https://naif.jpl.nasa.gov/pub/naif/JUNO/kernels/) | S‑5 | Ka/X‑band residuals 2016‑21 (10 GB) | fewer Ka files; merge tracks :contentReference[oaicite:25]{index=25} |
|27 | [INPOP21a SPICE kernels](https://www.imcce.fr/recherche/equipes/asd/inpop/download21a) | S‑4, P‑1 | SPK format variant (50 MB) | Body IDs differ from JPL set :contentReference[oaicite:26]{index=26} |
|28 | [CAIXA AGN catalogue](https://heasarc.gsfc.nasa.gov/w3browse/all/caixa.html) | AGN‑3 | 156 AGN spectra & metadata | Use SAS v21 for re‑extraction :contentReference[oaicite:27]{index=27} |
|29 | [Kamiokande/IMB event scans (SN 1987A)](https://cds.cern.ch/record/177295/files/198705321.pdf) | M‑2 | PDF tables (<1 MB) | OCR to CSV before use :contentReference[oaicite:28]{index=28} |
|30 | [`planetmapper.kernel_downloader`](https://planetmapper.readthedocs.io/en/stable/kernel_downloader.html) | All SPICE steps | Python helper (<1 MB) | Automates bulk kernel grabs :contentReference[oaicite:29]{index=29} |

† OMEGA VISAR data are request‑only; the PDF link above includes contact details.  

---

# 🖥️ Computational‑Only Test Suite  

| Test ID | Scientific goal | Primary datasets | Toolchain & tips | Est. GPU/CPU time | Pass / fail criterion |
|---------|-----------------|------------------|------------------|-------------------|----------------------|
| S‑1 | Fit PPN‑γ to Cassini range + Doppler | 1, 2 | `spiceyPy`, `poliastro`, least‑sq on residuals | 4 CPU‑h | |γ‑1| < 2.3 × 10⁻⁵ & loss ≤ GR |
| S‑2 | Fit PPN‑β via LLR | 2, 4 | `lmfit`, Moon libration model | 2 CPU‑h | |β‑1| < 1 × 10⁻⁴ |
| S‑3 | Mercury Shapiro delay test | 2, 3, 5 | Integrate light‑time via `spice` | 1 CPU‑h | Residuals ≤ 0.3 m |
| S‑4 | Global ephemeris χ² | 2, 3 | Patch INPOP → alt‑metric kernel | 8 CPU‑h | Δχ² ≤ 0 relative to INPOP |
| S‑5 | Juno conjunction delay | 2, 26 | Simple two‑way light‑time calc | 0.5 CPU‑h | Residuals within fit noise |
| P‑1 | Binary pulsar B1913+16 PK fit | 2, 6 | TEMPO2, `libstempo` | 1 CPU‑h | Energy‑loss matches GR 0.2 % |
| P‑2 | Double pulsar full PK fit | 2, 6 | TEMPO2 + MCMC | 4 GPU‑h | Shapiro s within 1σ data |
| P‑3 | NANOGrav PTA “info‑noise” | 6 | `enterprise`, `bilby` sampler | 12 GPU‑h | Non‑zero latency slope OR limit |
| P‑4 | EPTA cross‑array check | 7 | Same as P‑3 | 8 GPU‑h | Consistent sign with P‑3 |
| P‑5 | Ring‑down residuals (GWTC‑2/3) | 8 | `pyRing` w/ extra τ param | 6 GPU‑h | Common latency τ across events |
| M‑1 | GW170817 → GRB delay | 9, 10 | Joint Bayesian (jet + τ) | 2 CPU‑h | τ posterior excludes 0 or not |
| M‑2 | SN 1987A ν vs optical | 11 | Simple Δt fit incl. shock travel | minutes | Extra ≥1 h? |
| M‑3 | FRB dispersion excess | 12 | Hierarchical model in `pymc` | 24 GPU‑h | k (time ∝ entropy) 5 σ |
| M‑4 | Strong‑lensing forecast | 2, 30 | Monte‑Carlo lens sims | 2 CPU‑h | Detectable degeneracy breaking |
| AGN‑1 | 1H0707 rev‑lag vs flux | 13 | `stingray`, cross‑correlator | 1 GPU‑h | Lag grows with flux entropy |
| AGN‑2 | GX 339‑4 NICER lag scan | 14 | Same tools | 2 GPU‑h | Positive lag‑luminosity slope |
| AGN‑3 | 30 AGN meta‑regression | 13, 28 | `pandas`, OLS | 3 CPU‑h | Global significant slope |
| Lab‑1 | Water‑tank crest delay | 15, 25 | `opencv`, optic flow | 0.5 GPU‑h | Δt ∝ noise power >1 ms |
| Lab‑7 | Shock‑timing VISAR re‑fit | 18 | IDL→CSV, then Python | 1 GPU‑h | Extra 50 ps lag at hi‑entropy |

*(All other Lab/Field rows require hardware and are thus excluded here.)*  
